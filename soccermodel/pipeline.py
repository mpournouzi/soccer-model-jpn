"""The matchday update: data -> parameters -> Elo -> tilts -> match odds -> simulations -> report."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds as odds_mod
from .cup import CupSimulator, PenaltyModel, alive_from_results
from .data import load_config, load_fixture_file, load_initial_elo, load_results, validate
from .elo import ClubEloWeights, run_elo
from .goals import LambdaLookup, MatchModel, compute_tilts
from .league import league_table, season_outputs, simulate_season
from .params import TournamentParams, tournament_stats
from .poisson import over_prob
from .report import write_report

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / '.cache'
NEXT_WINDOW_DAYS = 6


def _price_matches(preds: pd.DataFrame, margins: dict) -> pd.DataFrame:
    preds = preds.copy()
    preds['over25'] = [over_prob(e1 + e2, 2.5) for e1, e2 in zip(preds['expg1'], preds['expg2'])]
    for col, p in (('odds1', 'prob1'), ('oddsX', 'probX'), ('odds2', 'prob2')):
        preds[col] = odds_mod.price(preds[p], 3, margins['three_way'])
    preds['oddsO25'] = odds_mod.price(preds['over25'], 2, margins['two_way'])
    preds['oddsU25'] = odds_mod.price(1 - preds['over25'], 2, margins['two_way'])
    return preds


def _predict(model: MatchModel, fixtures: pd.DataFrame, neutral_col: str = None) -> pd.DataFrame:
    rows = []
    for f in fixtures.itertuples(index=False):
        neutral = bool(getattr(f, neutral_col)) if neutral_col else False
        rows.append({'date': f.date, 'home': f.home, 'away': f.away, **model.predict(f.home, f.away, neutral)})
    cols = ['date', 'home', 'away', 'home_elo', 'away_elo', 'elo_diff_adj', 'home_tilt', 'away_tilt',
            'prob1', 'probX', 'prob2', 'expg1', 'expg2']
    return pd.DataFrame(rows, columns=cols)


def _outright_prices(market_probs: pd.DataFrame, margin: float) -> pd.DataFrame:
    out = market_probs[['team']].copy()
    for m in market_probs.columns[1:]:
        yes, no = odds_mod.yes_no(market_probs[m], margin)
        out[f'{m} %'] = (100 * market_probs[m]).round(1)
        out[f'{m} Yes'] = yes
        out[f'{m} No'] = no
    return out


def _next_window(preds: pd.DataFrame, latest_result) -> pd.DataFrame:
    """Matches in the next few days, plus any overdue (postponed) ones."""
    dated = preds.dropna(subset=['date'])
    upcoming = dated[dated['date'] > latest_result]
    if upcoming.empty:
        return dated
    start = upcoming['date'].min()
    return dated[dated['date'] <= start + timedelta(days=NEXT_WINDOW_DAYS)]


def _round_preds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    for c in ('home_elo', 'away_elo', 'elo_diff_adj'):
        df[c] = df[c].round(0).astype(int)
    for c in ('home_tilt', 'away_tilt', 'prob1', 'probX', 'prob2', 'over25'):
        df[c] = df[c].round(4)
    for c in ('expg1', 'expg2'):
        df[c] = df[c].round(3)
    return df


def update(country_dir: Path, out_dir: Path, n_sims: int = None, log=print) -> dict:
    cfg = load_config(country_dir)
    season = cfg['season']
    sim_cfg = cfg.get('simulation', {})
    n_sims = n_sims or int(sim_cfg.get('n_sims', 20000))
    rng = np.random.default_rng(sim_cfg.get('seed'))
    margins = cfg['margins']
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    results = load_results(cfg)
    initial_elo = load_initial_elo(cfg)
    validate(results, initial_elo, cfg)
    log(f"{len(results)} results loaded, latest {results['date'].max():%Y-%m-%d}")

    # ---- parameters, Elo, tilts
    stats = tournament_stats(results, season)
    params = TournamentParams(cfg, stats)
    elo_cfg = cfg.get('elo', {})
    weights = ClubEloWeights(step=int(elo_cfg.get('lookup_step', 10)), cache_dir=CACHE_DIR,
                             lambda_match=elo_cfg.get('lambda_match', 'win_prob'))
    ratings, history = run_elo(results, initial_elo, params, weights)
    tilts = compute_tilts(results, params)
    log("Elo ratings and tilts updated")

    last = pd.concat([history[['date', 'home', 'tournament']].rename(columns={'home': 'team'}),
                      history[['date', 'away', 'tournament']].rename(columns={'away': 'team'})])
    last = last.sort_values('date').groupby('team').last()
    ratings_df = pd.DataFrame({'team': list(ratings), 'elo': [round(v) for v in ratings.values()]})
    ratings_df['tilt'] = ratings_df['team'].map(tilts).fillna(1.0).round(4)
    ratings_df['last_match'] = ratings_df['team'].map(last['date']).dt.strftime('%Y-%m-%d')
    ratings_df['last_tournament'] = ratings_df['team'].map(last['tournament'])
    ratings_df = ratings_df.dropna(subset=['last_match']).sort_values(['elo', 'team'], ascending=[False, True])
    ratings_df.to_csv(out_dir / 'ratings.csv', index=False)

    keep = [c for c in stats.columns if not c.startswith('prior_') and
            c not in ('season_home_wins', 'season_away_wins', 'season_draws', 'season_goals')]
    params_df = stats[keep].reset_index().merge(params.elo_table(), on='tournament', how='outer')
    params_df.round(3).to_csv(out_dir / 'tournament_params.csv', index=False)

    summary = {'cfg': cfg, 'results': results, 'ratings': ratings_df, 'params': params_df,
               'leagues': {}, 'cups': {}, 'warnings': []}

    # ---- leagues
    lookup_step = int(elo_cfg.get('expg_lookup_step', 5))
    for key, lg in cfg['leagues'].items():
        tournament = lg['tournament']
        pred_params = params.prediction(tournament, lg.get('predict'))
        lookup = LambdaLookup(pred_params['draw_rate'], pred_params['total_goals'], lookup_step, CACHE_DIR)
        model = MatchModel(ratings, tilts, pred_params, lookup)

        fixtures = load_fixture_file(cfg['dir'] / lg['fixtures'], cfg['aliases'])
        season_res = results[(results['season'] == season) & (results['tournament'] == tournament)]
        played_keys = set(zip(season_res['home'], season_res['away']))
        remaining = fixtures[~fixtures['played'] &
                             ~pd.Series(list(zip(fixtures['home'], fixtures['away']))).isin(played_keys).values]

        teams = sorted(set(season_res['home']) | set(season_res['away']) | set(fixtures['home']) | set(fixtures['away']))
        table = league_table(season_res, teams)
        preds = _price_matches(_predict(model, remaining), margins)

        sim = simulate_season(table, preds, n_sims, rng)
        positions, market_probs = season_outputs(sim, lg.get('markets', {'Winner': [1, 1]}))
        outrights = _outright_prices(market_probs, margins['outright'])

        lg_dir = out_dir / key
        lg_dir.mkdir(exist_ok=True)
        table.to_csv(lg_dir / 'table.csv', index=False)
        _round_preds(preds).to_csv(lg_dir / 'predictions.csv', index=False)
        positions.round(4).to_csv(lg_dir / 'positions.csv', index=False)
        outrights.to_csv(lg_dir / 'outrights.csv', index=False)
        summary['leagues'][key] = {'name': lg['name'], 'table': table, 'next': _round_preds(_next_window(preds, results['date'].max())),
                                   'positions': positions, 'outrights': outrights, 'params': pred_params,
                                   'remaining': len(remaining)}
        log(f"{lg['name']}: {len(season_res)} played, {len(remaining)} to play, simulated {n_sims}x")

    # ---- cups
    penalties = PenaltyModel(cfg.get('penalties'), cfg['dir'])
    for key, cup in cfg['cups'].items():
        first_t = cup['rounds'][0]['tournament']
        pred_params = params.prediction(first_t, cup.get('predict'))
        lookup = LambdaLookup(pred_params['draw_rate'], pred_params['total_goals'], lookup_step, CACHE_DIR)
        model = MatchModel(ratings, tilts, pred_params, lookup)

        draw = load_fixture_file(cfg['dir'] / cup['draw'], cfg['aliases'])
        draw['tie'] = draw.get('tie', pd.Series(range(1, len(draw) + 1))).fillna(0).astype(int)
        draw['leg'] = draw.get('leg', 1)
        draw['leg'] = draw['leg'].fillna(1).astype(int)
        if 'winner' not in draw:
            draw['winner'] = np.nan
        # fill in scores that were entered in results.csv instead of the draw file
        labels = {r['name']: r['tournament'] for r in cup['rounds']}
        res_scores = results[results['season'] == season].set_index(['tournament', 'home', 'away'])
        for i, row in draw[~draw['played']].iterrows():
            k = (labels.get(row['round']), row['home'], row['away'])
            if k in res_scores.index:
                r = res_scores.loc[k]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                draw.loc[i, ['hg', 'ag', 'hg_90', 'ag_90']] = r[['hg', 'ag', 'hg_90', 'ag_90']].values
                draw.loc[i, 'played'] = True

        first_round = cup['rounds'][0]['name']
        first_rows = draw[draw['round'] == first_round]
        if len(first_rows):
            start = sorted(set(first_rows['home']) | set(first_rows['away']))
        else:
            start = alive_from_results(results, cup['results_prefix'], season, summary['warnings'])
        expected = 2 ** len(cup['rounds'])
        if len(start) != expected:
            summary['warnings'].append(f"{cup['name']}: {len(start)} teams alive but the configured "
                                       f"rounds need {expected} - check `rounds` in config.yaml")

        sim = CupSimulator(cup, draw, start, model, rng, penalties)
        probs = sim.run(n_sims)
        outrights = _outright_prices(probs, margins['outright'])

        upcoming = draw[~draw['played'] & ~draw['home'].str.startswith('W:') & ~draw['away'].str.startswith('W:')]
        neutral_rounds = {r['name'] for r in cup['rounds'] if r.get('neutral')}
        upcoming = upcoming.assign(neutral=upcoming['round'].isin(neutral_rounds))
        preds = _price_matches(_predict(model, upcoming, 'neutral'), margins) if len(upcoming) else None

        cup_dir = out_dir / key
        cup_dir.mkdir(exist_ok=True)
        outrights.to_csv(cup_dir / 'outrights.csv', index=False)
        if preds is not None:
            _round_preds(preds).to_csv(cup_dir / 'predictions.csv', index=False)
        summary['cups'][key] = {'name': cup['name'], 'start': start, 'outrights': outrights,
                                'next': _round_preds(preds) if preds is not None else None,
                                'params': pred_params, 'drawn': bool(len(first_rows))}
        log(f"{cup['name']}: {len(start)} teams left, simulated {n_sims}x")

    summary['warnings'] = params.warnings + summary['warnings']
    write_report(summary, out_dir)
    for w in summary['warnings']:
        log(f"WARNING: {w}")
    return summary
