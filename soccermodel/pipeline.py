"""The matchday update: data -> parameters -> Elo -> tilts -> match odds -> simulations -> report."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds as odds_mod
from .cup import CupSimulator, PenaltyModel, alive_from_results, bracket_problem, cup_rounds
from .data import load_config, load_fixture_file, load_initial_elo, load_results, validate
from .elo import ClubEloWeights, run_elo, season_lengths
from .goals import LambdaLookup, MatchModel, compute_tilts
from .hierarchical import run_hierarchical
from .league import (DEFAULT_TIEBREAK, PairSampler, check_tiebreak, league_table, market_table, season_outputs, simulate_final_group,
                     simulate_season)
from .params import TournamentParams, cup_phase_params, tournament_stats
from .poisson import calibrate_draw_rate, over_prob
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


def _predict(model: MatchModel, fixtures: pd.DataFrame, neutral_col: str = None, shifts: dict = None) -> pd.DataFrame:
    """shifts: team -> Elo points it loses in these prices (stakes adjustment for settled sides)."""
    shifts = shifts or {}
    rows = []
    for f in fixtures.itertuples(index=False):
        neutral = bool(getattr(f, neutral_col)) if neutral_col else False
        shift = shifts.get(f.away, 0.0) - shifts.get(f.home, 0.0)
        rows.append({'date': f.date, 'home': f.home, 'away': f.away, 'stakes_shift': shift,
                     **model.predict(f.home, f.away, neutral, shift)})
    cols = ['date', 'home', 'away', 'home_elo', 'away_elo', 'stakes_shift', 'elo_diff_adj', 'home_tilt', 'away_tilt',
            'prob1', 'probX', 'prob2', 'expg1', 'expg2']
    out = pd.DataFrame(rows, columns=cols)
    return out if shifts else out.drop(columns='stakes_shift')


def settled_teams(outrights: pd.DataFrame, progress: dict, stakes: dict) -> list:
    """
    Teams with nothing left to play for: every outright market already decided (each at most
    `decided` or at least 1 - `decided` in the simulations) once they have played `from` of their season.
    """
    decided = 100 * float(stakes.get('decided', 0.01))
    start = float(stakes.get('from', 0.6))
    pct = [c for c in outrights.columns if c.endswith(' %')]
    done = ((outrights[pct] <= decided) | (outrights[pct] >= 100 - decided)).all(axis=1)
    teams = outrights.loc[done, 'team']
    return sorted(t for t in teams if progress.get(t, 0.0) >= start)


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


def _natural(s: str):
    return (0, int(s)) if str(s).isdigit() else (1, str(s))


def _unplayed(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Fixture rows without a score here or in `results` (matched on home, away)."""
    played_keys = set(zip(results['home'], results['away']))
    keys = pd.Series(list(zip(fixtures['home'], fixtures['away'])), index=fixtures.index, dtype=object)
    return fixtures[~fixtures['played'] & ~keys.isin(played_keys)]


def _league_phases(lg: dict, n_teams: int):
    """`phases` as a list, or a dict keyed by group size (with an optional `default`)."""
    phases = lg.get('phases')
    if isinstance(phases, dict):
        phases = phases.get(n_teams, phases.get(str(n_teams), phases.get('default')))
    return [dict(p) for p in phases or []]


def _simulate_league_group(lg, fixtures, season_res, model, sampler, margins, n_sims, rng, warnings, shifts=None):
    """One table: the regular fixture list, then any position phases (see league.py)."""
    n_teams = len(set(fixtures['home']) | set(fixtures['away']))
    phases = _league_phases(lg, n_teams)
    phase_rows = fixtures['phase'].fillna('').astype(str).str.strip() if 'phase' in fixtures else \
        pd.Series('', index=fixtures.index)
    regular_fx = fixtures[phase_rows == '']
    regular_res = season_res
    phase_preds, note = [], None
    if (phase_rows != '').any():
        start = fixtures.loc[phase_rows != '', 'date'].min()
        regular_res = season_res[season_res['date'] < start] if pd.notna(start) else season_res
    for phase in phases:
        rows = fixtures[phase_rows == phase['name']]
        if len(rows):
            teams = set(rows['home']) | set(rows['away'])
            res = season_res[season_res['home'].isin(teams) & season_res['away'].isin(teams) &
                             ~season_res.index.isin(regular_res.index)]
            remaining = _price_matches(_predict(model, _unplayed(rows, res), shifts=shifts), margins)
            phase['known'] = {'teams': teams, 'results': res, 'remaining': remaining}
            phase_preds.append(remaining)
        elif note is None:
            note = 'after the fixture list: ' + ', '.join(
                f"{p['name']} {p['positions'][0]}-{p['positions'][1] % (n_teams + 1)} ({p.get('carry', 'full')} points, "
                f"{'double' if int(p.get('rounds', 2)) == 2 else 'single'} round robin)" for p in phases)

    teams = sorted(set(season_res['home']) | set(season_res['away']) | set(fixtures['home']) | set(fixtures['away']))
    tiebreak = lg.get('tiebreak', DEFAULT_TIEBREAK)
    for where, tb in [('tiebreak', tiebreak)] + [(f"phase {p['name']}", p.get('tiebreak', [])) for p in phases]:
        check_tiebreak(tb, f"{lg.get('name', '')} {where}")
    table = league_table(regular_res, teams, lg.get('adjustments'), tiebreak)
    for phase in phases:  # known phase members, best regular-season position first
        if 'known' in phase:
            phase['known']['teams'] = [t for t in table['team'] if t in phase['known']['teams']]
    remaining = _unplayed(regular_fx, regular_res)
    preds = _price_matches(_predict(model, remaining, shifts=shifts), margins)
    sim = simulate_season(table, preds, n_sims, rng, phases=phases, sampler=sampler, tiebreak=tiebreak,
                          results=regular_res)
    positions, market_probs = season_outputs(sim, lg.get('markets', {'Winner': [1, 1]}))
    frames = [p for p in (preds, *phase_preds) if len(p)]
    all_preds = pd.concat(frames, ignore_index=True) if len(frames) > 1 else (frames[0] if frames else preds)
    regular = None
    if sim['regular'] is not None and lg.get('regular_markets'):
        reg_positions, reg_markets = season_outputs(sim['regular'], lg['regular_markets'])
        regular = {'positions': reg_positions, 'outrights': _outright_prices(reg_markets, margins['outright'])}
    return {'table': table, 'preds': all_preds, 'positions': positions, 'sim': sim, 'phase_note': note,
            'regular': regular,
            'outrights': _outright_prices(market_probs, margins['outright'])}


def current_season_lengths(cfg: dict, results: pd.DataFrame) -> dict:
    """
    (team, tournament, season) -> matches each team plays in this season's leagues: its fixture-list
    rows, plus the expected second-phase matches while that phase isn't in the fixtures file yet
    (average over the phase groups for the team's table size). Used by the Elo season boost.
    """
    out = {}
    season = cfg['season']
    for lg in cfg['leagues'].values():
        fx = load_fixture_file(cfg['dir'] / lg['fixtures'], cfg['aliases'])
        groups = fx.groupby(fx['group'].astype(str)) if 'group' in fx else [(None, fx)]
        for _, g in groups:
            teams = pd.concat([g['home'], g['away']])
            n_teams = teams.nunique()
            extra = 0.0
            if not ('phase' in g and g['phase'].notna().any()):
                phases = _league_phases(lg, n_teams)
                sizes = []
                for p in phases:
                    lo, hi = p['positions']
                    hi = hi if hi > 0 else n_teams + hi + 1
                    sizes.append(hi - lo + 1)
                if sizes:
                    extra = sum(n * int(p.get('rounds', 2)) * (n - 1) for n, p in zip(sizes, phases)) / sum(sizes)
            for team, n in teams.value_counts().items():
                out[(team, lg['tournament'], season)] = int(round(n + extra))
    played = results[results['season'] == season]
    for key, n in season_lengths(played, {k[1] for k in out}).items():
        out[key] = max(out.get(key, 0), n)
    return out


def update(country_dir: Path, out_dir: Path, n_sims: int = None, log=print, engine: str = None) -> dict:
    """Matchday run into out_dir. engine: rating engine replacing the config's elo.engine (shadow runs)."""
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
    stats = tournament_stats(results, season, (cfg.get('prediction_params') or {}).get('hfa_halflife'))
    params = TournamentParams(cfg, stats)
    elo_cfg = cfg.get('elo', {})
    weights = ClubEloWeights(step=int(elo_cfg.get('lookup_step', 10)), cache_dir=CACHE_DIR,
                             lambda_match=elo_cfg.get('lambda_match', 'win_prob'))
    lengths = current_season_lengths(cfg, results)
    engine = engine or elo_cfg.get('engine', 'elo')
    if engine == 'hierarchical':
        ratings, history = run_hierarchical(results, initial_elo, params, weights, cfg['hierarchical'], lengths)
    elif engine == 'elo':
        ratings, history = run_elo(results, initial_elo, params, weights, lengths=lengths)
    else:
        raise ValueError(f"elo.engine must be elo or hierarchical, not {engine!r}")
    tilts = compute_tilts(results, params)
    log(f"Ratings ({engine}) and tilts updated")

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
        draw_model = cfg.get('prediction_params', {}).get('draw_model', 'league_rate')
        if draw_model == 'calibrated' and 'draw_rate' not in (lg.get('predict') or {}):
            past = history[history['tournament'] == tournament]
            pred_params['league_draw_rate'] = pred_params['draw_rate']
            pred_params['draw_rate'] = calibrate_draw_rate(
                (past['home_elo'] - past['away_elo']) * 400 / pred_params['scale'] + pred_params['hfa'],
                pred_params['draw_rate'])
        lookup = LambdaLookup(pred_params['draw_rate'], pred_params['total_goals'], lookup_step, CACHE_DIR)
        model = MatchModel(ratings, tilts, pred_params, lookup)

        fixtures = load_fixture_file(cfg['dir'] / lg['fixtures'], cfg['aliases'])
        season_res = results[(results['season'] == season) & (results['tournament'] == tournament)]
        sampler = PairSampler(sorted(set(fixtures['home']) | set(fixtures['away']) |
                                     set(season_res['home']) | set(season_res['away'])), model)
        if 'group' in fixtures:
            fixtures['group'] = fixtures['group'].astype(str)
            group_of = dict(zip(fixtures['home'], fixtures['group'])) | dict(zip(fixtures['away'], fixtures['group']))
            res_group = season_res['home'].map(group_of)
            groups = {g: (fixtures[fixtures['group'] == g], season_res[res_group == g])
                      for g in sorted(fixtures['group'].unique(), key=_natural)}
        else:
            groups = {None: (fixtures, season_res)}

        def simulate(shifts):
            parts, sims = [], {}
            for g, (g_fx, g_res) in groups.items():
                part = _simulate_league_group(lg, g_fx, g_res, model, sampler, margins, n_sims, rng,
                                              summary['warnings'], shifts)
                part['group'] = g
                if g is not None:
                    part['outrights'] = part['outrights'].assign(group=g)
                parts.append(part)
                sims[g] = part['sim']
            outrights = pd.concat([p['outrights'] for p in parts], ignore_index=True)
            finals = {}
            for fg in lg.get('final_groups', []) or []:
                fsim = simulate_final_group(sims, fg, sampler, n_sims, rng)
                fprobs = market_table(fsim['teams'], fsim['position_probs'], fg.get('markets', {'Winner': [1, 1]}))
                finals[fg['name']] = fsim
                for m in fprobs.columns[1:]:
                    yes, no = odds_mod.yes_no(fprobs[m], margins['outright'])
                    cols = pd.DataFrame({'team': fprobs['team'], f'{m} %': (100 * fprobs[m]).round(1),
                                         f'{m} Yes': yes, f'{m} No': no})
                    outrights = outrights.merge(cols, on='team', how='left', suffixes=('', '_dup'))
                    for c in (f'{m} %', f'{m} Yes', f'{m} No'):  # same market from North and South
                        if f'{c}_dup' in outrights:
                            outrights[c] = outrights[c].fillna(outrights.pop(f'{c}_dup'))
            if finals:
                pct = [c for c in outrights.columns if c.endswith(' %')]
                outrights[pct] = outrights[pct].fillna(0.0)
            return parts, outrights

        # stakes: price the remaining matches of sides with nothing left to play for (decided in a
        # first simulation) with an Elo penalty, then simulate again with those prices
        parts, outrights = simulate({})
        settled = []
        if lg.get('stakes'):
            progress = {}
            for team in set(fixtures['home']) | set(fixtures['away']):
                n = ((season_res['home'] == team) | (season_res['away'] == team)).sum()
                progress[team] = n / lengths.get((team, tournament, season), max(n, 1))
            settled = settled_teams(outrights, progress, lg['stakes'])
            if settled:
                shift = float(lg['stakes'].get('shift', 70))
                parts, outrights = simulate({t: shift for t in settled})

        def joined(name):
            frames = [p[name].assign(group=p['group']) if p['group'] is not None else p[name] for p in parts]
            df = pd.concat(frames, ignore_index=True)
            return df[['group'] + [c for c in df.columns if c != 'group']] if 'group' in df else df

        table, preds, positions = joined('table'), joined('preds'), joined('positions')
        if 'group' in outrights:
            outrights = outrights[['group'] + [c for c in outrights.columns if c != 'group']]
        lg_dir = out_dir / key
        lg_dir.mkdir(exist_ok=True)
        table.to_csv(lg_dir / 'table.csv', index=False)
        _round_preds(preds).to_csv(lg_dir / 'predictions.csv', index=False)
        positions.round(4).to_csv(lg_dir / 'positions.csv', index=False)
        outrights.to_csv(lg_dir / 'outrights.csv', index=False)
        regular = parts[0]['regular'] if len(parts) == 1 else None
        if regular:
            regular['positions'].round(4).to_csv(lg_dir / 'regular_positions.csv', index=False)
            regular['outrights'].to_csv(lg_dir / 'regular_outrights.csv', index=False)
        summary['leagues'][key] = {'name': lg['name'], 'table': table,
                                   'next': _round_preds(_next_window(preds, results['date'].max())),
                                   'positions': positions, 'outrights': outrights, 'params': pred_params,
                                   'remaining': len(preds), 'grouped': 'group' in fixtures, 'regular': regular,
                                   'final_markets': list(dict.fromkeys(
                                       f'{m} %' for fg in lg.get('final_groups', []) or []
                                       for m in fg.get('markets', {'Winner': [1, 1]}))),
                                   'phases': [p['phase_note'] for p in parts if p['phase_note']][:1],
                                   'settled': settled, 'stakes': lg.get('stakes')}
        log(f"{lg['name']}: {len(season_res)} played, {len(preds)} to play"
            f"{f' in {len(groups)} groups' if len(groups) > 1 else ''}, simulated {n_sims}x")

    # ---- cups
    penalties = PenaltyModel(cfg.get('penalties'), cfg['dir'])
    cup_method = (cfg.get('cup_params', {}) or {}).get('method', 'first_round')
    cup_fit_tables = []
    for key, cup in cfg['cups'].items():
        overrides = cup.get('predict') or {}
        if cup_method == 'anchored':
            phases, fit_table = cup_phase_params(history, cfg, cup)
            cup_fit_tables.append(fit_table)
            phase_params = {ph: {**v, **{k: float(x) for k, x in overrides.items()}} for ph, v in phases.items()}
            round_phase = {r['name']: 'early' if r['tournament'] in cup.get('early', []) else 'late'
                           for r in cup_rounds(cup)}
        else:  # original behaviour: one parameter set from the first remaining round's label
            phase_params = {'all': params.prediction(cup_rounds(cup)[0]['tournament'], overrides)}
            round_phase = {r['name']: 'all' for r in cup_rounds(cup)}
        phase_models = {}
        for ph, pp in phase_params.items():
            lookup = LambdaLookup(pp['draw_rate'], pp['total_goals'], lookup_step, CACHE_DIR)
            phase_models[ph] = MatchModel(ratings, tilts, pp, lookup)
        models = {name: phase_models[ph] for name, ph in round_phase.items()}

        draw = load_fixture_file(cfg['dir'] / cup['draw'], cfg['aliases'])
        draw['tie'] = draw.get('tie', pd.Series(range(1, len(draw) + 1))).fillna(0).astype(int)
        draw['leg'] = draw.get('leg', 1)
        draw['leg'] = draw['leg'].fillna(1).astype(int)
        if 'winner' not in draw:
            draw['winner'] = np.nan
        # fill in scores that were entered in results.csv instead of the draw file
        labels = {r['name']: r['tournament'] for r in cup_rounds(cup)}
        res_scores = results[results['season'] == season].set_index(['tournament', 'home', 'away'])
        for i, row in draw[~draw['played']].iterrows():
            k = (labels.get(row['round']), row['home'], row['away'])
            if k in res_scores.index:
                r = res_scores.loc[k]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                draw.loc[i, ['hg', 'ag', 'hg_90', 'ag_90']] = r[['hg', 'ag', 'hg_90', 'ag_90']].values
                draw.loc[i, 'played'] = True

        first_round = cup_rounds(cup)[0]['name']
        first_rows = draw[draw['round'] == first_round]
        if len(first_rows):
            start = sorted(set(first_rows['home']) | set(first_rows['away']))
        else:
            start = alive_from_results(results, cup['results_prefix'], season, summary['warnings'])
        problem = None if cup.get('league_phase') else bracket_problem(start, cup['rounds'])
        if problem:
            summary['warnings'].append(f"{cup['name']}: {len(start)} teams in the first round; {problem}")

        sim = CupSimulator(cup, draw, start, models, rng, penalties)
        probs = sim.run(n_sims)
        outrights = _outright_prices(probs, margins['outright'])
        lp_table = None
        if cup.get('league_phase'):  # current league-phase table + projection and markets
            lp_played = draw[(draw['round'] == cup['league_phase']['round']) & draw['played']]
            lp_now = league_table(lp_played.astype({'hg': int, 'ag': int}), start)
            lp_sim = sim.league_phase
            lp_prices = _outright_prices(lp_sim.drop(columns=['exp_pts', 'exp_pos']), margins['outright'])
            lp_table = lp_now.merge(lp_sim[['team', 'exp_pts', 'exp_pos']], on='team').merge(lp_prices, on='team')

        upcoming = draw[~draw['played'] & ~draw['home'].str.startswith('W:') & ~draw['away'].str.startswith('W:')]
        neutral_rounds = {r['name'] for r in cup['rounds'] if r.get('neutral')}
        upcoming = upcoming.assign(neutral=upcoming['round'].isin(neutral_rounds))
        parts = [_predict(models[rnd], grp, 'neutral') for rnd, grp in upcoming.groupby('round', sort=False)
                 if rnd in models]
        preds = _price_matches(pd.concat(parts, ignore_index=True), margins) if parts else None

        cup_dir = out_dir / key
        cup_dir.mkdir(exist_ok=True)
        outrights.to_csv(cup_dir / 'outrights.csv', index=False)
        if lp_table is not None:
            lp_table.to_csv(cup_dir / 'league_phase.csv', index=False)
        if preds is not None:
            _round_preds(preds).to_csv(cup_dir / 'predictions.csv', index=False)
        summary['cups'][key] = {'name': cup['name'], 'start': start, 'outrights': outrights, 'league_phase': lp_table,
                                'next': _round_preds(preds) if preds is not None else None,
                                'params': phase_params, 'drawn': bool(len(first_rows))}
        log(f"{cup['name']}: {len(start)} teams left, simulated {n_sims}x")

    if cup_fit_tables:
        pd.concat(cup_fit_tables, ignore_index=True).round(3).to_csv(out_dir / 'cup_params.csv', index=False)
    summary['warnings'] = params.warnings + summary['warnings']
    write_report(summary, out_dir)
    for w in summary['warnings']:
        log(f"WARNING: {w}")
    return summary
