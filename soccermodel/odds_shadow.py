"""
Odds shadow: the live prices of one league, corrected by what the closing market said about each club in
its EARLIER matches (`python -m soccermodel odds-shadow japan`, after `update`). Nothing from the match
being priced is used, and the published outputs are not touched.

    gap of a played match = market logit - model logit            (logit = ln(prob1 / prob2))
    club state            = Elo on those gaps: after each match both clubs move by k x (gap - expected gap)
    shadow price          = live price tilted by a x live logit + b x (state home - state away)

The model logit of a played match is its pre-match ratings priced with today's league parameters, i.e. the
way the live fixtures are priced. (a, b) are refitted every run on the league's matches with odds.
Backtest (J1, seasons 2021+, walk-forward): 1X2 log loss -0.8% v the live engine; docs/PIPELINE.md.

Config (`odds_shadow:` in the country's config.yaml):
    league: j1                       key under `leagues`
    file: odds/JPN.csv               closing odds of played matches, football-data.co.uk "new" layout
    url: https://...                 where --download refreshes that file from
    columns: [AvgCH, AvgCD, AvgCA]   home / draw / away closing odds
    names: {Chiba: JEF United}       odds-file club names -> names used here
    k: 0.3
    train_from: '2016'               first season used to fit (a, b)

Everything goes to output/<country>/odds_shadow/:
  README.md   summary: score so far, club states, largest price changes
  prices.csv  every unplayed fixture of the league: live v shadow 1X2 probabilities and odds, club states
  states.csv  every club's state (logit; 0.1 is about 17 Elo points) and its last match with odds
  log.csv     the last live and shadow price before each match was played, kept across runs
  score.csv   1X2 log loss of live, shadow and (where the odds are in) closing market on logged matches
"""

import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import odds as odds_mod
from .data import DataError, load_config, load_initial_elo, load_results
from .pipeline import league_pred_params, rate
from .poisson import davidson, nu_from_draw_rate
from .shadow import KEY, PROBS, _read, _table, score, update_log

MARKET = ['mk1', 'mkX', 'mk2']


def load_odds(path: Path, columns, names: dict) -> pd.DataFrame:
    """date, home, away and the market's 1X2 probabilities (margin removed proportionally)."""
    o = pd.read_csv(path, encoding='utf-8-sig')
    need = ['Date', 'Home', 'Away', *columns]
    missing = [c for c in need if c not in o.columns]
    if missing:
        raise DataError(f"{path.name}: missing columns {missing}")
    o = o.dropna(subset=list(columns))
    inv = 1 / o[list(columns)].to_numpy(float)
    out = pd.DataFrame({'date': pd.to_datetime(o['Date'], dayfirst=True),
                        'home': o['Home'].str.strip().replace(names), 'away': o['Away'].str.strip().replace(names)})
    out[MARKET] = inv / inv.sum(axis=1, keepdims=True)
    return out.sort_values('date', kind='stable').reset_index(drop=True)


def download(url: str, path: Path, columns) -> int:
    """Replace the odds file with a fresh copy, unless the copy is unreadable or shorter. Returns rows added."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=60).read()
    tmp = path.with_suffix('.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    try:
        new = len(load_odds(tmp, columns, {}))
        old = len(load_odds(path, columns, {})) if path.exists() else 0
        if new < old:
            raise DataError(f"downloaded odds file has {new} priced rows, the current one {old}: kept the current one")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return new - old


def attach_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """matches (date, home, away, ...) with the market columns of the same pairing within a day (NaN if none)."""
    m = matches.reset_index(drop=True).reset_index().sort_values('date', kind='stable')
    j = pd.merge_asof(m, odds, on='date', by=['home', 'away'], direction='nearest', tolerance=pd.Timedelta(days=1))
    return j.sort_values('index', kind='stable').drop(columns='index').reset_index(drop=True)


def club_states(home, away, gap, k: float):
    """
    Elo on gaps. Returns (pre-match state(home) - state(away) of every row, {club: state}).
    A row whose gap is NaN (no odds) only reads the states.
    """
    state, feature = {}, np.zeros(len(home))
    for i, (h, a) in enumerate(zip(home, away)):
        sh, sa = state.get(h, 0.0), state.get(a, 0.0)
        feature[i] = sh - sa
        if not np.isnan(gap[i]):
            e = gap[i] - feature[i]
            state[h], state[a] = sh + k * e, sa - k * e
    return feature, state


def tilt(p1, px, p2, shift):
    """Move `shift` (logit) from the away win to the home win, the draw keeping its relative weight."""
    a, b = p1 * np.exp(shift / 2), p2 * np.exp(-shift / 2)
    total = a + px + b
    return a / total, px / total, b / total


def fit_tilt(p1, px, p2, feature, outcome):
    """(a, b) maximising the likelihood of outcomes (0 home, 1 draw, 2 away) under tilt(a*logit + b*feature)."""
    z = np.log(p1 / p2)

    def nll(th):
        q = tilt(p1, px, p2, th[0] * z + th[1] * feature)
        return -np.log(np.choose(outcome, q)).sum()
    return minimize(nll, np.zeros(2), method='BFGS').x


def _loss(p, outcome):
    return -np.log(np.clip(np.asarray(p)[np.arange(len(outcome)), outcome], 1e-12, None))


def odds_shadow(country_dir: Path, live_dir: Path, fetch: bool = False, log=print) -> Path:
    cfg = load_config(country_dir)
    oc = cfg.get('odds_shadow')
    if not oc:
        raise DataError("config.yaml has no `odds_shadow:` section")
    lg = cfg['leagues'][oc['league']]
    tournament = lg['tournament']
    columns = oc.get('columns', ['AvgCH', 'AvgCD', 'AvgCA'])
    names = {**cfg.get('aliases', {}), **(oc.get('names') or {})}
    path = cfg['dir'] / oc['file']
    if fetch:
        try:
            log(f"Odds file refreshed: {download(oc['url'], path, columns):+d} priced matches")
        except Exception as e:      # the shadow still runs on the file it has
            log(f"WARNING: odds download failed ({e}); using the current file")
    odds = load_odds(path, columns, names)

    # ---- model price of every played match of the league, the way live fixtures are priced
    results = load_results(cfg)
    _, params, _, history, _, engine = rate(cfg, results, load_initial_elo(cfg))
    pred = league_pred_params(cfg, lg, params, history)
    h = history[history['tournament'] == tournament].sort_values('date', kind='stable')
    h = attach_odds(h[['season', 'date', 'home', 'away', 'home_elo', 'away_elo', 'hg_90', 'ag_90']], odds)
    known = set(h['home']) | set(h['away'])
    recent = odds[odds['date'] >= results.loc[results['season'] == cfg['season'], 'date'].min()]
    strangers = sorted((set(recent['home']) | set(recent['away'])) - known)      # this season's rows only
    if strangers:
        log(f"WARNING: odds-file clubs not in {tournament} results (add to odds_shadow.names): {strangers}")
    diff = (h['home_elo'] - h['away_elo']) * 400 / pred.get('scale', 400) + pred['hfa']
    p1, px, p2 = davidson(diff.to_numpy(float), nu_from_draw_rate(pred['draw_rate']))
    gap = (np.log(h['mk1'] / h['mk2']) - np.log(p1 / p2)).to_numpy(float)
    feature, state = club_states(h['home'].to_numpy(), h['away'].to_numpy(), gap, float(oc.get('k', 0.3)))
    outcome = np.where(h['hg_90'] > h['ag_90'], 0, np.where(h['hg_90'] < h['ag_90'], 2, 1))
    fit = (h['season'].astype(str) >= str(oc.get('train_from', ''))).to_numpy() & ~np.isnan(gap)
    a, b = fit_tilt(p1[fit], px[fit], p2[fit], feature[fit], outcome[fit])

    # ---- shadow price of the unplayed fixtures
    live = _read(live_dir / oc['league'] / 'predictions.csv')
    if live.empty:
        raise DataError(f"no live predictions in {live_dir / oc['league']}: run `update` first")
    season_res = results[(results['season'] == cfg['season']) & (results['tournament'] == tournament)]
    stale = pd.Series(list(zip(live['home'], live['away']))).isin(set(zip(season_res['home'], season_res['away'])))
    if stale.any():         # priced before these results were entered: the states already know their odds
        log(f"WARNING: {int(stale.sum())} live predictions are of matches already played (run `update` first); skipped")
        live = live[~stale.to_numpy()].reset_index(drop=True)
    sh, sa = live['home'].map(state).fillna(0.0), live['away'].map(state).fillna(0.0)
    lp = [live[c].to_numpy(float) for c in PROBS]
    q = tilt(*lp, a * np.log(lp[0] / lp[2]) + b * (sh - sa).to_numpy())
    prices = live[['date', 'home', 'away']].assign(competition=oc['league'])
    for c, live_p, shadow_p in zip(PROBS, lp, q):
        prices[f'{c}_live'], prices[f'{c}_shadow'] = live_p, np.round(shadow_p, 4)
    for c, o in zip(PROBS, ('odds1', 'oddsX', 'odds2')):
        prices[f'{o}_live'] = live[o]
        prices[f'{o}_shadow'] = odds_mod.price(prices[f'{c}_shadow'], 3, cfg['margins']['three_way'])
    prices['prob1_diff'] = (prices['prob1_shadow'] - prices['prob1_live']).round(4)
    prices['state_home'], prices['state_away'] = sh.round(3), sa.round(3)
    prices = prices[KEY + [c for c in prices.columns if c not in KEY]]

    last = h.dropna(subset=['mk1'])
    last = pd.concat([last[['date', 'home']].rename(columns={'home': 'team'}),
                      last[['date', 'away']].rename(columns={'away': 'team'})]).groupby('team')['date'].max()
    current = set(live['home']) | set(live['away'])
    states = pd.DataFrame({'team': list(state), 'state': [round(v, 3) for v in state.values()]})
    states['elo_points'] = (states['state'] * 400 / np.log(10)).round(0).astype(int)
    states['last_odds'] = states['team'].map(last).dt.strftime('%Y-%m-%d')
    states['in_league_now'] = states['team'].isin(current)
    states = states.sort_values(['state', 'team'], ascending=[False, True])

    # ---- log and score
    out = live_dir / 'odds_shadow'
    out.mkdir(parents=True, exist_ok=True)
    run = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    logged = update_log(_read(out / 'log.csv'), prices, run)
    scores = score(logged, results)
    played = logged.merge(results.assign(date=results['date'].dt.strftime('%Y-%m-%d'))[
        ['date', 'home', 'away', 'hg_90', 'ag_90']].drop_duplicates(['date', 'home', 'away']), on=['date', 'home', 'away'])
    if len(played):
        played = attach_odds(played.assign(date=pd.to_datetime(played['date'])), odds).dropna(subset=['mk1'])
    if len(played):         # the same comparison on the matches whose closing odds are in the file
        o = np.where(played['hg_90'] > played['ag_90'], 0, np.where(played['hg_90'] < played['ag_90'], 2, 1))
        scores = pd.concat([scores, pd.DataFrame([{
            'competition': 'with closing odds', 'matches': len(played),
            'live': round(_loss(played[[f'{c}_live' for c in PROBS]], o).mean(), 5),
            'shadow': round(_loss(played[[f'{c}_shadow' for c in PROBS]], o).mean(), 5),
            'market': round(_loss(played[MARKET], o).mean(), 5)}])], ignore_index=True)

    prices.to_csv(out / 'prices.csv', index=False)
    states.to_csv(out / 'states.csv', index=False)
    logged.to_csv(out / 'log.csv', index=False)
    scores.to_csv(out / 'score.csv', index=False)

    big = prices.reindex(prices['prob1_diff'].abs().sort_values(ascending=False).index).head(10)
    big = big[KEY[1:] + ['odds1_live', 'oddsX_live', 'odds2_live', 'odds1_shadow', 'oddsX_shadow', 'odds2_shadow',
                         'prob1_diff', 'state_home', 'state_away']]
    text = (f"# Odds shadow: {lg['name']} live prices corrected by past closing odds\n\n"
            f"Run {run} UTC. Live engine `{engine}`; nothing here changes the published outputs one folder up.\n\n"
            f"Closing odds to {odds['date'].max():%Y-%m-%d} ({int((~np.isnan(gap)).sum())} of {len(h)} "
            f"{tournament} matches priced by the market). Step k {oc.get('k', 0.3)}; tilt fitted on "
            f"{int(fit.sum())} matches: a {a:+.3f} (live logit), b {b:.3f} (club state difference).\n\n"
            f"## Score so far\n\nEach side's last price before kick-off, 1X2 log loss on 90-minute results "
            f"(negative change = shadow better; logging started with the first run).\n\n{_table(scores)}\n"
            f"## Club states\n\nPositive: the closing market has rated the club above the model in its recent "
            f"matches (logit; `elo_points` is the same in Elo points).\n\n"
            f"{_table(states[states['in_league_now']].drop(columns='in_league_now'))}\n"
            f"## Largest price changes, unplayed fixtures\n\n{_table(big)}")
    (out / 'README.md').write_text(text, encoding='utf-8', newline='\n')
    log(f"Odds shadow ({lg['name']}): {len(prices)} fixtures priced, "
        f"{int(scores['matches'].iloc[0]) if len(scores) else 0} logged matches scored -> {out}")
    return out
