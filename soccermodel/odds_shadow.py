"""
Odds shadow: live prices corrected by what the closing market said about each club in its EARLIER
matches (`python -m soccermodel odds-shadow japan [--download]`, after `update`). Nothing from the match
being priced is used, and the published outputs are not touched.

    gap of a played match = market logit - model logit            (logit = ln(prob1 / prob2))
    club state            = Elo on those gaps: after each match both clubs move by k x (gap - expected gap)
    shadow price          = live price tilted by a x live logit + b x (state home - state away)

The model logit of a played match is its pre-match ratings priced with today's parameters of its league,
i.e. the way live fixtures are priced. League matches of every division move the states (a promoted club
brings its state along); cup matches only read them.

Odds and what is kept. The repository is public and the API's odds may not be redistributed, so no API
price is stored: each run fetches the recent matches' closing odds, moves the states with the matches not
yet counted, and keeps only the results - every club's state (`states_file`) and the keys of the matches
already counted (`counted_file`, no prices). The J1 file from football-data.co.uk is free and stays in the
repo as before (it is preferred where both have a match). The tilt (a, b) per target is fitted offline on
the full history (soccer-model-odds tools/seed_odds_shadow.py, which also writes the first states) and set
in the config.

Backtest (walk-forward, seasons 2019+, base = live engines' backtest prices): 1X2 log loss J1 -0.57%,
J2 -0.92%, cups -0.48% (8 of 8 seasons each); J3 -0.22%, JFL none (not shadowed). docs/PIPELINE.md.

Config (`odds_shadow:` in the country's config.yaml):
    targets: {j1: [a, b], ...}         league or cup keys to shadow-price, with their tilt
    k: 0.3                             Elo step of the states
    max_age_days: 365                  a state older than this (no odds since) counts as 0
    state_tournaments: {J1 League: j1, ...}   tournaments whose odds move the states -> league key whose
                                       parameters price them
    states_file / counted_file         under the country folder
    football_data: {file, url, columns, names}
    api: {key_env, teams_file, leagues: [ids], lookback_days}

Everything goes to output/<country>/odds_shadow/:
  README.md   summary: score so far, club states, largest price changes
  prices.csv  every unplayed fixture of the targets: live v shadow 1X2 probabilities and odds, club states
  states.csv  every club's state (logit; 0.1 is about 17 Elo points) and its last match with odds
  log.csv     the last live and shadow price before each match was played, kept across runs
  score.csv   1X2 log loss of live and shadow (and the football-data closing market) on logged matches
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
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
API_BASE = 'https://api.5dollarfootballapi.com/v1'
OVERROUND = (1.03, 1.15)            # a real 1X2 line; outside = broken quote, ignored
COUNTED = ['tournament', 'date', 'home', 'away']


# ---------------------------------------------------------------- odds sources
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


def api_rows(fixtures: list, teams: dict) -> tuple:
    """API fixtures (with `include=odds`) -> (date, home, away, market columns) in our names, and the API
    clubs not in `teams` ({api team id: our name})."""
    rows, unknown = [], {}
    for fx in fixtures:
        x = ((fx.get('odds') or {}).get('1x2') or {}).get('closing') or {}
        h, a = fx['teams']['home'], fx['teams']['away']
        if not all(x.get(k) for k in ('home', 'draw', 'away')):
            continue
        names = [teams.get(str(t['id'])) for t in (h, a)]
        for t, n in zip((h, a), names):
            if n is None:
                unknown[str(t['id'])] = t['name']
        if None in names:
            continue
        inv = np.array([1 / x['home'], 1 / x['draw'], 1 / x['away']])
        if not OVERROUND[0] <= inv.sum() <= OVERROUND[1]:
            continue
        rows.append({'date': pd.Timestamp(fx['kickoff_utc'][:10]), 'home': names[0], 'away': names[1],
                     **dict(zip(MARKET, inv / inv.sum()))})
    return pd.DataFrame(rows, columns=['date', 'home', 'away', *MARKET]), unknown


def fetch_api(api: dict, teams: dict, key: str, log=print) -> pd.DataFrame:
    """Closing 1X2 of the finished matches of the last `lookback_days` in the API leagues. Kept in memory only."""
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - 86400 * int(api.get('lookback_days', 30))
    fixtures = []
    for lid in api['leagues']:
        page = 1
        while True:
            q = urllib.parse.urlencode({'start_time': start, 'end_time': end, 'status': 'finished', 'include': 'odds',
                                        'per_page': 50, 'page': page})
            req = urllib.request.Request(f"{API_BASE}/leagues/{lid}/fixtures?{q}",
                                         headers={'Authorization': f'Bearer {key}', 'User-Agent': 'soccer-model'})
            try:
                body = json.loads(urllib.request.urlopen(req, timeout=60).read())
            except urllib.error.HTTPError as e:
                log(f"WARNING: odds API league {lid}: HTTP {e.code} ({json.loads(e.read() or b'{}').get('error', {}).get('code')})")
                break
            fixtures += body.get('data', [])
            if not body.get('pagination', {}).get('has_more'):
                break
            page += 1
    rows, unknown = api_rows(fixtures, teams)
    if unknown:
        log(f"WARNING: odds API clubs without a name in the teams file (add them): {unknown}")
    return rows


def attach_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """matches (date, home, away, ...) with the market columns of the same pairing within a day (NaN if none)."""
    m = matches.reset_index(drop=True).reset_index().sort_values('date', kind='stable')
    j = pd.merge_asof(m, odds.sort_values('date', kind='stable'), on='date', by=['home', 'away'],
                      direction='nearest', tolerance=pd.Timedelta(days=1))
    return j.sort_values('index', kind='stable').drop(columns='index').reset_index(drop=True)


# ---------------------------------------------------------------- states and tilt
def club_states(home, away, gap, k: float, state: dict = None):
    """
    Elo on gaps. Returns (pre-match state(home) - state(away) of every row, {club: state}).
    A row whose gap is NaN (no odds) only reads the states. `state` (updated in place) continues earlier states.
    """
    state = {} if state is None else state
    feature = np.zeros(len(home))
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


def model_prices(cfg, params, history: pd.DataFrame, league_of: dict) -> pd.DataFrame:
    """Pre-match 1X2 of the played matches in `league_of`'s tournaments, priced with today's parameters of
    the league each tournament maps to."""
    h = history[history['tournament'].isin(league_of)].copy()
    h[['p1', 'px', 'p2']] = np.nan
    for t, key in league_of.items():
        sel = (h['tournament'] == t).to_numpy()
        if not sel.any():
            continue
        pred = league_pred_params(cfg, cfg['leagues'][key], params, history)
        diff = (h.loc[sel, 'home_elo'] - h.loc[sel, 'away_elo']) * 400 / pred.get('scale', 400) + pred['hfa']
        h.loc[sel, ['p1', 'px', 'p2']] = np.column_stack(davidson(diff.to_numpy(float), nu_from_draw_rate(pred['draw_rate'])))
    return h.sort_values('date', kind='stable').reset_index(drop=True)


def read_states(path: Path) -> tuple:
    if not path.exists():
        return {}, {}
    s = pd.read_csv(path)
    return dict(zip(s['team'], s['state'])), dict(zip(s['team'], pd.to_datetime(s['last_odds'])))


def write_states(path: Path, state: dict, last: dict):
    s = pd.DataFrame({'team': list(state), 'state': [round(v, 6) for v in state.values()],
                      'last_odds': [last[t].strftime('%Y-%m-%d') if t in last else '' for t in state]})
    s.sort_values('team').to_csv(path, index=False)


def count_new(h: pd.DataFrame, counted: pd.DataFrame, state: dict, last: dict, k: float) -> pd.DataFrame:
    """Move the states with the matches of h (model prices + market columns) that have odds and are not yet in
    `counted`, oldest first. Returns their keys (to add to `counted`)."""
    keys = counted.assign(date=pd.to_datetime(counted['date'])).set_index(COUNTED).index if len(counted) else None
    new = h.dropna(subset=['mk1'])
    if keys is not None:
        new = new[~new.set_index(COUNTED).index.isin(keys)]
    if new.empty:
        return new[COUNTED]
    gap = (np.log(new['mk1'] / new['mk2']) - np.log(new['p1'] / new['p2'])).to_numpy(float)
    club_states(new['home'].to_numpy(), new['away'].to_numpy(), gap, k, state)
    for d, ht, at in zip(new['date'], new['home'], new['away']):
        last[ht] = max(last.get(ht, d), d)
        last[at] = max(last.get(at, d), d)
    return new[COUNTED]


def current_states(state: dict, last: dict, today, max_age: int) -> dict:
    cut = pd.Timestamp(today) - pd.Timedelta(days=max_age)
    return {t: (v if t in last and last[t] >= cut else 0.0) for t, v in state.items()}


def _loss(p, outcome):
    return -np.log(np.clip(np.asarray(p)[np.arange(len(outcome)), outcome], 1e-12, None))


# ---------------------------------------------------------------- run
def odds_shadow(country_dir: Path, live_dir: Path, fetch: bool = False, log=print) -> Path:
    cfg = load_config(country_dir)
    oc = cfg.get('odds_shadow')
    if not oc:
        raise DataError("config.yaml has no `odds_shadow:` section")
    fd, api = oc.get('football_data') or {}, oc.get('api') or {}
    league_of = oc['state_tournaments']
    k = float(oc.get('k', 0.3))

    # ---- odds: the football-data file (kept) and the API's recent matches (not kept)
    odds = pd.DataFrame(columns=['date', 'home', 'away', *MARKET])
    if fd:
        path = cfg['dir'] / fd['file']
        if fetch:
            try:
                log(f"Odds file refreshed: {download(fd['url'], path, fd['columns']):+d} priced matches")
            except Exception as e:          # the shadow still runs on the file it has
                log(f"WARNING: odds download failed ({e}); using the current file")
        names = {**cfg.get('aliases', {}), **(fd.get('names') or {})}
        odds = load_odds(path, fd['columns'], names)
    fd_odds = odds
    if fetch and api:
        key = os.environ.get(api.get('key_env', 'FDFA_KEY'))
        if not key:
            log(f"WARNING: no odds API key in ${api.get('key_env', 'FDFA_KEY')}; states move with football-data odds only")
        else:
            t = pd.read_csv(cfg['dir'] / api['teams_file'], dtype={'id': str})
            recent = fetch_api(api, dict(zip(t['id'], t['name'])), key, log)
            log(f"Odds API: {len(recent)} recent matches with closing odds")
            odds = pd.concat([odds, recent], ignore_index=True).drop_duplicates(['date', 'home', 'away'], keep='first')

    # ---- move the states with the league matches not yet counted
    results = load_results(cfg)
    _, params, _, history, _, engine = rate(cfg, results, load_initial_elo(cfg))
    h = attach_odds(model_prices(cfg, params, history, league_of), odds)
    states_path, counted_path = cfg['dir'] / oc['states_file'], cfg['dir'] / oc['counted_file']
    state, last = read_states(states_path)
    counted = _read(counted_path)
    added = count_new(h, counted, state, last, k)
    if len(added):
        counted = pd.concat([counted, added.assign(date=added['date'].dt.strftime('%Y-%m-%d'))], ignore_index=True)
        counted.to_csv(counted_path, index=False)
        write_states(states_path, state, last)
    today = results['date'].max()
    now = current_states(state, last, today, int(oc.get('max_age_days', 365)))
    log(f"Odds shadow states: {len(added)} new matches counted, {len(counted)} in all")

    # ---- shadow prices of the targets' unplayed fixtures
    played = set(zip(results['date'], results['home'], results['away']))
    parts = []
    for target, (a, b) in oc['targets'].items():
        live = _read(live_dir / target / 'predictions.csv')
        if live.empty:
            log(f"WARNING: no live predictions in {live_dir / target}; skipped")
            continue
        done = [(pd.Timestamp(d), hh, aa) in played for d, hh, aa in zip(live['date'], live['home'], live['away'])]
        if any(done):           # priced before these results were entered: run `update` first
            log(f"WARNING: {target}: {sum(done)} live predictions are of matches already played; skipped")
            live = live[~np.array(done)].reset_index(drop=True)
        sh, sa = live['home'].map(now).fillna(0.0), live['away'].map(now).fillna(0.0)
        lp = [live[c].to_numpy(float) for c in PROBS]
        q = tilt(*lp, a * np.log(lp[0] / lp[2]) + b * (sh - sa).to_numpy())
        p = live[['date', 'home', 'away']].assign(competition=target)
        for c, live_p, shadow_p in zip(PROBS, lp, q):
            p[f'{c}_live'], p[f'{c}_shadow'] = live_p, np.round(shadow_p, 4)
        for c, o in zip(PROBS, ('odds1', 'oddsX', 'odds2')):
            p[f'{o}_live'] = live[o]
            p[f'{o}_shadow'] = odds_mod.price(p[f'{c}_shadow'], 3, cfg['margins']['three_way'])
        p['prob1_diff'] = (p['prob1_shadow'] - p['prob1_live']).round(4)
        p['state_home'], p['state_away'] = sh.round(3), sa.round(3)
        parts.append(p)
    if not parts:
        raise DataError(f"no live predictions for any odds-shadow target in {live_dir}: run `update` first")
    prices = pd.concat(parts, ignore_index=True)
    prices = prices[KEY + [c for c in prices.columns if c not in KEY]]

    current = set(prices['home']) | set(prices['away'])
    states = pd.DataFrame({'team': list(now), 'state': [round(v, 3) for v in now.values()]})
    states['elo_points'] = (states['state'] * 400 / np.log(10)).round(0).astype(int)
    states['last_odds'] = states['team'].map(lambda t: last[t].strftime('%Y-%m-%d') if t in last else '')
    states['in_targets_now'] = states['team'].isin(current)
    states = states.sort_values(['state', 'team'], ascending=[False, True])

    # ---- log and score
    out = live_dir / 'odds_shadow'
    out.mkdir(parents=True, exist_ok=True)
    run = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    logged = update_log(_read(out / 'log.csv'), prices, run)
    scores = score(logged, results)
    if len(fd_odds) and len(logged):    # J1: the same comparison with the free closing odds, where they are in
        pl = logged.merge(results.assign(date=results['date'].dt.strftime('%Y-%m-%d'))[
            ['date', 'home', 'away', 'hg_90', 'ag_90']].drop_duplicates(['date', 'home', 'away']), on=['date', 'home', 'away'])
        if len(pl):
            pl = attach_odds(pl.assign(date=pd.to_datetime(pl['date'])), fd_odds).dropna(subset=['mk1'])
        if len(pl):
            o = np.where(pl['hg_90'] > pl['ag_90'], 0, np.where(pl['hg_90'] < pl['ag_90'], 2, 1))
            scores = pd.concat([scores, pd.DataFrame([{
                'competition': 'with free closing odds', 'matches': len(pl),
                'live': round(_loss(pl[[f'{c}_live' for c in PROBS]], o).mean(), 5),
                'shadow': round(_loss(pl[[f'{c}_shadow' for c in PROBS]], o).mean(), 5),
                'market': round(_loss(pl[MARKET], o).mean(), 5)}])], ignore_index=True)

    prices.to_csv(out / 'prices.csv', index=False)
    states.to_csv(out / 'states.csv', index=False)
    logged.to_csv(out / 'log.csv', index=False)
    scores.to_csv(out / 'score.csv', index=False)

    big = prices.reindex(prices['prob1_diff'].abs().sort_values(ascending=False).index).head(10)
    big = big[KEY + ['odds1_live', 'oddsX_live', 'odds2_live', 'odds1_shadow', 'oddsX_shadow', 'odds2_shadow',
                     'prob1_diff', 'state_home', 'state_away']]
    tilts = pd.DataFrame([{'target': t, 'a (live logit)': a, 'b (state difference)': b} for t, (a, b) in oc['targets'].items()])
    text = (f"# Odds shadow: live prices corrected by past closing odds\n\n"
            f"Run {run} UTC. Live engine `{engine}`; nothing here changes the published outputs one folder up.\n\n"
            f"States from {len(counted)} league matches with closing odds (step k {k}; a state unchanged for "
            f"{oc.get('max_age_days', 365)} days counts as 0). Tilts, fitted offline:\n\n{_table(tilts)}\n"
            f"## Score so far\n\nEach side's last price before kick-off, 1X2 log loss on 90-minute results "
            f"(negative change = shadow better; logging started with the first run).\n\n{_table(scores)}\n"
            f"## Club states\n\nPositive: the closing market has rated the club above the model in its recent "
            f"matches (logit; `elo_points` is the same in Elo points). Clubs with a shadowed fixture to come.\n\n"
            f"{_table(states[states['in_targets_now']].drop(columns='in_targets_now'))}\n"
            f"## Largest price changes, unplayed fixtures\n\n{_table(big)}")
    (out / 'README.md').write_text(text, encoding='utf-8', newline='\n')
    log(f"Odds shadow ({', '.join(oc['targets'])}): {len(prices)} fixtures priced, "
        f"{int(scores['matches'].iloc[0]) if len(scores) else 0} logged matches scored -> {out}")
    return out
