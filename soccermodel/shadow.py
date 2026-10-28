"""
Shadow run: price the same matchday with another rating engine and compare it with the live prices,
without touching the published outputs (`python -m soccermodel shadow greece`, after `update`).

Everything goes to output/<country>/shadow/:

  README.md     summary: score so far, biggest rating and price differences
  prices.csv    every unplayed league and cup fixture, live v shadow 1X2 probabilities and odds
  ratings.csv   every club's live and shadow rating
  outrights.csv every outright market, live v shadow probability (%)
  log.csv       the last prediction of both engines before each match was played: kept across runs, so
                once the results are in, both can be scored on exactly the same matches (score.csv)
  score.csv     1X2 log loss of both engines on the logged matches that have been played, by competition
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_config, load_results
from .pipeline import update

KEY = ['competition', 'date', 'home', 'away']
PROBS = ['prob1', 'probX', 'prob2']


def _competitions(cfg) -> list:
    return list(cfg['leagues']) + list(cfg['cups'])


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def compare_prices(live_dir: Path, shadow_dir: Path, comps: list) -> pd.DataFrame:
    out = []
    for comp in comps:
        live, shadow = _read(live_dir / comp / 'predictions.csv'), _read(shadow_dir / comp / 'predictions.csv')
        if live.empty or shadow.empty:
            continue
        cols = ['date', 'home', 'away', 'home_elo', 'away_elo'] + PROBS + ['odds1', 'oddsX', 'odds2']
        m = live[cols].merge(shadow[cols], on=['date', 'home', 'away'], suffixes=('_live', '_shadow'))
        out.append(m.assign(competition=comp))
    if not out:
        return pd.DataFrame(columns=KEY)
    df = pd.concat(out, ignore_index=True)
    df['prob1_diff'] = (df['prob1_shadow'] - df['prob1_live']).round(4)
    return df[KEY + [c for c in df.columns if c not in KEY]]


def compare_outrights(live_dir: Path, shadow_dir: Path, comps: list) -> pd.DataFrame:
    out = []
    for comp in comps:
        live, shadow = _read(live_dir / comp / 'outrights.csv'), _read(shadow_dir / comp / 'outrights.csv')
        if live.empty or shadow.empty:
            continue
        for market in [c[:-2] for c in live.columns if c.endswith(' %') and c in shadow.columns]:
            m = live[['team', f'{market} %']].merge(shadow[['team', f'{market} %']], on='team', suffixes=('_live', '_shadow'))
            m.columns = ['team', 'live %', 'shadow %']
            out.append(m.assign(competition=comp, market=market))
    if not out:
        return pd.DataFrame(columns=['competition', 'market', 'team', 'live %', 'shadow %', 'diff'])
    df = pd.concat(out, ignore_index=True)
    df['diff'] = (df['shadow %'] - df['live %']).round(1)
    return df[['competition', 'market', 'team', 'live %', 'shadow %', 'diff']]


def update_log(log: pd.DataFrame, prices: pd.DataFrame, run: str) -> pd.DataFrame:
    """Keep each fixture's latest prediction while it is unplayed (prices only lists unplayed fixtures)."""
    new = prices[KEY + [f'{p}_live' for p in PROBS] + [f'{p}_shadow' for p in PROBS]].assign(last_run=run)
    if log.empty:
        return new.assign(first_run=run)
    first = log.set_index(KEY)['first_run']
    new = new.assign(first_run=[first.get(tuple(k), run) for k in new[KEY].itertuples(index=False)])
    keep = log[~log.set_index(KEY).index.isin(new.set_index(KEY).index)]
    return pd.concat([keep, new], ignore_index=True).sort_values(KEY, kind='stable').reset_index(drop=True)


def score(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """1X2 log loss (90 minutes) of both engines on logged matches that have a result."""
    res = results.assign(date=results['date'].dt.strftime('%Y-%m-%d'))[['date', 'home', 'away', 'hg_90', 'ag_90']]
    m = log.merge(res.drop_duplicates(['date', 'home', 'away']), on=['date', 'home', 'away'])
    if m.empty:
        return pd.DataFrame(columns=['competition', 'matches', 'live', 'shadow', 'change %', 'se %'])
    o = np.where(m['hg_90'] > m['ag_90'], 0, np.where(m['hg_90'] < m['ag_90'], 2, 1))
    for side in ('live', 'shadow'):
        p = m[[f'{q}_{side}' for q in PROBS]].to_numpy()[np.arange(len(m)), o]
        m[side] = -np.log(np.clip(p, 1e-12, None))
    rows = []
    for comp, g in [('all', m)] + list(m.groupby('competition')):
        d = g['shadow'] - g['live']
        se = d.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        rows.append({'competition': comp, 'matches': len(g), 'live': round(g['live'].mean(), 5),
                     'shadow': round(g['shadow'].mean(), 5), 'change %': round(100 * d.mean() / g['live'].mean(), 2),
                     'se %': round(100 * se / g['live'].mean(), 2)})
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return '_none_\n'
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |\n|' + '---|' * len(df.columns) + '\n'
    return head + ''.join('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row) + ' |\n'
                          for row in df.itertuples(index=False))


def shadow(country_dir: Path, live_dir: Path, engine: str = 'hierarchical', n_sims: int = None, log=print) -> Path:
    cfg = load_config(country_dir)
    out = live_dir / 'shadow'
    out.mkdir(parents=True, exist_ok=True)
    comps = _competitions(cfg)
    run = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    with tempfile.TemporaryDirectory() as tmp:
        update(country_dir, Path(tmp), n_sims=n_sims, log=lambda m: None, engine=engine)
        tmp = Path(tmp)
        prices = compare_prices(live_dir, tmp, comps)
        outrights = compare_outrights(live_dir, tmp, comps)
        live_r = pd.read_csv(live_dir / 'ratings.csv')[['team', 'elo', 'last_match', 'last_tournament']]
        shadow_r = pd.read_csv(tmp / 'ratings.csv')[['team', 'elo']]
    ratings = live_r.merge(shadow_r, on='team', suffixes=('_live', '_shadow'))
    ratings['diff'] = ratings['elo_shadow'] - ratings['elo_live']
    ratings = ratings[['team', 'last_tournament', 'last_match', 'elo_live', 'elo_shadow', 'diff']]

    log_path = out / 'log.csv'
    logged = update_log(_read(log_path), prices, run)
    scores = score(logged, load_results(cfg))

    prices.to_csv(out / 'prices.csv', index=False)
    outrights.to_csv(out / 'outrights.csv', index=False)
    ratings.to_csv(out / 'ratings.csv', index=False)
    logged.to_csv(log_path, index=False)
    scores.to_csv(out / 'score.csv', index=False)

    last = pd.to_datetime(ratings['last_match'])
    season = ratings[last >= last.max() - pd.Timedelta(days=365)]                    # clubs seen in the last year
    by_div = (season.groupby('last_tournament')[['elo_live', 'elo_shadow', 'diff']].mean().round(0).astype(int)
              .reset_index().rename(columns={'last_tournament': 'last played in'}))
    moved = season.reindex(season['diff'].abs().sort_values(ascending=False).index).head(10)
    big = prices.reindex(prices['prob1_diff'].abs().sort_values(ascending=False).index).head(10)
    big = big[KEY + ['odds1_live', 'oddsX_live', 'odds2_live', 'odds1_shadow', 'oddsX_shadow', 'odds2_shadow', 'prob1_diff']]
    gaps = (prices.assign(gap=prices['prob1_diff'].abs()).groupby('competition')['gap']
            .agg(['size', 'mean', 'max']).round(3).reset_index()
            .rename(columns={'size': 'fixtures', 'mean': 'mean |prob1 change|', 'max': 'max'}))
    text = (f"# Shadow run: {engine} engine v live prices\n\n"
            f"Run {run} UTC. The published outputs one folder up are the live engine "
            f"(`elo.engine: {cfg.get('elo', {}).get('engine', 'elo')}`); nothing here changes them.\n\n"
            f"## Score so far\n\nEach engine's last prediction before kick-off, 1X2 log loss on 90-minute results "
            f"(negative change = shadow better; logging started with the first shadow run).\n\n{_table(scores)}\n"
            f"## Price differences, unplayed fixtures\n\n{_table(gaps)}\nLargest:\n\n{_table(big)}\n"
            f"## Ratings\n\nAverage by division (clubs that have played this year):\n\n{_table(by_div)}\n"
            f"Largest differences:\n\n{_table(moved)}")
    (out / 'README.md').write_text(text, encoding='utf-8', newline='\n')
    log(f"Shadow run ({engine}): {len(prices)} fixtures compared, {int(scores['matches'].iloc[0]) if len(scores) else 0} "
        f"logged matches scored -> {out}")
    return out
