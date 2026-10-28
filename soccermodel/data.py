"""
Loading and validating the country data folder.

Scores can be entered in two places, and either works:
  * a new row in results.csv, or
  * the hg/ag columns of a row in a league fixtures file or a cup draw file.
Rows scored in fixture/draw files are merged into the results on load (skipping any
match already present in results.csv), so nothing has to be copied around by hand.
"""

import difflib
from pathlib import Path

import pandas as pd
import yaml

from .cup import cup_rounds

RESULT_COLS = ['tournament', 'season', 'date', 'home', 'away', 'hg', 'ag', 'hg_90', 'ag_90']
SCORE_COLS = ['hg', 'ag', 'hg_90', 'ag_90']


class DataError(Exception):
    pass


def load_config(country_dir: Path) -> dict:
    with open(country_dir / 'config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['dir'] = Path(country_dir)
    cfg['season'] = str(cfg['season'])
    cfg.setdefault('aliases', {})
    cfg.setdefault('leagues', {})
    cfg.setdefault('cups', {})
    return cfg


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'season': str}, keep_default_na=True, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    for col in ('home', 'away'):
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    return df


def _apply_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    if aliases:
        for col in ('home', 'away'):
            if col in df:
                df[col] = df[col].replace(aliases)
    return df


def _complete_scores(df: pd.DataFrame) -> pd.DataFrame:
    """hg_90/ag_90 are only needed when a match went to extra time."""
    for col in SCORE_COLS:
        if col not in df:
            df[col] = pd.NA
    df['hg_90'] = df['hg_90'].fillna(df['hg'])
    df['ag_90'] = df['ag_90'].fillna(df['ag'])
    return df


def load_fixture_file(path: Path, aliases: dict) -> pd.DataFrame:
    df = _apply_aliases(_read_csv(path), aliases)
    df = _complete_scores(df)
    df['date'] = pd.to_datetime(df.get('date'), errors='coerce')  # blank / TBD -> NaT
    df['played'] = df['hg'].notna() & df['ag'].notna()
    return df


def load_results(cfg: dict) -> pd.DataFrame:
    """results.csv plus any scores entered in fixture and cup draw files."""
    base = cfg['dir']
    results = _apply_aliases(_read_csv(base / cfg.get('results', 'results.csv')), cfg['aliases'])
    results = _complete_scores(results)
    results['date'] = pd.to_datetime(results['date'], errors='coerce')
    if results['date'].isna().any():
        bad = results[results['date'].isna()].head()
        raise DataError(f"results.csv has rows with an unreadable date:\n{bad.to_string()}")

    entered = []
    for lg in cfg['leagues'].values():
        fx = load_fixture_file(base / lg['fixtures'], cfg['aliases'])
        played = fx[fx['played']].copy()
        played['tournament'] = lg['tournament']
        played['season'] = cfg['season']
        entered.append(played)
    for cup in cfg['cups'].values():
        draw = load_fixture_file(base / cup['draw'], cfg['aliases'])
        labels = {r['name']: r['tournament'] for r in cup_rounds(cup)}
        played = draw[draw['played']].copy()
        played['tournament'] = played['round'].map(labels)
        played['season'] = cfg['season']
        entered.append(played)

    entered = [e for e in entered if len(e)]
    if entered:
        entered = pd.concat(entered, ignore_index=True)
        missing_date = entered['date'].isna()
        if missing_date.any():
            raise DataError("A scored fixture has no date - add the match date:\n"
                            f"{entered.loc[missing_date, ['tournament', 'home', 'away']].to_string()}")
        key = ['season', 'tournament', 'home', 'away']
        known = results[key].astype(str).agg('|'.join, axis=1)
        new_key = entered[key].astype(str).agg('|'.join, axis=1)
        entered = entered[~new_key.isin(known)]
        results = pd.concat([results, entered[RESULT_COLS]], ignore_index=True)

    for col in SCORE_COLS:
        results[col] = results[col].astype(int)
    results['season'] = results['season'].astype(str)
    return results[RESULT_COLS].sort_values('date', kind='stable').reset_index(drop=True)


def load_initial_elo(cfg: dict) -> dict:
    """
    Starting ratings. A team with aliases starts from the rating of whichever of its names
    appears first in results.csv (e.g. Iraklis Thessaloniki 2013 -> shown as Iraklis 1908 later).
    """
    df = _read_csv(cfg['dir'] / cfg.get('initial_elo', 'initial_elo.csv'))
    df['team'] = df['team'].astype(str).str.strip()
    ini = dict(zip(df['team'], df['ini_elo'].astype(float)))
    if cfg['aliases']:
        raw = _read_csv(cfg['dir'] / cfg.get('results', 'results.csv'))
        raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
        long = pd.concat([raw[['date', 'home']].set_axis(['date', 'team'], axis=1),
                          raw[['date', 'away']].set_axis(['date', 'team'], axis=1)])
        first_seen = long.groupby('team')['date'].min()
        names = {}
        for src, dst in cfg['aliases'].items():
            names.setdefault(dst, {dst}).add(src)
        for dst, group in names.items():
            seen = [n for n in group if n in first_seen.index and n in ini]
            if seen:
                ini[dst] = ini[min(seen, key=lambda n: first_seen[n])]
    return ini


def check_team_names(names, known: set, where: str):
    """Fail loudly on unknown team names (usually a typo) and suggest the closest match."""
    unknown = sorted(set(names) - known)
    if not unknown:
        return
    lines = []
    for team in unknown:
        close = difflib.get_close_matches(team, known, n=3, cutoff=0.6)
        hint = f" - did you mean {', '.join(repr(c) for c in close)}?" if close else \
               " - if this is a new club, add it to initial_elo.csv"
        lines.append(f"  '{team}'{hint}")
    raise DataError(f"Unknown team name(s) in {where}:\n" + "\n".join(lines))


def check_alias_overlap(cfg: dict):
    """
    An alias joins one club's names. Two names are two clubs if they played each other, or if
    joining them would give the club two matches on the same day (e.g. Chania and AOX Kissamikos,
    both in the Football League 2013-17). Using both names in one season is fine (Panionios2 in
    old rows, Panionios in new ones).
    """
    if not cfg['aliases']:
        return
    raw = _read_csv(cfg['dir'] / cfg.get('results', 'results.csv'))
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    groups = {}
    for src, dst in cfg['aliases'].items():
        groups.setdefault(dst, {dst}).add(src)
    problems = []
    for dst, names in groups.items():
        names = sorted(names)
        dates = {n: set(raw.loc[(raw['home'] == n) | (raw['away'] == n), 'date']) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                met = ((raw['home'] == a) & (raw['away'] == b)) | ((raw['home'] == b) & (raw['away'] == a))
                both = sorted(dates[a] & dates[b])
                if met.any():
                    problems.append(f"  '{a}' and '{b}' played each other ({raw.loc[met, 'date'].min():%Y-%m-%d})")
                elif both:
                    problems.append(f"  '{a}' and '{b}' both played on {both[0]:%Y-%m-%d}"
                                    + (f" (+{len(both) - 1} more days)" if len(both) > 1 else ''))
    if problems:
        raise DataError("Aliased names look like different clubs - remove the alias or fix the names:\n"
                        + "\n".join(problems))


def validate(results: pd.DataFrame, initial_elo: dict, cfg: dict):
    check_alias_overlap(cfg)
    known = set(initial_elo)
    check_team_names(pd.concat([results['home'], results['away']]), known, 'results')
    bad = results[(results['hg_90'] > results['hg']) | (results['ag_90'] > results['ag'])]
    if len(bad):
        raise DataError(f"hg_90/ag_90 cannot exceed the final score:\n{bad.to_string()}")
    for key, lg in cfg['leagues'].items():
        fx = load_fixture_file(cfg['dir'] / lg['fixtures'], cfg['aliases'])
        check_team_names(pd.concat([fx['home'], fx['away']]), known, lg['fixtures'])
    for key, cup in cfg['cups'].items():
        draw = load_fixture_file(cfg['dir'] / cup['draw'], cfg['aliases'])
        names = pd.concat([draw['home'], draw['away']])
        check_team_names(names[~names.str.startswith('W:')], known, cup['draw'])
        entrants = [t for r in cup['rounds'] for t in (r.get('entrants') or [])]
        check_team_names(entrants, known, f"{cup['name']} entrants in config.yaml")
