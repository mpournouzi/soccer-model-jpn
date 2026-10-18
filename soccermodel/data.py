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
        labels = {r['name']: r['tournament'] for r in cup['rounds']}
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
    df = _read_csv(cfg['dir'] / cfg.get('initial_elo', 'initial_elo.csv'))
    df['team'] = df['team'].astype(str).str.strip()
    return dict(zip(df['team'], df['ini_elo'].astype(float)))


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


def validate(results: pd.DataFrame, initial_elo: dict, cfg: dict):
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
