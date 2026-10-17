"""
One-off import of the Japan Excel workbooks into data/japan/ CSVs.

    python tools/import_japan_excel.py "C:/Users/Admin/Documents/Projects/Soccer Modelling/CSV Data/Japan"

Drops the per-row k/hfa columns (they now live in config.yaml) and fixes known
name inconsistencies. Safe to re-run: it overwrites results.csv, initial_elo.csv
and the league fixture files.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'japan'

NAME_FIXES = {'Thespakusatsu Gunma': 'ThespaKusatsu Gunma'}
LEAGUE_FILES = {'j1': 'Japan1_2026-27_Fixtures.xlsx', 'j2': 'Japan2_2026-27_Fixtures.xlsx',
                'j3': 'Japan3_2026-27_Fixtures.xlsx', 'jfl': 'Japan4_2026-27_Fixtures.xlsx'}


def main(src: Path):
    (OUT / 'fixtures').mkdir(parents=True, exist_ok=True)

    res = pd.read_excel(src / 'Japan All 2014-Today.xlsx', sheet_name='Results')
    res = res[['tournament', 'season', 'date', 'home', 'away', 'hg', 'ag', 'hg_90', 'ag_90']]
    res['season'] = res['season'].astype(str)
    res['date'] = pd.to_datetime(res['date']).dt.strftime('%Y-%m-%d')
    for col in ('home', 'away'):
        res[col] = res[col].str.strip().replace(NAME_FIXES)
    res.to_csv(OUT / 'results.csv', index=False)
    print(f"results.csv: {len(res)} rows")

    ini = pd.read_excel(src / 'Japan All 2014-Today Initial elo values.xlsx', sheet_name='vector')
    ini['team'] = ini['team'].str.strip().replace(NAME_FIXES)
    ini[['team', 'ini_elo']].to_csv(OUT / 'initial_elo.csv', index=False)
    print(f"initial_elo.csv: {len(ini)} teams")

    for key, fname in LEAGUE_FILES.items():
        fx = pd.read_excel(src / fname)
        fx = fx.rename(columns={'Home': 'home', 'Away': 'away'})
        fx['date'] = pd.to_datetime(fx['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        for col in ('home', 'away'):
            fx[col] = fx[col].str.strip().replace(NAME_FIXES)
        for col in ('hg', 'ag', 'hg_90', 'ag_90'):
            fx[col] = pd.NA
        fx[['date', 'home', 'away', 'hg', 'ag', 'hg_90', 'ag_90']].to_csv(
            OUT / 'fixtures' / f'{key}_2026-27.csv', index=False)
        print(f"fixtures/{key}_2026-27.csv: {len(fx)} rows")


if __name__ == '__main__':
    main(Path(sys.argv[1]))
