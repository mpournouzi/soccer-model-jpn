"""Matchday report: README.md (rendered by GitHub) and one Excel workbook with every table."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return '' if pd.isna(v) else f'{v:g}'
        return str(v)
    header = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    sep = '|' + '|'.join('---' for _ in df.columns) + '|'
    rows = ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, sep, *rows])


def _odds_view(next_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['date', 'home', 'away', 'odds1', 'oddsX', 'odds2', 'oddsO25', 'oddsU25', 'expg1', 'expg2']
    return next_df[cols].rename(columns={'odds1': '1', 'oddsX': 'X', 'odds2': '2',
                                         'oddsO25': 'O2.5', 'oddsU25': 'U2.5'})


def _outright_view(outrights: pd.DataFrame) -> pd.DataFrame:
    return outrights[['team'] + [c for c in outrights.columns if c.endswith(' %') or c.endswith(' Yes')]]


def write_report(summary: dict, out_dir: Path):
    cfg = summary['cfg']
    results = summary['results']
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f"# {cfg['name']} {cfg['season']} — matchday report", '',
             f"Results up to **{results['date'].max():%Y-%m-%d}** ({len(results):,} matches). "
             f"Generated {stamp}. Full tables are in the CSV files next to this report and in `report.xlsx`.", '']

    if summary['warnings']:
        lines += ['> **Warnings**', *[f'> - {w}' for w in summary['warnings']], '']

    for lg in summary['leagues'].values():
        p = lg['params']
        lines += [f"## {lg['name']}", '',
                  f"HFA {p['hfa']:.1f} Elo · draw rate {p['draw_rate']:.3f} · goals/game {p['total_goals']:.2f} · "
                  f"{lg['remaining']} matches left", '']
        if len(lg['next']):
            lines += ['### Next matches', '', _md(_odds_view(lg['next'])), '']
        table = lg['table'].merge(lg['positions'][['team', 'exp_pts', 'exp_pos']], on='team')
        lines += ['### Table and projection', '', _md(table), '',
                  '### Outrights', '', _md(_outright_view(lg['outrights'])), '']

    for cup in summary['cups'].values():
        p = cup['params']
        source = 'draw from cup file' if cup['drawn'] else 'random draw (draw not entered yet)'
        lines += [f"## {cup['name']}", '',
                  f"{len(cup['start'])} teams left · {source} · HFA {p['hfa']:.1f} · draw rate {p['draw_rate']:.3f}", '']
        if cup['next'] is not None and len(cup['next']):
            lines += ['### Next matches', '', _md(_odds_view(cup['next'])), '']
        lines += ['### Outrights', '', _md(_outright_view(cup['outrights'])), '']

    top = summary['ratings'].head(30)[['team', 'elo', 'tilt', 'last_tournament']]
    lines += ['## Top 30 Elo ratings', '', _md(top), '']
    (out_dir / 'README.md').write_text('\n'.join(lines), encoding='utf-8')

    with pd.ExcelWriter(out_dir / 'report.xlsx') as xl:
        summary['ratings'].to_excel(xl, sheet_name='Ratings', index=False)
        summary['params'].to_excel(xl, sheet_name='Parameters', index=False)
        for key, lg in summary['leagues'].items():
            for part, df in (('table', lg['table']), ('next', lg['next']),
                             ('outrights', lg['outrights']), ('positions', lg['positions'])):
                df.to_excel(xl, sheet_name=f'{key} {part}'[:31], index=False)
        for key, cup in summary['cups'].items():
            cup['outrights'].to_excel(xl, sheet_name=f'{key} outrights'[:31], index=False)
            if cup['next'] is not None:
                cup['next'].to_excel(xl, sheet_name=f'{key} next'[:31], index=False)
