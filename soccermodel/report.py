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
    cols = (['group'] if 'group' in next_df else []) + ['date', 'home', 'away', 'odds1', 'oddsX', 'odds2', 'oddsO25', 'oddsU25', 'expg1', 'expg2']
    view = next_df[cols].rename(columns={'odds1': '1', 'oddsX': 'X', 'odds2': '2',
                                         'oddsO25': 'O2.5', 'oddsU25': 'U2.5'})
    # non-breaking hyphens: GitHub otherwise wraps dates in wide tables
    return view.assign(date=view['date'].astype(str).str.replace('-', '‑'))


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
                  f"HFA {p['hfa']:.1f} Elo · draw rate {p['draw_rate']:.3f}"
                  + (f" (level teams; league average {p['league_draw_rate']:.3f})" if 'league_draw_rate' in p else '')
                  + f" · goals/game {p['total_goals']:.2f} · "
                  + (f"scale {p['scale']:.0f} · " if p.get('scale', 400) != 400 else '')
                  + f"{lg['remaining']} matches left", '']
        if lg.get('phases'):
            lines += [f"Format {lg['phases'][0]}", '']
        if lg.get('stakes'):
            shift = lg['stakes'].get('shift', 70)
            names = lg['settled']
            if not names:
                lines += ["Stakes: no side has nothing left to play for yet (no stakes adjustment).", '']
            elif len(names) <= 10:
                lines += [f"Stakes: {', '.join(names)} have nothing left to play for and are priced −{shift:g} Elo "
                          "in their remaining matches.", '']
            else:
                lines += ['<details>', f"<summary>Stakes: {len(names)} sides have nothing left to play for and are priced "
                          f"−{shift:g} Elo in their remaining matches</summary>", '', ', '.join(names), '', '</details>', '']
        if len(lg['next']):
            lines += ['### Next matches', '', _md(_odds_view(lg['next'])), '']
        table = lg['table'].merge(lg['positions'][['team', 'exp_pts', 'exp_pos']], on='team')
        if not lg.get('grouped'):
            lines += ['### Table and projection', '', _md(table), '',
                      '### Outrights', '', _md(_outright_view(lg['outrights'])), '']
            if lg.get('regular'):
                reg = lg['regular']
                view = lg['table'][['team', 'P', 'Pts']].merge(
                    reg['positions'][['team', 'exp_pts', 'exp_pos']], on='team').merge(
                    _outright_view(reg['outrights']), on='team').sort_values('exp_pos')
                lines += ['### Regular season (before the play-off split)', '', _md(view), '']
            continue
        if lg.get('final_markets'):
            m = lg['final_markets'][0]
            top = lg['outrights'].sort_values(m, ascending=False).head(12)
            lines += [f"### {m[:-2]} (group winners' play-off groups)", '',
                      _md(top[['group', 'team'] + [c for c in top.columns if c in lg['final_markets'] or
                                                   c.replace(' Yes', ' %') in lg['final_markets']]]), '']
        lines += ['### Tables, projections and outrights by group', '']
        for g, grp in table.groupby('group', sort=False):
            outr = lg['outrights'][lg['outrights']['group'] == g].drop(columns='group')
            lines += ['<details>', f'<summary>Group {g} — leader {grp.iloc[0]["team"]}</summary>', '',
                      _md(grp.drop(columns='group')), '', _md(_outright_view(outr)), '', '</details>', '']

    for cup in summary['cups'].values():
        source = 'draw from cup file' if cup['drawn'] else 'random draw (draw not entered yet)'
        phases = ' · '.join(f"{ph}: HFA {p['hfa']:.1f}, draw {p['draw_rate']:.3f}, goals {p['total_goals']:.2f}"
                            for ph, p in cup['params'].items())
        lines += [f"## {cup['name']}", '', f"{len(cup['start'])} teams in the first remaining round · {source}", '',
                  f"Parameters — {phases}", '']
        if cup['next'] is not None and len(cup['next']):
            lines += ['### Next matches', '', _md(_odds_view(cup['next'])), '']
        if cup.get('league_phase') is not None:
            lp = cup['league_phase']
            keep = [c for c in lp.columns if not c.endswith(' No')]
            lines += ['### League phase', '', _md(lp[keep]), '']
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
            if lg.get('regular'):
                lg['regular']['outrights'].to_excel(xl, sheet_name=f'{key} regular season'[:31], index=False)
        for key, cup in summary['cups'].items():
            cup['outrights'].to_excel(xl, sheet_name=f'{key} outrights'[:31], index=False)
            if cup.get('league_phase') is not None:
                cup['league_phase'].to_excel(xl, sheet_name=f'{key} league phase'[:31], index=False)
            if cup['next'] is not None:
                cup['next'].to_excel(xl, sheet_name=f'{key} next'[:31], index=False)
