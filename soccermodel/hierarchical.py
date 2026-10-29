"""
Hierarchical rating (`elo.engine: hierarchical`): a club's rating is its division's level plus its own
deviation, each with an uncertainty, updated by the simplified Kalman filter of Szczecinski & Tihon (2023).
Port of the prototype tools/hierarchical.py (rolling-origin backtests in docs/PIPELINE.md).

For a match, d = r_home - r_away + HFA, E = logistic(d), I = E(1 - E) / c^2 with c = 400 / ln 10; w is
the tournament's K relative to elo.default_k and g the ClubElo goal multiplier (as in elo.py):

    innovation = w * g * (y - E) / c / (1 + w * (u_h + u_a + P_h + P_a) * I)
    deviation_h += v_h * innovation          (away side mirrored)
    u_h          = v_h * (1 - w * v_h * I / den) + q^2       (variance; q = drift per match)

* League matches (both clubs in one division) move only the two deviations, and the division's deviations
  are re-centred to mean zero after every match, so a division can't inflate or deflate on its own.
* Matches between divisions (cups, play-offs) also move the two division levels, by their variance P.
* Top division on plain Elo (`plain_top: true`): its clubs move by exactly default_k * g * (y - E) * w and
  add nothing to the denominator, so SL1 league matches are the plain Elo update.
* Season start: a club keeps its rating, shrinks by `rho` toward its old division level, adds the summer
  variance (s_top / s_low), and on changing division `up` Elo per division climbed (or -`down` per division
  dropped) plus move^2 variance. A new club starts at initial_elo + `new` with variance new_sd^2 (clubs of
  the first season: init_sd^2). Each division's level becomes the mean rating of its new members, with
  variance P + p_summer^2 (a new division: p_init^2). With `away_summers: true` a club back after seasons
  without a match gets the summer variance once for every season since its last match; with `away_cap: true`
  those extra summers stop at a new club's variance new_sd^2 (a club long away is no less known than a new one).
* Clubs outside the leagues (cup-only) keep a plain rating with their own variance.

Divisions come from `hierarchical.tiers`: tournament -> tier, or {first season: tier, ...} when it changed,
or `next`: a transition tournament whose clubs belong to the division they play the following season (a
group stage mixing two divisions, e.g. Japan's spring 2026 season; the previous season's if none).
A club's division in a season is the league it played most in; for the current season the fixture lists
count too, so clubs that haven't played yet are placed.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

C = 400 / np.log(10)
PARAMS = ['q_top', 'q_low', 's_top', 's_low', 'rho', 'up', 'down', 'move', 'new', 'new_sd', 'init_sd',
          'p_init', 'p_summer']


def tier_of(tiers: dict, tournament: str, season: str):
    """Tier of a league tournament in a season, or None for non-league tournaments."""
    spec = tiers.get(tournament)
    if spec is None or not isinstance(spec, dict):
        return spec
    tier = None
    for start, t in sorted(spec.items()):
        if str(season) >= str(start):
            tier = t
    return tier


def club_tiers(results: pd.DataFrame, tiers: dict, lengths: dict = None) -> dict:
    """(team, season) -> tier of the league it played most in that season. lengths: (team, tournament,
    season) -> matches, e.g. the current season's fixture lists; replaces the results' count for those."""
    lg = results[results['tournament'].isin(list(tiers))]
    long = pd.concat([lg[['home', 'season', 'tournament']].set_axis(['team', 'season', 'tournament'], axis=1),
                      lg[['away', 'season', 'tournament']].set_axis(['team', 'season', 'tournament'], axis=1)])
    main = long.groupby(['team', 'season'])['tournament'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    best = {}
    for (team, t, s), n in (lengths or {}).items():
        if t in tiers and n > best.get((team, s), (0, None))[0]:
            best[(team, s)] = (n, t)
    main.update({k: t for k, (_, t) in best.items()})
    out = {(team, s): tier_of(tiers, t, s) for (team, s), t in main.items()}
    seasons = sorted({s for _, s in out})
    for (team, s), t in list(out.items()):
        if t == 'next':                            # transition tournament: the club's division next season
            i = seasons.index(s)
            after, before = out.get((team, seasons[i + 1])) if i + 1 < len(seasons) else None,                 out.get((team, seasons[i - 1])) if i else None
            out[(team, s)] = next((x for x in (after, before) if x not in (None, 'next')), None)
    return out


def run_hierarchical(results: pd.DataFrame, initial_elo: dict, params, weights, hcfg: dict, lengths: dict = None):
    """Same interface as elo.run_elo: (ratings dict, history with pre-match home_elo / away_elo)."""
    p = {k: float(hcfg['params'][k]) for k in PARAMS}
    plain_top = bool(hcfg.get('plain_top', False))
    away_summers = bool(hcfg.get('away_summers', False))
    away_cap = bool(hcfg.get('away_cap', False))
    tiers = hcfg['tiers']
    q = {True: p['q_top'] ** 2, False: p['q_low'] ** 2}
    summer = {True: p['s_top'] ** 2, False: p['s_low'] ** 2}
    floor = float(hcfg.get('top_k', params.default_k)) * C   # plain-Elo gain: K ~ variance K * c

    names = results['tournament'].unique()
    w_by_t = {t: params.k(t) / float(params.default_k) for t in names}
    hfa_by_t = {t: params.elo_hfa(t) for t in names}
    goals_by_t = {t: params.avg_goals(t) for t in names}
    seasons = sorted(results['season'].unique())
    first, order = seasons[0], {s: i for i, s in enumerate(seasons)}
    members = defaultdict(lambda: defaultdict(list))
    for (team, s), t in club_tiers(results, tiers, lengths).items():
        if t is not None:
            members[s][t].append(team)

    dev, var, div, last_t = {}, {}, {}, {}   # deviation (plain rating if div None), variance, division
    last_seen = {}                            # season index of a club's last match
    level, pvar, centre, size = {}, {}, defaultdict(float), {}

    def total(team):
        t = div[team]
        return dev[team] if t is None else level[t] + dev[team] - centre[t]

    def start_of(team, s):
        """(rating, variance) of a club seen for the first time in season s."""
        return (initial_elo[team] + (p['new'] if s != first else 0.0),
                (p['new_sd'] if s != first else p['init_sd']) ** 2)

    def season_start(s):
        new_div = {team: t for t, teams in members[s].items() for team in teams}
        ratings = {}
        for team, t in new_div.items():
            if team in dev:
                r = total(team)
                before = last_t.get(team)
                if div[team] is not None:
                    r -= p['rho'] * (r - level[div[team]])
                away = order[s] - last_seen[team] if away_summers else 1
                v = var[team] + summer[t == 1] * away
                if away_cap and away > 1:
                    v = min(v, max(var[team] + summer[t == 1], p['new_sd'] ** 2))
                if before is not None and before != t:
                    steps = before - t
                    r += p['up'] * steps if steps > 0 else p['down'] * steps
                    v += p['move'] ** 2
            else:
                r, v = start_of(team, s)
            ratings[team] = r
            var[team] = v
        for team in dev:                          # clubs leaving the leagues keep a plain rating
            if team not in new_div and div[team] is not None:
                dev[team] = total(team)
                div[team] = None
        for t, teams in members[s].items():
            level[t] = float(np.mean([ratings[team] for team in teams]))
            pvar[t] = pvar[t] + p['p_summer'] ** 2 if t in pvar else p['p_init'] ** 2
            centre[t] = 0.0
            size[t] = len(teams)
        for team, t in new_div.items():
            dev[team] = ratings[team] - level[t]
            div[team] = t
            last_t[team] = t

    def ensure(team, s):
        if team not in dev:                       # club outside the leagues
            dev[team], var[team] = start_of(team, s)
            div[team] = None

    pre_home, pre_away, current = [], [], -1
    for t, s, home, away, hg, ag in results[['tournament', 'season', 'home', 'away', 'hg', 'ag']].itertuples(index=False):
        if order[s] > current:
            season_start(s)
            current = order[s]
        ensure(home, s)
        ensure(away, s)
        last_seen[home] = last_seen[away] = order[s]
        th, ta = div[home], div[away]
        rh, ra = total(home), total(away)
        pre_home.append(rh)
        pre_away.append(ra)
        top_h, top_a = plain_top and th == 1, plain_top and ta == 1
        vh, va = var[home], var[away]
        uh, ua = (0.0 if top_h else vh), (0.0 if top_a else va)          # uncertainty
        gh, ga = (floor if top_h else vh), (floor if top_a else va)      # gain
        d = rh - ra + hfa_by_t[t]
        e = 1 / (1 + 10 ** (-d / 400))
        y = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g = weights.multiplier(d, abs(hg - ag), hg > ag, goals_by_t[t])
        w = w_by_t[t]
        info = e * (1 - e) / C ** 2
        same = th is not None and th == ta
        ph = 0.0 if same or th is None else pvar[th]
        pa = 0.0 if same or ta is None else pvar[ta]
        den = 1 + w * (uh + ua + ph + pa) * info
        innov = w * g * (y - e) / C / den
        dh, da = gh * innov, -ga * innov
        dev[home] += dh
        dev[away] += da
        if th is not None:
            centre[th] += dh / size[th]           # keep the division's deviations centred on zero
        if ta is not None:
            centre[ta] += da / size[ta]
        if ph:
            level[th] += ph * innov
            pvar[th] = ph * (1 - w * ph * info / den)
        if pa:
            level[ta] -= pa * innov
            pvar[ta] = pa * (1 - w * pa * info / den)
        var[home] = floor if top_h else vh * (1 - w * vh * info / den) + q[last_t.get(home) == 1]
        var[away] = floor if top_a else va * (1 - w * va * info / den) + q[last_t.get(away) == 1]

    weights.save()
    last = seasons[-1]
    ratings = {team: total(team) if team in dev else start_of(team, last)[0] for team in initial_elo}
    history = results.assign(home_elo=pre_home, away_elo=pre_away)
    return ratings, history
