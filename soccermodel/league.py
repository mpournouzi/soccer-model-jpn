"""
League tables and Monte Carlo season simulation.

Each remaining match is simulated by drawing a scoreline whose outcome probabilities
are exactly the Davidson 1X2 probabilities, with the scoreline shape (and so goal
difference) taken from the Poisson expected goals.
Ranking: points, goal difference, goals scored, then random.

Formats beyond a plain round robin (all optional, see config.yaml):
  * adjustments  points deductions/bonuses, e.g. {Volos NFC: -3}
  * phases       after the fixture list, the table splits into groups by position
                 (e.g. 1-4, 5-8, 9-14) that play another round robin, starting from their
                 regular-season points carried in full, halved or reset. Final positions run
                 through the groups in order, so 5th is the best team of the 5-8 group.
                 Until a phase's fixtures are in the fixtures file (rows with that `phase`),
                 its groups are formed from each simulated table and the round robin is
                 generated (single round robin: random home side).
  * groups       the fixture file has a `group` column; every group is its own table
                 (see simulate_final_group for play-offs between group winners).
"""

import numpy as np
import pandas as pd

from .poisson import MAX_GOALS, score_matrix

_G = MAX_GOALS + 1
_H, _A = np.divmod(np.arange(_G * _G), _G)
_OUTCOME = np.sign(_H - _A)  # 1 home win, 0 draw, -1 away win

CARRY_RULES = ('full', 'half_up', 'half_down', 'half', 'none')


def scoreline_cdf(p1: float, px: float, p2: float, expg1: float, expg2: float) -> np.ndarray:
    m = score_matrix(expg1, expg2).ravel()
    w = np.zeros_like(m)
    for outcome, p in ((1, p1), (0, px), (-1, p2)):
        mask = _OUTCOME == outcome
        w[mask] = m[mask] / m[mask].sum() * p
    cdf = np.cumsum(w)
    return cdf / cdf[-1]


def sample_scores(cdf: np.ndarray, u: np.ndarray):
    idx = np.searchsorted(cdf, u, side='right').clip(max=len(cdf) - 1)
    return _H[idx], _A[idx]


def league_table(results: pd.DataFrame, teams, adjustments: dict = None, tiebreak=None) -> pd.DataFrame:
    rows = {t: dict(team=t, P=0, W=0, D=0, L=0, GF=0, GA=0) for t in teams}
    for home, away, hg, ag in results[['home', 'away', 'hg', 'ag']].itertuples(index=False):
        for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
            r = rows[team]
            r['P'] += 1
            r['GF'] += gf
            r['GA'] += ga
            r['W' if gf > ga else 'D' if gf == ga else 'L'] += 1
    table = pd.DataFrame(rows.values())
    table['GD'] = table['GF'] - table['GA']
    table['Pts'] = 3 * table['W'] + table['D']
    if adjustments:
        table['Adj'] = table['team'].map(adjustments).fillna(0).astype(int)
        table['Pts'] += table['Adj']
    table = table.sort_values(['Pts', 'GD', 'GF'], ascending=False, kind='stable').reset_index(drop=True)
    if tiebreak and list(tiebreak) != list(DEFAULT_TIEBREAK):
        one = Tally.from_table(table, 1, results, any(t.startswith('h2h_') for t in tiebreak))
        tb = [t for t in tiebreak if t != 'regular_pos']
        table = table.iloc[one.order(np.random.default_rng(0), tb)[0]].reset_index(drop=True)
    table.insert(0, 'pos', range(1, len(table) + 1))
    return table


def carry_points(pts: np.ndarray, rule: str) -> np.ndarray:
    if rule == 'full':
        return pts.copy()
    if rule == 'none':
        return np.zeros_like(pts)
    if rule == 'half':
        return pts / 2
    if rule == 'half_up':
        return np.ceil(pts / 2)
    if rule == 'half_down':
        return np.floor(pts / 2)
    raise ValueError(f"carry must be one of {', '.join(CARRY_RULES)}, not {rule!r}")


class PairSampler:
    """Scorelines for (home, away) team-index arrays, one match per simulation; CDFs built on first use."""

    def __init__(self, teams, model):
        self.teams = list(teams)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.model = model
        self._cdfs = {}

    def _cdf(self, h: int, a: int) -> np.ndarray:
        if (h, a) not in self._cdfs:
            p = self.model.predict(self.teams[h], self.teams[a])
            self._cdfs[(h, a)] = scoreline_cdf(p['prob1'], p['probX'], p['prob2'], p['expg1'], p['expg2'])
        return self._cdfs[(h, a)]

    def sample(self, h: np.ndarray, a: np.ndarray, rng):
        n = len(self.teams)
        code = h * n + a
        u = rng.random(len(code))
        hg, ag = np.empty(len(code), int), np.empty(len(code), int)
        order = np.argsort(code, kind='stable')
        uniq, starts = np.unique(code[order], return_index=True)
        for c, rows in zip(uniq, np.split(order, starts[1:])):
            hg[rows], ag[rows] = sample_scores(self._cdf(c // n, c % n), u[rows])
        return hg, ag


DEFAULT_TIEBREAK = ('GD', 'GF')
TIEBREAKS = ('h2h_pts', 'h2h_gd', 'h2h_gf', 'GD', 'GF', 'W', 'away_GF', 'away_W', 'regular_pos')


def check_tiebreak(tiebreak, where: str):
    bad = [t for t in tiebreak if t not in TIEBREAKS]
    if bad:
        raise ValueError(f"{where}: unknown tiebreak {bad} - use {', '.join(TIEBREAKS)}")


class Tally:
    """
    Points, goals and wins for every simulation (rows) and team (columns), plus head-to-head
    points / goal difference / goals between every pair when `h2h` is on.

    Head-to-head criteria form a mini-table of all teams level on points (applied once:
    if they are still level, the next criterion decides).
    """

    STATS = ('pts', 'gd', 'gf', 'w', 'away_gf', 'away_w')
    H2H = ('h2h_pts', 'h2h_gd', 'h2h_gf')

    def __init__(self, n_sims: int, n: int, h2h: bool = False):
        for s in self.STATS:
            setattr(self, s, np.zeros((n_sims, n)))
        for s in self.H2H:
            setattr(self, s, np.zeros((n_sims, n, n), np.float32) if h2h else None)
        self.rows = np.arange(n_sims)

    @property
    def has_h2h(self) -> bool:
        return self.h2h_pts is not None

    @classmethod
    def from_table(cls, table: pd.DataFrame, n_sims: int, results: pd.DataFrame = None, h2h: bool = False):
        """Totals from the table (incl. adjustments); away stats and head-to-heads from `results`."""
        teams = table['team'].tolist()
        one = cls(1, len(teams), h2h)
        if results is not None:
            idx = {t: i for i, t in enumerate(teams)}
            for home, away, hg, ag in results[['home', 'away', 'hg', 'ag']].itertuples(index=False):
                one.add(idx[home], idx[away], np.array([int(hg)]), np.array([int(ag)]))
        for s, col in (('pts', 'Pts'), ('gd', 'GD'), ('gf', 'GF'), ('w', 'W')):
            getattr(one, s)[0] = table[col].to_numpy(float)
        out = cls(0, 0)
        out.rows = np.arange(n_sims)
        for s in cls.STATS:
            setattr(out, s, np.repeat(getattr(one, s), n_sims, axis=0))
        for s in cls.H2H:
            setattr(out, s, np.repeat(getattr(one, s), n_sims, axis=0) if h2h else None)
        return out

    def subset(self, members: np.ndarray, carry: str, h2h: bool):
        """The tally restricted to `members` (simulations x k columns), points carried over by `carry`."""
        rows = self.rows[:, None]
        sub = Tally(0, 0)
        sub.rows = self.rows
        for s in self.STATS:
            setattr(sub, s, getattr(self, s)[rows, members])
        sub.pts = carry_points(sub.pts, carry)
        for s in self.H2H:
            m = getattr(self, s)
            if h2h and m is not None:
                setattr(sub, s, m[rows[:, :, None], members[:, :, None], members[:, None, :]])
            elif h2h:
                setattr(sub, s, np.zeros(members.shape + members.shape[1:], np.float32))
            else:
                setattr(sub, s, None)
        return sub

    def add(self, h, a, hg, ag):
        """h/a: a column index, or one column index per simulation."""
        rows = self.rows if np.ndim(h) else slice(None)
        hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        self.pts[rows, h] += hp
        self.pts[rows, a] += ap
        self.gd[rows, h] += hg - ag
        self.gd[rows, a] += ag - hg
        self.gf[rows, h] += hg
        self.gf[rows, a] += ag
        self.w[rows, h] += hg > ag
        self.w[rows, a] += ag > hg
        self.away_gf[rows, a] += ag
        self.away_w[rows, a] += ag > hg
        if self.has_h2h:
            self.h2h_pts[rows, h, a] += hp
            self.h2h_pts[rows, a, h] += ap
            self.h2h_gd[rows, h, a] += hg - ag
            self.h2h_gd[rows, a, h] += ag - hg
            self.h2h_gf[rows, h, a] += hg
            self.h2h_gf[rows, a, h] += ag

    def order(self, rng, tiebreak=DEFAULT_TIEBREAK, regular_pos: np.ndarray = None) -> np.ndarray:
        """
        Column indices from first to last in every simulation: points, then `tiebreak`, then random.
        regular_pos: each column's regular-season position (0 = top), for the `regular_pos` criterion.
        """
        plain = {'GD': self.gd, 'GF': self.gf, 'W': self.w, 'away_GF': self.away_gf, 'away_W': self.away_w}
        tied = None
        keys = []
        for name in tiebreak:
            if name.startswith('h2h_'):
                if tied is None:
                    tied = self.pts[:, :, None] == self.pts[:, None, :]
                keys.append((getattr(self, name) * tied).sum(axis=2))
            elif name == 'regular_pos':
                keys.append(-np.broadcast_to(regular_pos, self.pts.shape))
            else:
                keys.append(plain[name])
        return np.lexsort((-rng.random(self.pts.shape), *[-k for k in reversed(keys)], -self.pts), axis=-1)


def _ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    ranks[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])
    return ranks


def _sample_fixture(f, n_sims, rng):
    cdf = scoreline_cdf(f.prob1, f.probX, f.prob2, f.expg1, f.expg2)
    return sample_scores(cdf, rng.random(n_sims))


def _resolve_positions(positions, n: int):
    lo, hi = positions
    lo = lo if lo > 0 else n + lo + 1
    hi = hi if hi > 0 else n + hi + 1
    return lo, hi


def _play_phase(phase, members, tally, sampler, glob, n_sims, rng):
    """
    One position group of a later phase. members: simulations x k team columns (by regular-season
    order). Returns the group's final order as team columns (simulations x k) and phase points.
    """
    k = members.shape[1]
    tiebreak = phase.get('tiebreak', DEFAULT_TIEBREAK)
    h2h = any(t.startswith('h2h_') for t in tiebreak)
    sub = tally.subset(members, phase.get('carry', 'full'), h2h) if tally is not None else Tally(n_sims, k, h2h)
    known = phase.get('known')
    if known is not None:  # fixtures entered: fixed membership, team columns -> slots
        slot = {t: i for i, t in enumerate(known['teams'])}
        for r in known['results'].itertuples(index=False):
            sub.add(slot[r.home], slot[r.away], np.full(n_sims, r.hg), np.full(n_sims, r.ag))
        for f in known['remaining'].itertuples(index=False):
            hg, ag = _sample_fixture(f, n_sims, rng)
            sub.add(slot[f.home], slot[f.away], hg, ag)
    else:
        legs = int(phase.get('rounds', 2))
        for i in range(k):
            for j in range(i + 1, k):
                for leg in range(legs):
                    if legs == 1:  # single round robin: home side unknown, so random
                        swap = rng.random(n_sims) < 0.5
                        h, a = np.where(swap, j, i), np.where(swap, i, j)
                    else:
                        h, a = (np.full(n_sims, i), np.full(n_sims, j)) if leg % 2 == 0 else \
                               (np.full(n_sims, j), np.full(n_sims, i))
                    th, ta = members[np.arange(n_sims), h], members[np.arange(n_sims), a]
                    hg, ag = sampler.sample(glob[th], glob[ta], rng)
                    sub.add(h, a, hg, ag)
    order = sub.order(rng, tiebreak, regular_pos=np.arange(k))  # members are in regular-season order
    return members[np.arange(n_sims)[:, None], order], sub.pts[np.arange(n_sims)[:, None], order]


def needs_results(tiebreak) -> bool:
    """Criteria beyond the table's points / GD / GF / wins need the match results."""
    return any(t not in ('GD', 'GF', 'W', 'regular_pos') for t in tiebreak)


def simulate_season(table: pd.DataFrame, fixtures: pd.DataFrame, n_sims: int, rng,
                    phases: list = None, sampler: PairSampler = None, tiebreak=DEFAULT_TIEBREAK,
                    results: pd.DataFrame = None) -> dict:
    """
    fixtures needs home, away, prob1, probX, prob2, expg1, expg2 (the rest of the fixture list).
    phases: list of {name, positions: [first, last], rounds, carry, tiebreak?, known?} - see module
    docstring; `known` = {teams, results, remaining} once that phase's fixtures are in the fixtures file.
    tiebreak: criteria after points (TIEBREAKS); head-to-head and away criteria need `results`
    (this season's matches so far). A phase's head-to-heads include the regular-season matches.
    Returns teams, position probabilities, final points and each simulation's positions.
    """
    teams = table['team'].tolist()
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    all_tb = [t for x in [tiebreak] + [p.get('tiebreak', DEFAULT_TIEBREAK) for p in phases or []] for t in x]
    h2h = any(t.startswith('h2h_') for t in all_tb)
    if needs_results(all_tb) and results is None:
        raise ValueError(f"tiebreak {all_tb} needs this season's results")
    tally = Tally.from_table(table, n_sims, results if needs_results(all_tb) else None, h2h)
    for f in fixtures.itertuples(index=False):
        hg, ag = _sample_fixture(f, n_sims, rng)
        tally.add(idx[f.home], idx[f.away], hg, ag)

    order = tally.order(rng, tiebreak)
    pts = tally.pts
    regular = None
    if phases:  # the table after the fixture list, before the split
        regular = {'teams': teams, 'position_probs': position_probs(_ranks(order)), 'points': tally.pts}
    if phases:
        glob = np.array([sampler.index[t] for t in teams])
        final = np.empty_like(order)
        pts = tally.pts.copy()
        for phase in phases:
            lo, hi = _resolve_positions(phase['positions'], n)
            if phase.get('known') is not None:
                members = np.tile([idx[t] for t in phase['known']['teams']], (n_sims, 1))
            else:
                members = order[:, lo - 1:hi]
            grp_order, grp_pts = _play_phase(phase, members, tally, sampler, glob, n_sims, rng)
            final[:, lo - 1:hi] = grp_order
            pts[np.arange(n_sims)[:, None], grp_order] = grp_pts
        order = final

    positions = _ranks(order)
    return {'teams': teams, 'position_probs': position_probs(positions), 'points': pts, 'positions': positions,
            'regular': regular}


def position_probs(positions: np.ndarray) -> np.ndarray:
    """teams x positions share of simulations, from each simulation's positions (0 = top)."""
    n = positions.shape[1]
    counts = np.zeros((n, n))
    for pos in range(n):
        counts[:, pos] = (positions == pos).sum(axis=0)
    return counts / positions.shape[0]


def simulate_final_group(group_sims: dict, cfg: dict, sampler: PairSampler, n_sims: int, rng) -> dict:
    """
    Round robin between the teams finishing `position` (default 1st) in each listed group, e.g.
    the Gamma Ethniki North/South groups of group winners. Points start from zero.
    group_sims: group name -> simulate_season() output. Returns teams and position probabilities.
    """
    pos = int(cfg.get('position', 1)) - 1
    members = []
    for g in cfg['groups']:
        sim = group_sims[str(g)]
        team_cols = np.argmax(sim['positions'] == pos, axis=1)
        members.append(np.array([sampler.index[t] for t in sim['teams']])[team_cols])
    members = np.stack(members, axis=1)          # simulations x k, global team indices
    k = members.shape[1]
    phase = {'rounds': cfg.get('rounds', 2), 'tiebreak': cfg.get('tiebreak', DEFAULT_TIEBREAK)}
    identity = np.arange(len(sampler.teams))
    order, _ = _play_phase(phase, members, None, sampler, identity, n_sims, rng)
    teams = sorted(set(np.unique(members).tolist()))
    probs = np.zeros((len(teams), k))
    col = {t: i for i, t in enumerate(teams)}
    for p in range(k):
        u, c = np.unique(order[:, p], return_counts=True)
        for t, cnt in zip(u, c):
            probs[col[t], p] = cnt / n_sims
    return {'teams': [sampler.teams[t] for t in teams], 'position_probs': probs}


def season_outputs(sim: dict, markets: dict):
    teams = sim['teams']
    n = len(teams)
    probs = sim['position_probs']
    positions = pd.DataFrame(probs, columns=[f'P{i}' for i in range(1, n + 1)])
    positions.insert(0, 'team', teams)
    positions.insert(1, 'exp_pts', sim['points'].mean(axis=0).round(1))
    positions.insert(2, 'exp_pos', (probs * np.arange(1, n + 1)).sum(axis=1).round(2))
    positions = positions.sort_values('exp_pos').reset_index(drop=True)
    market_probs = market_table(teams, probs, markets).set_index('team').loc[positions['team']].reset_index()
    return positions, market_probs


def market_table(teams, probs: np.ndarray, markets: dict) -> pd.DataFrame:
    """[first, last] finishing-position markets; negative numbers count from the bottom."""
    n = probs.shape[1]
    out = pd.DataFrame({'team': teams})
    for name, positions in markets.items():
        lo, hi = _resolve_positions(positions, n)
        out[name] = probs[:, lo - 1:hi].sum(axis=1)
    return out
