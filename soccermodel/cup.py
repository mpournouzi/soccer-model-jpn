"""
Knockout cup simulation.

The draw file lists ties as they become known:
    round,tie,leg,date,home,away,hg,ag,hg_90,ag_90,winner
* Rounds come from config (name, legs, neutral). A round with no rows in the draw file
  is drawn at random from the teams still alive, in every simulation.
* home/away may refer to an earlier winner as "W:<round>-<tie>", e.g. "W:QF-2".
* Scored legs are used as-is (hg/ag = final score incl. extra time). For a tie that
  ended level and went to penalties, put the team that went through in `winner`.
* If the first remaining round has no draw rows yet, the teams still alive are
  worked out from this season's cup results.
* A round can list `entrants`: seeded teams that join the competition at that round.
* Each round can be priced with its own match model (e.g. early- v late-round parameters).
* `extra_time: false` on a round: a level tie goes straight to penalties (Greek Cup 2026-27).

League-phase cups (Greek Cup from 2025-26): `league_phase` names the draw-file round holding the
league-phase fixtures. Its table (points, then the `tiebreak` list, then random) seeds the
knockout rounds, whose `ties` list pairings by league-phase position or earlier winner, home side
first, e.g. [[5, 12], ...] or [[1, W:PO-4], ...]. Ties entered in the draw file take precedence.
"""

from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

from .league import scoreline_cdf, sample_scores


def alive_from_results(results: pd.DataFrame, prefix: str, season: str, warnings: list) -> list:
    cup = results[(results['season'] == season) & results['tournament'].str.startswith(prefix)]
    teams, eliminated = set(), set()
    ties = defaultdict(list)
    for row in cup.itertuples(index=False):
        teams.update((row.home, row.away))
        ties[(row.tournament, frozenset((row.home, row.away)))].append(row)
    for (_, pair), legs in ties.items():
        goals = defaultdict(int)
        for leg in legs:
            goals[leg.home] += leg.hg
            goals[leg.away] += leg.ag
        a, b = sorted(pair)
        if goals[a] == goals[b]:
            warnings.append(f"{prefix}: can't tell who won {a} v {b} (level on aggregate) - "
                            "add the tie to the draw file with a `winner`")
            continue
        eliminated.add(a if goals[a] < goals[b] else b)
    return sorted(teams - eliminated)


def shootout_win_prob(p_a: float, p_b: float, kicks: int = 5) -> float:
    """
    P(team A wins a shoot-out): each side takes `kicks` penalties converting with its own
    probability, then sudden death. (Stopping early once a side can't catch up doesn't
    change who wins, so all five kicks are counted.)
    """
    pa = [comb(kicks, k) * p_a ** k * (1 - p_a) ** (kicks - k) for k in range(kicks + 1)]
    pb = [comb(kicks, k) * p_b ** k * (1 - p_b) ** (kicks - k) for k in range(kicks + 1)]
    win = sum(pa[i] * pb[j] for i in range(kicks + 1) for j in range(i))
    level = sum(pa[i] * pb[i] for i in range(kicks + 1))
    a_round, b_round = p_a * (1 - p_b), p_b * (1 - p_a)
    sudden = a_round / (a_round + b_round) if a_round + b_round > 0 else 0.5
    return win + level * sudden


class PenaltyModel:
    """
    method: coin  -> every shoot-out is 50/50
    method: skill -> per-team conversion rates from penalties.csv (`skill` column, else
                     scored/missed shrunk towards the file average by prior_kicks)
    """

    def __init__(self, cfg: dict, country_dir):
        cfg = cfg or {}
        self.method = cfg.get('method', 'coin')
        if self.method not in ('coin', 'skill'):
            raise ValueError(f"penalties.method must be coin or skill, not {self.method!r}")
        self.skills = {}
        self.default = float(cfg.get('default_skill', 0.75))
        if self.method == 'skill':
            df = pd.read_csv(country_dir / cfg.get('file', 'penalties.csv'))
            df['team'] = df['team'].str.strip()
            kicks = df['scored'] + df['missed']
            base = df['scored'].sum() / kicks.sum() if kicks.sum() else self.default
            prior = float(cfg.get('prior_kicks', 70))
            shrunk = (df['scored'] + prior * base) / (kicks + prior)
            skill = df['skill'] if 'skill' in df else pd.Series(np.nan, index=df.index)
            self.skills = dict(zip(df['team'], skill.fillna(shrunk)))
        self._cache = {}

    def skill(self, team: str) -> float:
        return self.skills.get(team, self.default)

    def home_win_prob(self, home: str, away: str) -> float:
        """Shoot-out win probability for `home`; home kicks first (no first-kicker edge modelled)."""
        if self.method == 'coin':
            return 0.5
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = shootout_win_prob(self.skill(home), self.skill(away))
        return self._cache[key]


def cup_rounds(cup: dict) -> list:
    """All rounds incl. a league phase (as a round dict with name/tournament), in order."""
    lp = cup.get('league_phase')
    return ([{'name': lp['round'], 'tournament': lp['tournament']}] if lp else []) + list(cup['rounds'])


def league_phase_order(lp_cfg: dict, teams: list, played, remaining, n_sims: int, rng, sample):
    """
    League-phase finishing order (team indices, simulations x teams) and points (simulations x teams).
    played/remaining: rows with home, away (and hg, ag); sample(row, n) -> (hg, ag) arrays.
    Tiebreaks after points: `tiebreak` from GD, GF, away_GF, W, away_W (default GD, GF), then random.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    stats = {k: np.zeros((n_sims, n)) for k in ('Pts', 'GD', 'GF', 'away_GF', 'W', 'away_W')}

    def add(h, a, hg, ag):
        for k, v in (('Pts', np.where(hg > ag, 3, np.where(hg == ag, 1, 0))), ('GD', hg - ag), ('GF', hg),
                     ('W', hg > ag)):
            stats[k][:, h] += v
        for k, v in (('Pts', np.where(ag > hg, 3, np.where(hg == ag, 1, 0))), ('GD', ag - hg), ('GF', ag),
                     ('W', ag > hg), ('away_GF', ag), ('away_W', ag > hg)):
            stats[k][:, a] += v

    for r in played:
        add(idx[r.home], idx[r.away], np.full(n_sims, int(r.hg)), np.full(n_sims, int(r.ag)))
    for r in remaining:
        hg, ag = sample(r, n_sims)
        add(idx[r.home], idx[r.away], hg, ag)
    keys = [rng.random((n_sims, n))] + [-stats[k] for k in reversed(lp_cfg.get('tiebreak', ['GD', 'GF']))]
    return np.lexsort(keys + [-stats['Pts']], axis=-1), stats['Pts']


def bracket_problem(start_teams: list, rounds: list):
    """None if the team count works through every round down to one winner, else a message."""
    n = len(start_teams)
    for rnd in rounds:
        n += len(rnd.get('entrants', []) or [])
        if n % 2:
            return f"{n} teams in {rnd['name']} (odd) - check `rounds`/`entrants` in config.yaml"
        n //= 2
    return None if n == 1 else f"the rounds end with {n} teams instead of 1 - check `rounds`/`entrants`"


class CupSimulator:
    def __init__(self, cup_cfg: dict, draw: pd.DataFrame, start_teams: list, models, rng,
                 penalties: PenaltyModel = None):
        """models: one MatchModel for every round, or a dict round name -> MatchModel."""
        self.rounds = cup_cfg['rounds']
        self.lp = cup_cfg.get('league_phase')
        self.models = models if isinstance(models, dict) else {r['name']: models for r in cup_rounds(cup_cfg)}
        self.rng = rng
        self.penalties = penalties or PenaltyModel({}, None)
        self.start_teams = start_teams
        self.ties = defaultdict(dict)  # round -> tie id -> list of leg rows
        for row in draw.sort_values(['round', 'tie', 'leg']).itertuples(index=False):
            self.ties[row.round].setdefault(int(row.tie), []).append(row)
        self._cache = {}

    def _match(self, home, away, neutral, model):
        key = (id(model), home, away, neutral)
        if key not in self._cache:
            p = model.predict(home, away, neutral)
            cdf = scoreline_cdf(p['prob1'], p['probX'], p['prob2'], p['expg1'], p['expg2'])
            self._cache[key] = (cdf, p['expg1'], p['expg2'])
        return self._cache[key]

    def _play_90(self, home, away, neutral, model):
        cdf, _, _ = self._match(home, away, neutral, model)
        hg, ag = sample_scores(cdf, self.rng.random(1))
        return int(hg[0]), int(ag[0])

    def _extra_time_and_pens(self, home, away, neutral, model):
        _, e1, e2 = self._match(home, away, neutral, model)
        hg, ag = self.rng.poisson(e1 / 3), self.rng.poisson(e2 / 3)
        if hg != ag:
            return home if hg > ag else away
        return self._shootout(home, away)

    def _shootout(self, home, away):
        return home if self.rng.random() < self.penalties.home_win_prob(home, away) else away

    def _play_tie(self, a, b, legs, rnd):
        """a hosts leg 1. legs = scored/unscored rows from the draw file, or None."""
        neutral = bool(rnd.get('neutral', False))
        n_legs = int(rnd.get('legs', 1))
        model = self.models[rnd['name']]
        goals = {a: 0, b: 0}
        for i in range(n_legs):
            home, away = (a, b) if i == 0 else (b, a)
            row = legs[i] if legs and i < len(legs) else None
            if row is not None and row.played:
                goals[home] += int(row.hg)
                goals[away] += int(row.ag)
                if i == n_legs - 1 and goals[a] == goals[b]:
                    if isinstance(row.winner, str) and row.winner in (a, b):
                        return row.winner
                    return self._shootout(home, away)
                continue
            hg, ag = self._play_90(home, away, neutral, model)
            goals[home] += hg
            goals[away] += ag
            if i == n_legs - 1 and goals[a] == goals[b]:
                if not rnd.get('extra_time', True):
                    return self._shootout(home, away)
                return self._extra_time_and_pens(home, away, neutral, model)
        return a if goals[a] > goals[b] else b

    def _league_phase(self, n_sims: int) -> np.ndarray:
        rows = [r for legs in self.ties.get(self.lp['round'], {}).values() for r in legs]
        model = self.models[self.lp['round']]

        def sample(r, n):
            cdf, _, _ = self._match(r.home, r.away, False, model)
            return sample_scores(cdf, self.rng.random(n))

        return league_phase_order(self.lp, self.start_teams, [r for r in rows if r.played],
                                  [r for r in rows if not r.played], n_sims, self.rng, sample)

    def run(self, n_sims: int) -> pd.DataFrame:
        names = [r['name'] for r in self.rounds]
        reached = defaultdict(lambda: np.zeros(len(names) + 1))
        lp_order = None
        if self.lp:
            lp_order, lp_pts = self._league_phase(n_sims)
            self.league_phase = self._league_phase_summary(lp_order, lp_pts)
        for s in range(n_sims):
            teams = list(self.start_teams)
            refs = {}
            seeds = [self.start_teams[i] for i in lp_order[s]] if lp_order is not None else None
            for r_i, rnd in enumerate(self.rounds):
                if rnd.get('ties'):
                    teams = self._seeded_round(rnd, r_i, seeds, refs, reached)
                    continue
                teams = teams + [t for t in rnd.get('entrants', []) or [] if t not in teams]
                for t in teams:
                    reached[t][r_i] += 1
                winners, used = [], set()
                for tie_id, legs in self.ties.get(rnd['name'], {}).items():
                    a, b = (refs.get(t[2:], t) if t.startswith('W:') else t for t in (legs[0].home, legs[0].away))
                    w = self._play_tie(a, b, legs, rnd)
                    refs[f"{rnd['name']}-{tie_id}"] = w
                    winners.append(w)
                    used.update((a, b))
                # teams not (yet) in the draw file are paired at random
                rest = [t for t in teams if t not in used]
                rest = [rest[i] for i in self.rng.permutation(len(rest))]
                for j in range(0, len(rest) - 1, 2):
                    winners.append(self._play_tie(rest[j], rest[j + 1], None, rnd))
                teams = winners
            for t in teams:
                reached[t][len(names)] += 1

        cols = [f'Reach {n}' for n in names[1:]] + ['Winner']
        teams = list(dict.fromkeys(list(self.start_teams) + list(reached)))
        out = pd.DataFrame([{'team': t, **dict(zip(cols, reached[t][1:] / n_sims))} for t in teams])
        return out.sort_values(['Winner', *cols[::-1]], ascending=False).reset_index(drop=True)

    def _league_phase_summary(self, order: np.ndarray, pts: np.ndarray) -> pd.DataFrame:
        """Expected points / position and the league phase's `markets` ([first, last] positions)."""
        n_sims, n = order.shape
        ranks = np.empty_like(order)
        ranks[np.arange(n_sims)[:, None], order] = np.arange(n)
        out = pd.DataFrame({'team': self.start_teams, 'exp_pts': pts.mean(axis=0).round(1),
                            'exp_pos': (ranks.mean(axis=0) + 1).round(2)})
        for name, (lo, hi) in (self.lp.get('markets') or {}).items():
            lo, hi = (lo if lo > 0 else n + lo + 1), (hi if hi > 0 else n + hi + 1)
            out[name] = ((ranks >= lo - 1) & (ranks <= hi - 1)).mean(axis=0)
        return out.sort_values('exp_pos').reset_index(drop=True)

    def _seeded_round(self, rnd, r_i, seeds, refs, reached) -> list:
        """A round whose `ties` name league-phase positions (1 = top) or earlier winners."""
        def resolve(x):
            if isinstance(x, str) and x.startswith('W:'):
                return refs[x[2:]]
            if isinstance(x, str) and not x.isdigit():
                return x  # a team name
            if seeds is None:
                raise ValueError(f"{rnd['name']}: ties refer to positions but the cup has no league_phase")
            return seeds[int(x) - 1]

        drawn = self.ties.get(rnd['name'], {})
        winners = []
        for tie_id, pair in enumerate(rnd['ties'], 1):
            legs = drawn.get(tie_id)
            a, b = (resolve(t) for t in ((legs[0].home, legs[0].away) if legs else pair))
            reached[a][r_i] += 1
            reached[b][r_i] += 1
            w = self._play_tie(a, b, legs, rnd)
            refs[f"{rnd['name']}-{tie_id}"] = w
            winners.append(w)
        return winners
