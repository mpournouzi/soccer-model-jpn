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
"""

from collections import defaultdict

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


class CupSimulator:
    def __init__(self, cup_cfg: dict, draw: pd.DataFrame, start_teams: list, model, rng):
        self.rounds = cup_cfg['rounds']
        self.model = model
        self.rng = rng
        self.start_teams = start_teams
        self.ties = defaultdict(dict)  # round -> tie id -> list of leg rows
        for row in draw.sort_values(['round', 'tie', 'leg']).itertuples(index=False):
            self.ties[row.round].setdefault(int(row.tie), []).append(row)
        self._cache = {}

    def _match(self, home, away, neutral):
        key = (home, away, neutral)
        if key not in self._cache:
            p = self.model.predict(home, away, neutral)
            cdf = scoreline_cdf(p['prob1'], p['probX'], p['prob2'], p['expg1'], p['expg2'])
            self._cache[key] = (cdf, p['expg1'], p['expg2'])
        return self._cache[key]

    def _play_90(self, home, away, neutral):
        cdf, _, _ = self._match(home, away, neutral)
        hg, ag = sample_scores(cdf, self.rng.random(1))
        return int(hg[0]), int(ag[0])

    def _extra_time_and_pens(self, home, away, neutral):
        _, e1, e2 = self._match(home, away, neutral)
        hg, ag = self.rng.poisson(e1 / 3), self.rng.poisson(e2 / 3)
        if hg != ag:
            return home if hg > ag else away
        return home if self.rng.random() < 0.5 else away

    def _play_tie(self, a, b, legs, rnd):
        """a hosts leg 1. legs = scored/unscored rows from the draw file, or None."""
        neutral = bool(rnd.get('neutral', False))
        n_legs = int(rnd.get('legs', 1))
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
                    return a if self.rng.random() < 0.5 else b
                continue
            hg, ag = self._play_90(home, away, neutral)
            goals[home] += hg
            goals[away] += ag
            if i == n_legs - 1 and goals[a] == goals[b]:
                return self._extra_time_and_pens(home, away, neutral)
        return a if goals[a] > goals[b] else b

    def run(self, n_sims: int) -> pd.DataFrame:
        names = [r['name'] for r in self.rounds]
        reached = defaultdict(lambda: np.zeros(len(names) + 1))
        for _ in range(n_sims):
            teams = list(self.start_teams)
            refs = {}
            for r_i, rnd in enumerate(self.rounds):
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
        out = pd.DataFrame([{'team': t, **dict(zip(cols, v[1:] / n_sims))} for t, v in reached.items()])
        return out.sort_values('Winner', ascending=False).reset_index(drop=True)
