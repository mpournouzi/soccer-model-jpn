"""
Per-tournament parameters, recomputed from the results on every run
(replaces running the HFA script and copying the numbers into the other scripts).

  hfa_elo     = 400 * log10(home wins / away wins), using 90-minute scores
  draw_rate   = draws / matches
  avg_goals   = mean of hg_90 + ag_90

Prediction parameters can either use all history (method: history) or shrink this
season's rates towards earlier seasons (method: shrink):

  rate = (prior_matches * rate_before_this_season + n * rate_this_season) / (prior_matches + n)

applied to the home-win, away-win and draw shares and to goals per game, so a handful of
early-season matches moves the estimate only a little and a full season moves it a lot.

`hfa_halflife: {tournament: h}` replaces the prediction HFA of those tournaments (either method) with
one from exponentially weighted matches: the latest match counts 1, a match h matches earlier 1/2:

  hfa_recent = 400 * log10(sum w * home win / sum w * away win),   w = 0.5 ** (matches since / h)

`scale: {tournament: s}` sets the Davidson scale used to price those tournaments (default 400, the Elo
scale): the rating difference counts 400 / s times as much, while HFA and price shifts keep their meaning,
so level teams still get the observed home/away win ratio. The Elo update always uses 400.
"""

import numpy as np
import pandas as pd

MIN_MATCHES_FOR_AUTO = 20


def _counts(df: pd.DataFrame) -> dict:
    return {
        'matches': len(df),
        'home_wins': int((df['hg_90'] > df['ag_90']).sum()),
        'away_wins': int((df['hg_90'] < df['ag_90']).sum()),
        'draws': int((df['hg_90'] == df['ag_90']).sum()),
        'goals': int((df['hg_90'] + df['ag_90']).sum()),
    }


def _derived(c: dict) -> dict:
    n, hw, aw = c['matches'], c['home_wins'], c['away_wins']
    return {
        'matches': n,
        'hfa_elo': 400 * np.log10(hw / aw) if hw and aw else np.nan,
        'draw_rate': c['draws'] / n if n else np.nan,
        'avg_goals': c['goals'] / n if n else np.nan,
    }


def _recent_hfa(grp: pd.DataFrame, halflife: float) -> float:
    g = grp[grp['hg_90'].notna()]
    w = 0.5 ** (np.arange(len(g))[::-1] / halflife)        # results are in date order: last match weight 1
    hw, aw = (w * (g['hg_90'] > g['ag_90'])).sum(), (w * (g['hg_90'] < g['ag_90'])).sum()
    return 400 * np.log10(hw / aw) if hw and aw else np.nan


def tournament_stats(results: pd.DataFrame, season: str, hfa_halflife: dict = None) -> pd.DataFrame:
    """hfa_halflife: {tournament: half-life in matches} adds hfa_recent for those tournaments."""
    hfa_halflife = hfa_halflife or {}
    rows = []
    for name, grp in results.groupby('tournament'):
        current = grp['season'] == season
        overall, now, before = _counts(grp), _counts(grp[current]), _counts(grp[~current])
        recent = {'hfa_recent': _recent_hfa(grp, float(hfa_halflife[name]))} if name in hfa_halflife else {}
        rows.append({'tournament': name, **_derived(overall), **recent,
                     **{f'season_{k}': v for k, v in _derived(now).items()},
                     **{f'season_{k}': v for k, v in now.items() if k != 'matches'},
                     **{f'prior_{k}': v for k, v in before.items()}})
    return pd.DataFrame(rows).set_index('tournament')


class TournamentParams:
    """Resolves Elo (k, hfa) and prediction (hfa, draw_rate, total_goals) settings."""

    def __init__(self, cfg: dict, stats: pd.DataFrame):
        self.cfg = cfg
        self.stats = stats
        self.elo_cfg = cfg.get('elo', {})
        self.tournaments = cfg.get('tournaments', {}) or {}
        self.default_k = self.elo_cfg.get('default_k', 30)
        self.default_goals = float(stats['avg_goals'].mean()) if len(stats) else 2.5
        pred_cfg = cfg.get('prediction_params', {}) or {}
        self.pred_method = pred_cfg.get('method', 'history')
        self.prior_matches = float(pred_cfg.get('prior_matches', 300))
        if self.pred_method not in ('history', 'shrink'):
            raise ValueError(f"prediction_params.method must be history or shrink, not {self.pred_method!r}")
        self.hfa_halflife = pred_cfg.get('hfa_halflife') or {}
        self.scale = pred_cfg.get('scale') or {}
        self.warnings = []

    def _auto(self, tournament: str, field: str, fallback: float, min_matches: int = MIN_MATCHES_FOR_AUTO) -> float:
        if tournament in self.stats.index:
            row = self.stats.loc[tournament]
            if row['matches'] >= min_matches and not np.isnan(row[field]):
                return float(row[field])
        warning = f"{tournament}: too few matches for auto {field}, using {fallback:.3f}"
        if warning not in self.warnings:
            self.warnings.append(warning)
        return fallback

    def k(self, tournament: str) -> float:
        return float(self.tournaments.get(tournament, {}).get('k', self.default_k))

    def season_boost(self, tournament: str):
        """(k_mult, share) from `season_boost: {k_mult, share}`, or None."""
        b = self.tournaments.get(tournament, {}).get('season_boost')
        return (float(b['k_mult']), float(b.get('share', 0.5))) if b else None

    def elo_hfa(self, tournament: str) -> float:
        value = self.tournaments.get(tournament, {}).get('hfa', 'auto')
        return self._auto(tournament, 'hfa_elo', 0.0) if value == 'auto' else float(value)

    def avg_goals(self, tournament: str) -> float:
        # Only selects which ClubElo goal-weight table is used, so any sample will do.
        return self._auto(tournament, 'avg_goals', self.default_goals, min_matches=1)

    def _shrunk(self, tournament: str):
        """(hfa, draw_rate, total_goals) blending this season into earlier seasons, or None."""
        if tournament not in self.stats.index:
            return None
        r = self.stats.loc[tournament]
        n0, n = r['prior_matches'], r['season_matches']
        if n0 < MIN_MATCHES_FOR_AUTO:
            return None
        w0 = self.prior_matches / n0  # each earlier match counts w0, so the prior totals prior_matches
        denom = self.prior_matches + n

        def share(prior_count, season_count):
            return (w0 * prior_count + season_count) / denom

        hw = share(r['prior_home_wins'], r['season_home_wins'])
        aw = share(r['prior_away_wins'], r['season_away_wins'])
        return (400 * np.log10(hw / aw),
                share(r['prior_draws'], r['season_draws']),
                share(r['prior_goals'], r['season_goals']))

    def prediction(self, tournament: str, overrides: dict = None) -> dict:
        """hfa / draw_rate / total_goals / scale for predicting fixtures in one competition."""
        overrides = overrides or {}
        shrunk = self._shrunk(tournament) if self.pred_method == 'shrink' else None
        if shrunk:
            auto = dict(zip(('hfa', 'draw_rate', 'total_goals'), shrunk))
        else:
            auto = {'hfa': lambda: self._auto(tournament, 'hfa_elo', 0.0),
                    'draw_rate': lambda: self._auto(tournament, 'draw_rate', 0.25),
                    'total_goals': lambda: self.avg_goals(tournament)}
        if tournament in self.hfa_halflife:
            if 'hfa_recent' not in self.stats.columns:
                raise ValueError(f"prediction_params.hfa_halflife set for {tournament}: pass it to tournament_stats")
            auto['hfa'] = lambda: self._auto(tournament, 'hfa_recent', 0.0)
        out = {}
        for key in ('hfa', 'draw_rate', 'total_goals'):
            if key in overrides:
                out[key] = float(overrides[key])
            else:
                out[key] = float(auto[key]() if callable(auto[key]) else auto[key])
        out['scale'] = float(overrides.get('scale', self.scale.get(tournament, 400)))
        return out

    def elo_table(self) -> pd.DataFrame:
        names = sorted(self.stats.index)
        return pd.DataFrame([{'tournament': t, 'k': self.k(t), 'elo_hfa': round(self.elo_hfa(t), 1),
                              'avg_goals': round(self.avg_goals(t), 3)} for t in names])


# --------------------------
# Cup parameters: Elo-adjusted, split early/late, anchored to the leagues
# --------------------------
def fit_hfa_draw(elo_diffs, hg, ag):
    """
    Maximum-likelihood (hfa, level-teams draw rate) of the Davidson model given each match's
    pre-match Elo difference (home - away, without HFA) and 90-minute score.

    Unlike 400*log10(home wins / away wins), this separates home advantage from team strength,
    which matters in cups where the home side is systematically stronger (pro clubs hosting
    amateurs) or weaker (lower-division clubs hosting).
    """
    from scipy.optimize import minimize
    d = np.asarray(elo_diffs, dtype=float)
    outcome = np.where(np.asarray(hg) > np.asarray(ag), 0, np.where(np.asarray(hg) < np.asarray(ag), 1, 2))

    def nll(x):
        hfa, log_nu = x
        theta = 10 ** ((d + hfa) / 400)
        root = np.sqrt(theta)
        nu = np.exp(log_nu)
        den = theta + 1 + nu * root
        p = np.stack([theta / den, 1 / den, nu * root / den], axis=1)[np.arange(len(d)), outcome]
        return -np.log(np.clip(p, 1e-12, None)).sum()

    hfa, log_nu = minimize(nll, [30.0, np.log(0.6)], method='Nelder-Mead').x
    nu = np.exp(log_nu)
    return float(hfa), float(nu / (2 + nu))


def _group_fit(history: pd.DataFrame, labels) -> dict:
    g = history[history['tournament'].isin(labels)]
    if len(g) < MIN_MATCHES_FOR_AUTO:
        return {'matches': len(g), 'hfa': np.nan, 'draw_rate': np.nan, 'total_goals': np.nan, 'raw_hfa': np.nan}
    hfa, draw = fit_hfa_draw(g['home_elo'] - g['away_elo'], g['hg_90'], g['ag_90'])
    hw, aw = (g['hg_90'] > g['ag_90']).sum(), (g['hg_90'] < g['ag_90']).sum()
    return {'matches': len(g), 'hfa': hfa, 'draw_rate': draw,
            'total_goals': float((g['hg_90'] + g['ag_90']).mean()),
            'raw_hfa': 400 * np.log10(hw / aw) if hw and aw else np.nan}


def cup_phase_params(history: pd.DataFrame, cfg: dict, cup: dict):
    """
    {'early': params, 'late': params} for one cup, plus a table of the fits.

    Each phase is fitted on its own tournament labels (Elo-adjusted maximum likelihood), then
    blended with the league anchor, which counts as `anchor_matches` matches:
        value = (n_phase * phase_value + anchor_matches * anchor_value) / (n_phase + anchor_matches)
    """
    cp = cfg.get('cup_params', {}) or {}
    anchor_w = float(cp.get('anchor_matches', 300))
    anchor = _group_fit(history, cp.get('anchor', ['J1 League', 'J2 League', 'J3 League']))
    rows = [{'cup': cup['name'], 'group': 'anchor', **anchor}]
    out = {}
    for phase in ('early', 'late'):
        fit = _group_fit(history, cup.get(phase, []))
        n = fit['matches'] if not np.isnan(fit['hfa']) else 0
        blended = {k: (anchor[k] if n == 0 else (n * fit[k] + anchor_w * anchor[k]) / (n + anchor_w))
                   for k in ('hfa', 'draw_rate', 'total_goals')}
        out[phase] = blended
        rows.append({'cup': cup['name'], 'group': phase, **fit, **{f'used_{k}': v for k, v in blended.items()}})
    return out, pd.DataFrame(rows)
