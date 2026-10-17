"""Checks against numbers taken from the existing workbooks / scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soccermodel import odds
from soccermodel.cup import alive_from_results
from soccermodel.data import DataError, check_team_names
from soccermodel.league import league_table, scoreline_cdf, season_outputs, simulate_season
from soccermodel.poisson import davidson, nu_from_draw_rate, over_prob

ROOT = Path(__file__).resolve().parent.parent


def test_power_exponents_match_workbook():
    assert odds.power_exponent(3, 1.11) == pytest.approx(0.905007, abs=1e-6)
    assert odds.power_exponent(2, 1.09) == pytest.approx(0.875672, abs=1e-6)
    assert odds.power_exponent(2, 1.08) == pytest.approx(0.889, abs=1e-3)


def test_match_prices_match_league_cup_workbook():
    # Gainare Tottori v V-Varen Nagasaki, League Cup workbook: 4.5 / 3.75 / 1.65 and O/U 1.65 / 2.1
    p1, px, p2 = 0.187726, 0.236973, 0.575301
    assert list(odds.price([p1, px, p2], 3, 1.11)) == [4.5, 3.75, 1.65]
    over = 0.567770623
    assert list(odds.price([over, 1 - over], 2, 1.09)) == [1.65, 2.1]


def test_davidson_matches_workbook():
    # dr_ = -194.547736 with draw 0.265 -> prob1 0.187726, probtie 0.236973
    p1, px, p2 = davidson(-194.547736, nu_from_draw_rate(0.265))
    assert (p1, px, p2) == pytest.approx((0.187726, 0.236973, 0.575301), abs=1e-5)


def test_over_prob():
    assert over_prob(1.052953 + 1.906967) == pytest.approx(0.567771, abs=1e-5)


def test_scoreline_sampling_keeps_davidson_outcomes():
    cdf = scoreline_cdf(0.5, 0.3, 0.2, 1.6, 1.0)
    probs = np.diff(np.concatenate([[0], cdf]))
    h, a = np.divmod(np.arange(len(probs)), 13)
    assert probs[h > a].sum() == pytest.approx(0.5)
    assert probs[h == a].sum() == pytest.approx(0.3)


def test_season_sim_probabilities_sum_to_one():
    res = pd.DataFrame({'home': ['A', 'C'], 'away': ['B', 'D'], 'hg': [2, 0], 'ag': [0, 0]})
    table = league_table(res, ['A', 'B', 'C', 'D'])
    fx = pd.DataFrame({'home': ['B', 'D'], 'away': ['A', 'C'], 'prob1': [.4, .4], 'probX': [.3, .3],
                       'prob2': [.3, .3], 'expg1': [1.3, 1.3], 'expg2': [1.1, 1.1]})
    sim = simulate_season(table, fx, 2000, np.random.default_rng(1))
    positions, markets = season_outputs(sim, {'Winner': [1, 1], 'Bottom 1': [-1, -1]})
    assert np.allclose(sim['position_probs'].sum(axis=0), 1)
    assert markets['Winner'].sum() == pytest.approx(1)


def test_unknown_team_suggests_close_name():
    with pytest.raises(DataError, match='ThespaKusatsu Gunma'):
        check_team_names(['Thespakusatsu Gunma'], {'ThespaKusatsu Gunma', 'Tochigi'}, 'test')


def test_cup_alive_two_legged_aggregate():
    res = pd.DataFrame({'season': ['s'] * 2, 'tournament': ['Cup QF'] * 2, 'home': ['A', 'B'],
                        'away': ['B', 'A'], 'hg': [1, 3], 'ag': [2, 1]})
    assert alive_from_results(res, 'Cup', 's', []) == ['B']


def test_japan_data_is_valid():
    from soccermodel.data import load_config, load_initial_elo, load_results, validate
    cfg = load_config(ROOT / 'data' / 'japan')
    results = load_results(cfg)
    validate(results, load_initial_elo(cfg), cfg)


def test_shootout_probabilities():
    from soccermodel.cup import shootout_win_prob
    assert shootout_win_prob(0.75, 0.75) == pytest.approx(0.5)
    p = shootout_win_prob(0.80, 0.70)
    assert 0.5 < p < 0.75
    assert p + shootout_win_prob(0.70, 0.80) == pytest.approx(1)


def test_shrink_without_current_season_equals_history_before_it():
    from soccermodel.params import TournamentParams, tournament_stats
    res = pd.DataFrame({'tournament': ['L'] * 40, 'season': ['old'] * 40,
                        'hg_90': [2, 1, 0, 1] * 10, 'ag_90': [0, 1, 1, 0] * 10})
    stats = tournament_stats(res, 'new')
    shrink = TournamentParams({'prediction_params': {'method': 'shrink'}}, stats).prediction('L')
    hist = TournamentParams({}, stats).prediction('L')
    assert shrink == pytest.approx(hist)


def test_calibrated_draw_rate_matches_average_and_exceeds_league_rate():
    from soccermodel.poisson import calibrate_draw_rate
    diffs = np.random.default_rng(0).normal(30, 110, 5000)
    level = calibrate_draw_rate(diffs, 0.26)
    assert level > 0.26
    assert davidson(diffs, nu_from_draw_rate(level))[1].mean() == pytest.approx(0.26, abs=1e-6)
