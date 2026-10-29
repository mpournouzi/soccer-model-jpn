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


def test_hfa_halflife_follows_recent_matches_only_where_configured():
    from soccermodel.params import TournamentParams, tournament_stats
    old = {'hg_90': [2, 2, 2, 0] * 50, 'ag_90': [0, 0, 0, 1] * 50}      # home wins 3:1
    new = {'hg_90': [1, 0] * 100, 'ag_90': [0, 1] * 100}                # home wins 1:1
    one = pd.DataFrame({'season': ['a'] * 200 + ['b'] * 200,
                        'hg_90': old['hg_90'] + new['hg_90'], 'ag_90': old['ag_90'] + new['ag_90']})
    res = pd.concat([one.assign(tournament='L'), one.assign(tournament='M')], ignore_index=True)
    cfg = {'prediction_params': {'hfa_halflife': {'L': 50}}}
    stats = tournament_stats(res, 'b', cfg['prediction_params']['hfa_halflife'])
    params = TournamentParams(cfg, stats)
    hist = 400 * np.log10((150 + 100) / (50 + 100))                    # home 250 wins, away 150
    assert params.prediction('M')['hfa'] == pytest.approx(hist)         # not configured: all history
    assert 0 < params.prediction('L')['hfa'] < 15                       # recent matches: close to 0
    assert params.prediction('L', {'hfa': 40})['hfa'] == 40             # explicit override still wins
    huge = tournament_stats(res, 'b', {'L': 1e9})
    assert huge.loc['L', 'hfa_recent'] == pytest.approx(huge.loc['L', 'hfa_elo'])
    with pytest.raises(ValueError):
        TournamentParams(cfg, tournament_stats(res, 'b')).prediction('L')


def test_scale_steepens_rating_differences_but_keeps_home_advantage():
    from soccermodel.goals import MatchModel
    from soccermodel.params import TournamentParams, tournament_stats
    res = pd.DataFrame({'tournament': ['L'] * 40 + ['M'] * 40, 'season': ['a'] * 80,
                        'hg_90': [2, 1, 0, 1] * 20, 'ag_90': [0, 1, 1, 0] * 20})
    params = TournamentParams({'prediction_params': {'scale': {'L': 300}}}, tournament_stats(res, 'a'))
    assert params.prediction('L')['scale'] == 300
    assert params.prediction('M')['scale'] == 400                        # not configured: the Elo scale
    assert params.prediction('L', {'scale': 350})['scale'] == 350

    class Lookup:
        def expg(self, d, total):
            return total / 2, total / 2

    pred = {'hfa': 60.0, 'draw_rate': 0.27, 'total_goals': 2.5}
    ratings = {'A': 1600, 'B': 1500, 'C': 1500}
    steep = MatchModel(ratings, {}, {**pred, 'scale': 300}, Lookup())
    flat = MatchModel(ratings, {}, pred, Lookup())
    nu = nu_from_draw_rate(0.27)
    # Davidson at scale 300 with HFA on that scale (60 x 300/400): same level-teams home/away ratio
    p = steep.predict('A', 'B')
    assert (p['prob1'], p['probX'], p['prob2']) == pytest.approx(davidson(100 + 60 * 300 / 400, nu, 300))
    assert steep.predict('B', 'C')['prob1'] == pytest.approx(flat.predict('B', 'C')['prob1'])
    assert steep.predict('A', 'B')['prob1'] > flat.predict('A', 'B')['prob1']
    assert steep.predict('A', 'B', shift=-70)['elo_diff_adj'] == pytest.approx(100 * 4 / 3 + 60 - 70)


def test_calibrated_draw_rate_matches_average_and_exceeds_league_rate():
    from soccermodel.poisson import calibrate_draw_rate
    diffs = np.random.default_rng(0).normal(30, 110, 5000)
    level = calibrate_draw_rate(diffs, 0.26)
    assert level > 0.26
    assert davidson(diffs, nu_from_draw_rate(level))[1].mean() == pytest.approx(0.26, abs=1e-6)


def test_bracket_check_counts_entrants():
    from soccermodel.cup import bracket_problem
    rounds = [{'name': 'PO'}, {'name': 'QF', 'entrants': ['E1', 'E2', 'E3', 'E4']}, {'name': 'SF'}, {'name': 'F'}]
    assert bracket_problem([f'T{i}' for i in range(8)], rounds) is None
    assert bracket_problem([f'T{i}' for i in range(10)], rounds) is not None


def test_fit_separates_home_advantage_from_strength():
    from soccermodel.params import fit_hfa_draw
    rng = np.random.default_rng(3)
    d = rng.normal(200, 120, 40000)                 # home side much stronger, as in early cup rounds
    p1, px, p2 = davidson(d + 15, nu_from_draw_rate(0.25))
    u = rng.random(len(d))
    hg = np.where(u < p1, 1, 0)
    ag = np.where(u > p1 + px, 1, 0)
    hfa, draw = fit_hfa_draw(d, hg, ag)
    assert hfa == pytest.approx(15, abs=8)
    assert draw == pytest.approx(0.25, abs=0.01)
    raw = 400 * np.log10((hg > ag).sum() / (hg < ag).sum())
    assert raw > 150                                # the raw ratio would call this ~200 "HFA"


def test_cup_entrants_join_and_one_winner():
    from soccermodel.cup import CupSimulator

    class Flat:
        def predict(self, h, a, neutral=False):
            return {'prob1': .4, 'probX': .3, 'prob2': .3, 'expg1': 1.3, 'expg2': 1.1}

    cfg = {'rounds': [{'name': 'PO', 'legs': 2}, {'name': 'QF', 'entrants': ['E1', 'E2', 'E3', 'E4']},
                      {'name': 'SF'}, {'name': 'Final', 'neutral': True}]}
    draw = pd.DataFrame(columns=['round', 'tie', 'leg', 'home', 'away', 'played', 'hg', 'ag', 'winner'])
    out = CupSimulator(cfg, draw, [f'T{i}' for i in range(8)], Flat(), np.random.default_rng(1)).run(500)
    assert out['Winner'].sum() == pytest.approx(1)
    assert out.set_index('team').loc['E1', 'Reach SF'] > 0


class _Flat:
    def predict(self, h, a, neutral=False, shift=0.0):
        return {'prob1': .4, 'probX': .3, 'prob2': .3, 'expg1': 1.3, 'expg2': 1.1}


def test_carry_points_rules():
    from soccermodel.league import carry_points
    pts = np.array([7.0, 8.0])
    assert list(carry_points(pts, 'half_up')) == [4, 4]
    assert list(carry_points(pts, 'half_down')) == [3, 4]
    assert list(carry_points(pts, 'none')) == [0, 0]


def test_phases_keep_teams_within_their_position_group():
    from soccermodel.league import PairSampler
    teams = [f'T{i}' for i in range(8)]
    res = pd.DataFrame({'home': ['T0', 'T1', 'T2', 'T3'], 'away': ['T7', 'T6', 'T5', 'T4'],
                        'hg': [9, 9, 0, 0], 'ag': [0, 0, 9, 9]})
    table = league_table(res, teams, {'T0': -3})
    assert table.set_index('team').loc['T0', 'Pts'] == 0
    fx = pd.DataFrame(columns=['home', 'away', 'prob1', 'probX', 'prob2', 'expg1', 'expg2'])
    phases = [{'name': 'Top', 'positions': [1, 4], 'rounds': 2, 'carry': 'full'},
              {'name': 'Bottom', 'positions': [5, -1], 'rounds': 1, 'carry': 'half_up'}]
    sim = simulate_season(table, fx, 500, np.random.default_rng(2), phases=phases,
                          sampler=PairSampler(teams, _Flat()))
    probs = pd.DataFrame(sim['position_probs'], index=sim['teams'])
    top = set(table['team'][:4])
    assert np.allclose(probs.loc[sorted(top)].iloc[:, :4].sum(axis=1), 1)    # never drop below 4th
    assert np.allclose(probs.sum(axis=0), 1)


def test_final_group_takes_one_team_per_group():
    from soccermodel.league import PairSampler, simulate_final_group
    sampler = PairSampler(['A1', 'A2', 'B1', 'B2'], _Flat())
    sims = {'1': {'teams': ['A1', 'A2'], 'positions': np.tile([0, 1], (300, 1))},
            '2': {'teams': ['B1', 'B2'], 'positions': np.tile([1, 0], (300, 1))}}
    out = simulate_final_group(sims, {'groups': [1, 2]}, sampler, 300, np.random.default_rng(0))
    assert out['teams'] == ['A1', 'B2']
    assert np.allclose(out['position_probs'].sum(axis=0), 1)


def test_league_phase_cup_seeds_bracket():
    from soccermodel.cup import CupSimulator
    teams = [f'T{i}' for i in range(1, 17)]
    rows = [{'round': 'LP', 'tie': i + 1, 'leg': 1, 'home': teams[i], 'away': teams[(i + 1) % 16],
             'played': True, 'hg': 16 - i, 'ag': 0, 'winner': np.nan} for i in range(16)]
    draw = pd.DataFrame(rows)
    cfg = {'league_phase': {'round': 'LP', 'tournament': 'LP', 'markets': {'Top 4': [1, 4]}},
           'rounds': [{'name': 'PO', 'extra_time': False, 'ties': [[5, 12], [6, 11], [7, 10], [8, 9]]},
                      {'name': 'QF', 'extra_time': False,
                       'ties': [[1, 'W:PO-4'], [2, 'W:PO-3'], [3, 'W:PO-2'], [4, 'W:PO-1']]},
                      {'name': 'SF', 'legs': 2, 'ties': [['W:QF-4', 'W:QF-1'], ['W:QF-3', 'W:QF-2']]},
                      {'name': 'Final', 'neutral': True, 'ties': [['W:SF-1', 'W:SF-2']]}]}
    sim = CupSimulator(cfg, draw, teams, _Flat(), np.random.default_rng(1))
    out = sim.run(400).set_index('team')
    lp = sim.league_phase.set_index('team')
    assert out['Winner'].sum() == pytest.approx(1)
    assert lp.loc['T1', 'Top 4'] == 1 and out.loc['T1', 'Reach QF'] == 1       # top 4 skip the play-offs
    assert lp['Top 4'].sum() == pytest.approx(4) and lp['exp_pos'].iloc[0] == 1
    assert out.loc['T16', 'Reach QF'] == 0                                       # 13th-16th are out


def test_known_phase_fixtures_fix_membership():
    from soccermodel.league import PairSampler
    from soccermodel.pipeline import _simulate_league_group
    teams = ['A', 'B', 'C', 'D']
    reg = pd.DataFrame({'home': ['A', 'B', 'A', 'C'], 'away': ['D', 'C', 'C', 'D'], 'hg': [3, 2, 1, 1],
                        'ag': [0, 0, 0, 0], 'date': pd.to_datetime(['2027-01-01'] * 4)})
    fx = pd.DataFrame({'date': pd.to_datetime(['2027-01-01'] * 4 + ['2027-04-01', '2027-04-02']),
                       'home': ['A', 'B', 'A', 'C', 'A', 'C'], 'away': ['D', 'C', 'C', 'D', 'B', 'D'],
                       'phase': [None] * 4 + ['Top', 'Bottom']})
    fx['played'] = [True] * 4 + [False, False]
    for c in ('hg', 'ag'):
        fx[c] = list(reg[c]) + [np.nan, np.nan]
    phase_res = pd.DataFrame({'home': ['A'], 'away': ['B'], 'hg': [0], 'ag': [5],
                              'date': pd.to_datetime(['2027-04-01'])})
    season = pd.concat([reg, phase_res], ignore_index=True)
    lg = {'phases': [{'name': 'Top', 'positions': [1, 2], 'rounds': 1},
                     {'name': 'Bottom', 'positions': [3, 4], 'rounds': 1}]}
    margins = {'three_way': 1.08, 'two_way': 1.08, 'outright': 1.08}
    out = _simulate_league_group(lg, fx, season, _Flat(), PairSampler(teams, _Flat()), margins, 300,
                                 np.random.default_rng(0), [])
    probs = pd.DataFrame(out['sim']['position_probs'], index=out['sim']['teams'])
    assert out['table'].set_index('team').loc['A', 'P'] == 2        # the play-off match isn't regular season
    assert probs.loc['B', 0] == 1                                   # B won the play-off 5-0 and leads on GD
    assert probs.loc[['C', 'D'], [2, 3]].values.sum() == pytest.approx(2)
    assert len(out['preds']) == 1                                   # only C v D left


def test_head_to_head_beats_goal_difference():
    # A and B level on 3 points; B has the better goal difference but A won the meeting
    res = pd.DataFrame({'home': ['A', 'B', 'A', 'C'], 'away': ['B', 'C', 'D', 'D'],
                        'hg': [1, 5, 0, 0], 'ag': [0, 0, 1, 0]})
    teams = ['A', 'B', 'C', 'D']
    assert league_table(res, teams)['team'].tolist() == ['D', 'B', 'A', 'C']
    greek = ['h2h_pts', 'h2h_gd', 'GD', 'GF']
    table = league_table(res, teams, tiebreak=greek)
    assert table['team'].tolist() == ['D', 'A', 'B', 'C']
    fx = pd.DataFrame(columns=['home', 'away', 'prob1', 'probX', 'prob2', 'expg1', 'expg2'])
    sim = simulate_season(table, fx, 50, np.random.default_rng(0), tiebreak=greek, results=res)
    assert sim['positions'][:, sim['teams'].index('A')].tolist() == [1] * 50


def test_head_to_head_mini_table_of_three():
    from soccermodel.league import Tally
    t = Tally(1, 3, h2h=True)                      # A beats B, B beats C, C beats A: all 3 points
    for h, a, hg, ag in ((0, 1, 1, 0), (1, 2, 4, 0), (2, 0, 2, 1)):
        t.add(h, a, np.array([hg]), np.array([ag]))
    order = t.order(np.random.default_rng(0), ['h2h_pts', 'h2h_gd'])[0]
    assert order.tolist() == [1, 0, 2]            # mini-table GD: B +3, A 0, C -3


def test_aliased_club_starts_from_earliest_name(tmp_path):
    from soccermodel.data import load_config, load_initial_elo
    (tmp_path / 'config.yaml').write_text('name: T\nseason: s\naliases: {Old: New}\n', encoding='utf-8')
    (tmp_path / 'results.csv').write_text('tournament,season,date,home,away,hg,ag\n'
                                          'L,s,2013-01-01,Old,X,1,0\nL,s,2021-01-01,New,X,1,0\n', encoding='utf-8')
    (tmp_path / 'initial_elo.csv').write_text('team,ini_elo\nOld,1050\nNew,650\nX,800\n', encoding='utf-8')
    assert load_initial_elo(load_config(tmp_path))['New'] == 1050


def test_alias_between_overlapping_names_is_refused(tmp_path):
    from soccermodel.data import check_alias_overlap, load_config
    (tmp_path / 'config.yaml').write_text('name: T\nseason: s\naliases: {Chania: AOX Kissamikos}\n', encoding='utf-8')
    (tmp_path / 'results.csv').write_text('tournament,season,date,home,away,hg,ag\n'
                                          'L,s,2013-01-01,Chania,X,1,0\nL,s,2014-01-01,AOX Kissamikos,Chania,1,0\n'
                                          'L,s,2016-01-01,Chania,X,1,0\n', encoding='utf-8')
    with pytest.raises(DataError, match='played each other'):
        check_alias_overlap(load_config(tmp_path))
    # one club under both names in the same season (old rows, new rows) is fine
    (tmp_path / 'config.yaml').write_text('name: T\nseason: s\naliases: {Panionios2: Panionios}\n', encoding='utf-8')
    (tmp_path / 'results.csv').write_text('tournament,season,date,home,away,hg,ag\n'
                                          'L,s,2026-10-19,Panionios2,X,1,1\nL,s,2026-10-25,Panionios,Y,0,4\n',
                                          encoding='utf-8')
    check_alias_overlap(load_config(tmp_path))


def test_settled_teams_need_every_market_decided_and_late_season():
    from soccermodel.pipeline import settled_teams
    out = pd.DataFrame({'team': ['Mid', 'Chaser', 'Early', 'Champ'],
                        'Winner %': [0.0, 12.0, 0.0, 99.5], 'Relegation %': [0.4, 0.0, 0.0, 0.0]})
    progress = {'Mid': 0.8, 'Chaser': 0.8, 'Early': 0.3, 'Champ': 0.9}
    assert settled_teams(out, progress, {'decided': 0.01, 'from': 0.6}) == ['Champ', 'Mid']


def test_stakes_shift_moves_prices_against_the_settled_side():
    from soccermodel.pipeline import _predict

    class Model:
        def predict(self, h, a, neutral=False, shift=0.0):
            return {'home_elo': 1, 'away_elo': 1, 'elo_diff_adj': shift, 'home_tilt': 1, 'away_tilt': 1,
                    'prob1': 0, 'probX': 0, 'prob2': 0, 'expg1': 0, 'expg2': 0}

    fx = pd.DataFrame({'date': pd.to_datetime(['2027-04-01'] * 3), 'home': ['S', 'A', 'S'], 'away': ['A', 'S', 'S']})
    out = _predict(Model(), fx, shifts={'S': 70})
    assert out['stakes_shift'].tolist() == [-70, 70, 0]
    assert 'stakes_shift' not in _predict(Model(), fx).columns


# --------------------------------------------------------------------------------------------------
# Hierarchical rating engine
# --------------------------------------------------------------------------------------------------
H_PARAMS = {'q_top': 0.0, 'q_low': 22.0, 's_top': 0.0, 's_low': 143.1, 'rho': 0.0, 'up': 3.4, 'down': 100.9,
            'move': 107.7, 'new': -60.3, 'new_sd': 203.9, 'init_sd': 237.0, 'p_init': 13.1, 'p_summer': 0.6}
H_CFG = {'plain_top': True, 'tiers': {'A1': 1, 'A2': 2}, 'params': H_PARAMS}


def hierarchical_league():
    """Three seasons of two 4-club divisions (A1 top, A2 below) and a cup: promotion/relegation swaps,
    a new club in season 2 and a cup-only club."""
    rng = np.random.default_rng(7)
    strength = {'T1': 300, 'T2': 150, 'T3': 0, 'T4': -100, 'L1': -150, 'L2': -250, 'L3': -350, 'L4': -400,
                'N1': -200, 'X1': -600}
    divisions = {'s1': (['T1', 'T2', 'T3', 'T4'], ['L1', 'L2', 'L3', 'L4']),
                 's2': (['T1', 'T2', 'T3', 'L1'], ['T4', 'L2', 'L3', 'N1']),      # L1 up, T4 down, N1 new
                 's3': (['T1', 'T2', 'L1', 'T4'], ['T3', 'L2', 'L3', 'N1'])}
    rows, day = [], pd.Timestamp('2020-08-01')
    for season, (top, low) in divisions.items():
        fixtures = [('A1', h, a) for h in top for a in top if h != a] + [('A2', h, a) for h in low for a in low if h != a]
        fixtures += [('CUP', low[i], top[i]) for i in range(4)] + [('CUP', 'X1', top[0])]
        for t, h, a in fixtures:
            edge = (strength[h] - strength[a] + 60) / 400
            hg, ag = rng.poisson(1.35 * 10 ** (edge / 4)), rng.poisson(1.35 * 10 ** (-edge / 4))
            rows.append({'tournament': t, 'season': season, 'date': day, 'home': h, 'away': a,
                         'hg': hg, 'ag': ag, 'hg_90': hg, 'ag_90': ag})
            day += pd.Timedelta(days=1)
        day += pd.Timedelta(days=60)
    initial = {'T1': 1600, 'T2': 1500, 'T3': 1450, 'T4': 1400, 'L1': 1300, 'L2': 1250, 'L3': 1200, 'L4': 1150,
               'N1': 1100, 'X1': 900}
    return pd.DataFrame(rows), initial


_WEIGHTS = []     # one ClubEloWeights for every test: its goal-weight tables take a while to build


def hierarchical_inputs(results):
    from soccermodel.elo import ClubEloWeights
    from soccermodel.params import TournamentParams, tournament_stats
    cfg = {'elo': {'default_k': 30}, 'tournaments': {'A1': {'k': 30, 'hfa': 60}, 'A2': {'k': 30, 'hfa': 60},
                                                     'CUP': {'k': 20, 'hfa': 40}}}
    if not _WEIGHTS:
        _WEIGHTS.append(ClubEloWeights(step=10))
    return TournamentParams(cfg, tournament_stats(results, 's3')), _WEIGHTS[0]


# Pre-match ratings from the prototype, tools/hierarchical.py (run_hierarchical, top_plain='all'), on
# hierarchical_league() with H_PARAMS: the first and last match of each season, and the sums over all 87
# (on the full Greek data the engine matched the prototype exactly for all six rolling-origin fits).
PROTOTYPE_PRE = {0: (1600.0, 1500.0), 28: (900.0, 1612.1415), 29: (1612.6374, 1454.672), 57: (893.1728, 1612.9198),
                 58: (1613.3602, 1476.8959), 86: (887.204, 1577.9476)}
PROTOTYPE_SUMS = (113642.1757, 118696.4508)


def test_hierarchical_reproduces_prototype_ratings():
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    _, hist = run_hierarchical(results, initial, *hierarchical_inputs(results), H_CFG)
    for i, pair in PROTOTYPE_PRE.items():
        assert (hist['home_elo'].iloc[i], hist['away_elo'].iloc[i]) == pytest.approx(pair, abs=1e-4)
    assert (hist['home_elo'].sum(), hist['away_elo'].sum()) == pytest.approx(PROTOTYPE_SUMS, abs=1e-3)


def test_hierarchical_away_summers_reopen_a_returning_club():
    # L2 sits out season 2; with away_summers it returns with two summers' uncertainty and moves more.
    # Numbers from the prototype (AWAY_SUMMERS = True) on the same data.
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    results = results[~((results['season'] == 's2') & ((results['home'] == 'L2') | (results['away'] == 'L2')))]
    results = results.reset_index(drop=True)
    params, weights = hierarchical_inputs(results)
    _, on = run_hierarchical(results, initial, params, weights, {**H_CFG, 'away_summers': True})
    _, off = run_hierarchical(results, initial, params, weights, H_CFG)
    assert (on['home_elo'][63], on['away_elo'][63]) == pytest.approx((1401.1164, 1307.9053), abs=1e-4)
    assert (on['home_elo'][66], on['away_elo'][66]) == pytest.approx((1270.7588, 1450.426), abs=1e-4)
    assert (on['home_elo'].sum(), on['away_elo'].sum()) == pytest.approx((105433.7051, 109585.2256), abs=1e-3)
    # rows 63 and 66: L2's first two matches back, a 1-0 defeat at T3 then the return match
    assert on['away_elo'][63] - on['home_elo'][66] > off['away_elo'][63] - off['home_elo'][66] > 0


def test_hierarchical_away_cap_stops_at_a_new_clubs_uncertainty():
    # Same returning club: two summers' variance exceeds new_sd^2, so the cap holds it there and L2's
    # defeat moves it less. Numbers from the prototype (AWAY_SUMMERS = AWAY_CAP = True) on the same data.
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    results = results[~((results['season'] == 's2') & ((results['home'] == 'L2') | (results['away'] == 'L2')))]
    results = results.reset_index(drop=True)
    params, weights = hierarchical_inputs(results)
    _, cap = run_hierarchical(results, initial, params, weights, {**H_CFG, 'away_summers': True, 'away_cap': True})
    _, on = run_hierarchical(results, initial, params, weights, {**H_CFG, 'away_summers': True})
    assert (cap['home_elo'][66], cap['away_elo'][66]) == pytest.approx((1278.4246, 1449.0287), abs=1e-4)
    assert (cap['home_elo'].sum(), cap['away_elo'].sum()) == pytest.approx((105440.4738, 109575.6541), abs=1e-3)
    assert cap['away_elo'][63] == on['away_elo'][63]               # before L2's first match back: no change
    assert on['away_elo'][63] - on['home_elo'][66] > cap['away_elo'][63] - cap['home_elo'][66] > 0


def test_hierarchical_top_division_alone_is_plain_elo():
    from soccermodel.elo import run_elo
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    results = results[(results['tournament'] == 'A1') & (results['season'] == 's1')]
    params, weights = hierarchical_inputs(results)
    top = ['T1', 'T2', 'T3', 'T4']
    elo, _ = run_elo(results, initial, params, weights)
    hier, _ = run_hierarchical(results, initial, params, weights, H_CFG)
    assert [hier[t] for t in top] == pytest.approx([elo[t] for t in top])


def test_hierarchical_top_k_is_plain_elo_at_that_k():
    from soccermodel.elo import run_elo
    from soccermodel.params import TournamentParams, tournament_stats
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    results = results[(results['tournament'] == 'A1') & (results['season'] == 's1')]
    params, weights = hierarchical_inputs(results)
    k20 = TournamentParams({'elo': {'default_k': 20}, 'tournaments': {'A1': {'k': 20, 'hfa': 60}}},
                           tournament_stats(results, 's3'))
    top = ['T1', 'T2', 'T3', 'T4']
    elo, _ = run_elo(results, initial, k20, weights)
    hier, _ = run_hierarchical(results, initial, params, weights, {**H_CFG, 'top_k': 20})
    assert [hier[t] for t in top] == pytest.approx([elo[t] for t in top])
    base, _ = run_hierarchical(results, initial, params, weights, H_CFG)
    assert [hier[t] for t in top] != pytest.approx([base[t] for t in top])

def test_hierarchical_league_matches_keep_the_division_mean():
    from soccermodel.hierarchical import run_hierarchical
    results, initial = hierarchical_league()
    results = results[results['season'] == 's1']
    low = ['L1', 'L2', 'L3', 'L4']
    start = np.mean([initial[t] for t in low])
    league = results[results['tournament'] != 'CUP']
    alone, _ = run_hierarchical(league, initial, *hierarchical_inputs(league), H_CFG)
    assert np.mean([alone[t] for t in low]) == pytest.approx(start)
    with_cup, _ = run_hierarchical(results, initial, *hierarchical_inputs(results), H_CFG)
    assert abs(np.mean([with_cup[t] for t in low]) - start) > 1e-3    # only cross-division matches move a level


def test_shadow_log_keeps_the_last_prediction_before_kick_off_and_scores_it():
    from soccermodel.shadow import score, update_log

    def prices(p_live, p_shadow, fixtures):
        return pd.DataFrame([{'competition': 'sl2', 'date': d, 'home': h, 'away': a,
                              'prob1_live': p_live, 'probX_live': 0.3, 'prob2_live': 0.7 - p_live,
                              'prob1_shadow': p_shadow, 'probX_shadow': 0.3, 'prob2_shadow': 0.7 - p_shadow}
                             for d, h, a in fixtures])

    both = [('2026-11-01', 'A', 'B'), ('2026-11-08', 'B', 'A')]
    log = update_log(pd.DataFrame(), prices(0.4, 0.5, both), 'run 1')
    log = update_log(log, prices(0.45, 0.55, both[1:]), 'run 2')       # A v B played: no longer listed
    assert log['prob1_live'].tolist() == [0.4, 0.45] and log['first_run'].tolist() == ['run 1', 'run 1']
    assert log['last_run'].tolist() == ['run 1', 'run 2']
    results = pd.DataFrame({'date': pd.to_datetime(['2026-11-01']), 'home': ['A'], 'away': ['B'],
                            'hg_90': [1], 'ag_90': [0]})
    s = score(log, results).set_index('competition')
    assert s.loc['all', 'matches'] == 1
    assert (s.loc['sl2', 'live'], s.loc['sl2', 'shadow']) == pytest.approx((-np.log(0.4), -np.log(0.5)), abs=1e-5)
    assert s.loc['sl2', 'change %'] < 0                                 # shadow gave the winner more


def test_hierarchical_tiers_change_by_season():
    from soccermodel.hierarchical import club_tiers, tier_of
    tiers = {'SL1': 1, 'FL': {'2013-14': 2, '2019-20': 3}}
    assert [tier_of(tiers, 'FL', s) for s in ('2013-14', '2018-19', '2019-20')] == [2, 2, 3]
    assert tier_of(tiers, 'SL1', '2020-21') == 1 and tier_of(tiers, 'CUP', '2020-21') is None
    res = pd.DataFrame({'tournament': ['FL', 'FL'], 'season': ['2020-21'] * 2, 'home': ['A', 'B'], 'away': ['B', 'A']})
    # the current season's fixture lists place a club that hasn't played yet
    assert club_tiers(res, tiers, {('C', 'SL1', '2020-21'): 26}) == {('A', '2020-21'): 3, ('B', '2020-21'): 3,
                                                                  ('C', '2020-21'): 1}


def test_hierarchical_transition_tournament_takes_the_next_seasons_division():
    from soccermodel.hierarchical import club_tiers
    # a spring group stage mixing two divisions, then the new season (one club placed by its fixture list)
    tiers = {'D1': 1, 'D2': 2, 'D3': 3, 'G1': 1, 'G23': 'next'}
    res = pd.DataFrame({'tournament': ['D2', 'D3', 'G23', 'G23', 'G1', 'D2', 'D3'],
                        'season': ['s1', 's1', 's2', 's2', 's2', 's3', 's3'],
                        'home': ['A', 'C', 'A', 'B', 'E', 'A', 'B'], 'away': ['X', 'D', 'C', 'D', 'F', 'C', 'X']})
    t = club_tiers(res, tiers, {('D', 'D2', 's3'): 30})
    assert [t[(c, 's2')] for c in 'ABCD'] == [2, 3, 2, 2]       # A, C, D play D2 in s3; B plays D3
    assert t[('E', 's2')] == 1                                    # plain tier in the same season
    res = res[res['season'] != 's3']
    assert club_tiers(res, tiers)[('C', 's2')] == 3               # no next season: the previous one
    assert club_tiers(res, tiers)[('B', 's2')] is None            # neither: outside the leagues
