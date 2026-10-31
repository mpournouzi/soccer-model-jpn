# soccer-model

Ratings, match odds and season/cup simulations for Japanese football (J1, J2, J3, JFL, Emperor's Cup,
League Cup), rebuilt automatically from a results CSV. Consolidates the old `elo_upd_jpn.py`,
`hfa claude/main.py`, `jpn.py` and the cup/season-sim workbooks into one pipeline.

**Latest report → [output/japan/README.md](output/japan/README.md)** (also `output/japan/report.xlsx`)

## Matchday routine

1. **Enter scores** — either:
   - type `hg,ag` into the match's row in `data/japan/fixtures/<league>_2026-27.csv`, or
   - add a row to `data/japan/results.csv`.

   Fill `hg_90,ag_90` only when a match went to extra time (they default to `hg,ag`).
   Shoot-outs: record the final score after extra time and put the team that went through in `winner`
   (cup draw files only).
2. **Push** (or edit the file directly on github.com). GitHub Actions validates the data, runs the
   model and commits the new `output/`. A typo in a team name stops the run with a "did you mean …?" message.

Locally the same run is:

```bash
pip install -r requirements.txt
python -m soccermodel update japan
```

`python -m soccermodel check japan` only validates the data; `--sims 2000` gives a quicker run.
`python -m soccermodel shadow japan --engine elo` (after an update) prices the matchday with the other rating engine
and logs both in `output/japan/shadow/`; it isn't part of the matchday run.

**How it works, step by step, with the maths and backtests → [docs/PIPELINE.md](docs/PIPELINE.md)**

## Occasional jobs

| When | What |
|---|---|
| Cup draw announced | Add the ties to `data/japan/cups/<cup>_2026-27.csv` (`round,tie,leg,date,home,away`). Later rounds may refer to winners as `W:R16-3`. Rounds without rows are drawn at random in each simulation. |
| New season | Add the new fixture files, update `season`, `leagues` (fixtures, markets) and `cups` (rounds, `early`/`late` labels, seeded `entrants`) in `data/japan/config.yaml`. A new league tournament label also needs its tier under `hierarchical.tiers`. |
| New club appears | Add it with a starting rating to `data/japan/initial_elo.csv` (the hierarchical engine starts it there with a wide uncertainty, so it settles quickly). |
| Shoot-out data | Update the conversion rates in `data/japan/penalties.csv`. |
| Name spelled two ways | Add it under `aliases:` in the config. |

## What the run does

1. **Parameters** — per tournament: HFA = 400·log10(home wins / away wins), draw rate and goals per game
   from 90-minute scores over all history (`output/japan/tournament_params.csv`). `hfa: auto` in the config
   uses these; a number overrides (0 for neutral venues).
2. **Ratings** — hierarchical engine (`elo.engine: hierarchical`, live since 2026-10-28): each club's rating is its
   division's level plus its own deviation, both with an uncertainty. League matches move only the deviations; cups
   and play-offs also move the division levels; J1 clubs use the plain Elo update at K 20 (`hierarchical.top_k`).
   All updates use ClubElo dynamic goal weighting (K·√margin / E[√margin | win]) and the final score incl. extra
   time. `elo.engine: elo` switches back to plain Elo. Backtest v plain Elo: cup log loss −1.34%, league −0.09%;
   J1 at K 20 v 30 a further −0.23% on J1 matches.
3. **Tilts** — each team's tendency for high/low-scoring games (0.98/0.02 exponential update).
4. **Match prices** — 1X2 from Elo-Davidson with the draw parameter calibrated per league; expected goals from the
   lambda lookup × both teams' tilts; O/U 2.5 from Poisson. Configurable power-method margins (1.08 for 1X2, 1.08
   for O/U), rounded to the price ladder.
5. **Season simulation** — 20,000 runs of the remaining fixtures (Davidson outcome probabilities,
   Poisson scorelines; ranked on points, GD, goals scored). Outright markets are defined per league in
   the config and priced yes/no with a 1.08 margin.
6. **Cups** — knockout simulation with one- or two-legged ties, seeded entrants, extra time and per-team penalty
   skill. Cup HFA/draw rate/goals are fitted separately for early and late rounds, adjusted for the rating gap and
   anchored to J1–J3 (`output/japan/cup_params.csv`).
7. **Report** — `output/japan/README.md`, `report.xlsx`, `ratings.csv` and per-competition CSVs.

### Optional model switches (`config.yaml`)

Current values are listed in [docs/PIPELINE.md](docs/PIPELINE.md#current-model-settings), which also covers the
engine's other options, all off for Japan (recency-weighted HFA, Davidson scale, stakes, split-table phases).

| Setting | Options | Effect |
|---|---|---|
| `elo.engine` | `elo` / `hierarchical` | Plain Elo, or division level + club deviation (live). Settings under `hierarchical:`. |
| `hierarchical.top_k` | number (Japan 20) | K of J1 clubs in every match they play; unset = `elo.default_k` (30). |
| `elo.lambda_match` | `win_prob` (original) / `expected_score` | How an Elo edge becomes Poisson goal rates for the ClubElo goal weights. `expected_score` stops counting draws twice. |
| `prediction_params.method` | `history` / `shrink` | Pricing HFA, draw rate and goals/game from all seasons, or this season blended with earlier ones (`prior_matches`). |
| `prediction_params.draw_model` | `league_rate` / `calibrated` | Davidson draw parameter set to the league draw rate, or calibrated per league so average predicted draws match observed draws. |
| `cup_params.method` | `first_round` / `anchored` | Cup pricing parameters from the first remaining round's raw history, or rating-adjusted early/late fits blended with the J1–J3 anchor (`anchor_matches`). |
| `penalties.method` | `coin` / `skill` | Cup shoot-outs 50/50, or per-team conversion rates from `data/japan/penalties.csv` (5 kicks each, then sudden death). |

## Layout

```
data/japan/          config.yaml, results.csv, initial_elo.csv, penalties.csv, fixtures/, cups/   <- you edit these
output/japan/        generated: README.md report, report.xlsx, ratings, parameters, per-competition CSVs
soccermodel/         the model code (pipeline.py runs the steps; elo.py / hierarchical.py are the rating engines)
docs/PIPELINE.md     how the model works, with backtests
tests/               prices v the old workbooks, simulations, cups, hierarchical engine
tools/               one-off import from the old Excel files
.github/workflows/   matchday.yml: check, test, update and commit output/ on every push to data/ or code
```

## License

The code is licensed under the [Apache License 2.0](LICENSE). The match data in `data/` is **not** covered by that
license and isn't licensed for reuse: all rights reserved. See [NOTICE](NOTICE).
