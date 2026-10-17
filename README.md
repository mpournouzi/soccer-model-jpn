# soccer-model

Elo ratings, match odds and season/cup simulations for Japanese football, rebuilt automatically
from a results CSV. Consolidates the old `elo_upd_jpn.py`, `hfa claude/main.py`, `jpn.py` and the
cup/season-sim workbooks into one pipeline.

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

## Occasional jobs

| When | What |
|---|---|
| Cup draw announced | Add the ties to `data/japan/cups/<cup>_2026-27.csv` (`round,tie,leg,date,home,away`). Later rounds may refer to winners as `W:R16-3`. Rounds without rows are drawn at random in each simulation. |
| New season | Add the new fixture files, update `season`, `leagues` (fixtures, markets) and `cups` (rounds) in `data/japan/config.yaml`. |
| New club appears | Add it with a starting rating to `data/japan/initial_elo.csv`. |
| Name spelled two ways | Add it under `aliases:` in the config. |

## What the run does

1. **Parameters** — per tournament: HFA = 400·log10(home wins / away wins), draw rate and goals per game
   from 90-minute scores over all history (`output/japan/tournament_params.csv`). `hfa: auto` in the config
   uses these; a number overrides (0 for neutral venues).
2. **Elo** — chronological updates with ClubElo dynamic goal weighting (K·√margin / E[√margin | win],
   from Poisson rates fitted to each tournament's goals per game). Uses the final score incl. extra time.
3. **Tilts** — each team's tendency for high/low-scoring games (0.98/0.02 exponential update).
4. **Match prices** — 1X2 from Elo-Davidson; expected goals from the lambda lookup × both teams' tilts;
   O/U 2.5 from Poisson. Configurable power-method margins (1.08 for 1X2, 1.08 for O/U) and rounded to the price ladder.
5. **Season simulation** — 20,000 runs of the remaining fixtures (Davidson outcome probabilities,
   Poisson scorelines; ranked on points, GD, goals scored). Outright markets are defined per league in
   the config and priced yes/no with a 1.08 margin.
6. **Cups** — knockout simulation with one- or two-legged ties, extra time and penalties.

### Optional model switches (`config.yaml`)

| Setting | Options | Effect |
|---|---|---|
| `elo.lambda_match` | `win_prob` (original) / `expected_score` | How an Elo edge becomes Poisson goal rates for the ClubElo goal weights. `expected_score` stops counting draws twice. |
| `prediction_params.method` | `history` / `shrink` | Pricing HFA, draw rate and goals/game from all seasons, or this season blended with earlier ones (`prior_matches`). |
| `prediction_params.draw_model` | `league_rate` / `calibrated` | Davidson draw parameter set to the league draw rate, or calibrated per league so average predicted draws match observed draws. |
| `penalties.method` | `coin` / `skill` | Cup shoot-outs 50/50, or per-team conversion rates from `data/japan/penalties.csv` (5 kicks each, then sudden death). |

## Layout

```
data/japan/          config.yaml, results.csv, initial_elo.csv, fixtures/, cups/   <- you edit these
output/japan/        generated: README.md report, report.xlsx, ratings, per-competition CSVs
soccermodel/         the model code
tests/               checks against prices/probabilities from the old workbooks
tools/               one-off import from the old Excel files
```
