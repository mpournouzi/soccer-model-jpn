# Odds shadow: live prices corrected by past closing odds

Run 2026-11-24 18:45 UTC. Live engine `hierarchical`; nothing here changes the published outputs one folder up.

States from 16719 league matches with closing odds (step k 0.3; a state unchanged for 365 days counts as 0). Tilts, fitted offline:

| target | a (live logit) | b (state difference) |
|---|---|---|
| j1 | 0.038 | 0.916 |
| j2 | 0.147 | 0.921 |
| emperors_cup | 0.035 | 0.803 |
| league_cup | 0.035 | 0.803 |

## Score so far

Each side's last price before kick-off, 1X2 log loss on 90-minute results (negative change = shadow better; logging started with the first run).

| competition | matches | live | shadow | change % | se % | market |
|---|---|---|---|---|---|---|
| all | 8 | 1.21398 | 1.23996 | 2.14 | 4.78 |  |
| j1 | 8 | 1.21398 | 1.23996 | 2.14 | 4.78 |  |
| with free closing odds | 8 | 1.21398 | 1.23996 |  |  | 1.2346 |

## Club states

Positive: the closing market has rated the club above the model in its recent matches (logit; `elo_points` is the same in Elo points). Clubs with a shadowed fixture to come.

| team | state | elo_points | last_odds |
|---|---|---|---|
| Tokyo | 0.597 | 104 | 2026-11-21 |
| Sanfrecce Hiroshima | 0.558 | 97 | 2026-11-21 |
| Sagan Tosu | 0.544 | 95 | 2026-11-21 |
| Kawasaki Frontale | 0.421 | 73 | 2026-11-21 |
| Urawa Reds | 0.372 | 65 | 2026-11-21 |
| Nagoya Grampus | 0.351 | 61 | 2026-11-21 |
| Kashiwa Reysol | 0.317 | 55 | 2026-11-20 |
| Jubilo Iwata | 0.286 | 50 | 2026-11-21 |
| JEF United | 0.272 | 47 | 2026-11-20 |
| Avispa Fukuoka | 0.268 | 47 | 2026-11-21 |
| Gamba Osaka | 0.243 | 42 | 2026-11-21 |
| Vegalta Sendai | 0.237 | 41 | 2026-11-21 |
| Fagiano Okayama | 0.214 | 37 | 2026-11-21 |
| Omiya Ardija | 0.195 | 34 | 2026-11-21 |
| Tegevajaro Miyazaki | 0.182 | 32 | 2026-11-21 |
| Yokohama | 0.161 | 28 | 2026-11-21 |
| Kashima Antlers | 0.142 | 25 | 2026-11-20 |
| Vissel Kobe | 0.133 | 23 | 2026-11-20 |
| Kataller Toyama | 0.083 | 14 | 2026-11-21 |
| Machida Zelvia | 0.051 | 9 | 2026-11-21 |
| Iwaki | 0.006 | 1 | 2026-11-21 |
| Shimizu S-Pulse | 0.005 | 1 | 2026-11-20 |
| Mito Hollyhock | -0.008 | -1 | 2026-11-20 |
| Vanraure Hachinohe | -0.032 | -6 | 2026-11-21 |
| Imabari | -0.033 | -6 | 2026-11-21 |
| Tokyo Verdy | -0.036 | -6 | 2026-11-21 |
| Oita Trinita | -0.084 | -15 | 2026-11-21 |
| Yokohama F. Marinos | -0.087 | -15 | 2026-11-21 |
| Cerezo Osaka | -0.104 | -18 | 2026-11-21 |
| Kyoto Sanga | -0.106 | -18 | 2026-11-21 |
| Tokushima Vortis | -0.107 | -19 | 2026-11-22 |
| Consadole Sapporo | -0.161 | -28 | 2026-11-21 |
| Tochigi City | -0.167 | -29 | 2026-11-21 |
| V-Varen Nagasaki | -0.205 | -36 | 2026-11-21 |
| Montedio Yamagata | -0.237 | -41 | 2026-11-21 |
| Shonan Bellmare | -0.239 | -42 | 2026-11-22 |
| Fujieda MYFC | -0.245 | -43 | 2026-11-21 |
| Ventforet Kofu | -0.361 | -63 | 2026-11-21 |
| Blaublitz Akita | -0.366 | -64 | 2026-11-21 |
| Albirex Niigata | -0.402 | -70 | 2026-11-21 |

## Largest price changes, unplayed fixtures

| competition | date | home | away | odds1_live | oddsX_live | odds2_live | odds1_shadow | oddsX_shadow | odds2_shadow | prob1_diff | state_home | state_away |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| j2 | 2027-02-20 | Sagan Tosu | Albirex Niigata | 2.5 | 3.25 | 2.75 | 1.8 | 3.5 | 4.35 | 0.1619 | 0.544 | -0.402 |
| j2 | 2027-05-03 | Sagan Tosu | Blaublitz Akita | 2.25 | 3.25 | 3.1 | 1.65 | 3.6 | 5.25 | 0.1608 | 0.544 | -0.366 |
| j2 | 2027-03-06 | Sagan Tosu | Montedio Yamagata | 2.0 | 3.35 | 3.6 | 1.57 | 3.8 | 6.0 | 0.1428 | 0.544 | -0.237 |
| j2 | 2027-05-15 | Sagan Tosu | Shonan Bellmare | 2.55 | 3.2 | 2.65 | 1.9 | 3.4 | 3.85 | 0.1336 | 0.544 | -0.239 |
| j2 | 2027-04-10 | Sagan Tosu | Consadole Sapporo | 2.4 | 3.25 | 2.85 | 1.87 | 3.4 | 4.0 | 0.1238 | 0.544 | -0.161 |
| j2 | 2027-02-20 | Jubilo Iwata | Ventforet Kofu | 1.8 | 3.5 | 4.5 | 1.47 | 4.0 | 7.25 | 0.1228 | 0.286 | -0.361 |
| j1 | 2027-02-20 | Tokyo | V-Varen Nagasaki | 1.62 | 4.0 | 5.0 | 1.36 | 4.85 | 8.5 | 0.1217 | 0.597 | -0.205 |
| j2 | 2027-04-24 | Sagan Tosu | Tokushima Vortis | 2.15 | 3.25 | 3.25 | 1.72 | 3.6 | 4.75 | 0.1205 | 0.544 | -0.107 |
| j2 | 2027-05-03 | Vegalta Sendai | Albirex Niigata | 2.1 | 3.3 | 3.4 | 1.67 | 3.6 | 5.0 | 0.1201 | 0.237 | -0.402 |
| j2 | 2026-12-13 | Jubilo Iwata | Albirex Niigata | 2.45 | 3.25 | 2.75 | 1.9 | 3.4 | 3.85 | 0.1198 | 0.286 | -0.402 |
