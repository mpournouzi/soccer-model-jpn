# Odds shadow: live prices corrected by past closing odds

Run 2026-11-25 16:44 UTC. Live engine `hierarchical`; nothing here changes the published outputs one folder up.

States from 16746 league matches with closing odds (step k 0.3; a state unchanged for 365 days counts as 0). Tilts, fitted offline:

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
| all | 25 | 1.08729 | 1.10972 | 2.06 | 2.8 |  |
| j1 | 15 | 1.1183 | 1.11966 | 0.12 | 3.22 |  |
| j2 | 10 | 1.04076 | 1.09482 | 5.19 | 5.24 |  |
| with free closing odds | 8 | 1.21398 | 1.23996 |  |  | 1.2346 |

## Club states

Positive: the closing market has rated the club above the model in its recent matches (logit; `elo_points` is the same in Elo points). Clubs with a shadowed fixture to come.

| team | state | elo_points | last_odds |
|---|---|---|---|
| Tokyo | 0.703 | 122 | 2026-11-25 |
| Sagan Tosu | 0.478 | 83 | 2026-11-25 |
| Urawa Reds | 0.477 | 83 | 2026-11-25 |
| Sanfrecce Hiroshima | 0.453 | 79 | 2026-11-25 |
| Kawasaki Frontale | 0.426 | 74 | 2026-11-25 |
| Nagoya Grampus | 0.391 | 68 | 2026-11-25 |
| Kashiwa Reysol | 0.317 | 55 | 2026-11-20 |
| Vegalta Sendai | 0.279 | 48 | 2026-11-25 |
| Avispa Fukuoka | 0.276 | 48 | 2026-11-25 |
| Iwaki | 0.244 | 42 | 2026-11-25 |
| Gamba Osaka | 0.243 | 42 | 2026-11-21 |
| JEF United | 0.232 | 40 | 2026-11-25 |
| Fagiano Okayama | 0.211 | 37 | 2026-11-25 |
| Jubilo Iwata | 0.177 | 31 | 2026-11-25 |
| Omiya Ardija | 0.149 | 26 | 2026-11-25 |
| Kashima Antlers | 0.142 | 25 | 2026-11-20 |
| Vissel Kobe | 0.133 | 23 | 2026-11-20 |
| Yokohama | 0.119 | 21 | 2026-11-25 |
| Tegevajaro Miyazaki | 0.111 | 19 | 2026-11-25 |
| Vanraure Hachinohe | 0.058 | 10 | 2026-11-25 |
| Machida Zelvia | 0.051 | 9 | 2026-11-21 |
| Imabari | 0.038 | 7 | 2026-11-25 |
| Shimizu S-Pulse | 0.008 | 1 | 2026-11-25 |
| Mito Hollyhock | -0.005 | -1 | 2026-11-25 |
| Oita Trinita | -0.018 | -3 | 2026-11-25 |
| Tokyo Verdy | -0.039 | -7 | 2026-11-25 |
| Tokushima Vortis | -0.06 | -10 | 2026-11-25 |
| Yokohama F. Marinos | -0.095 | -17 | 2026-11-25 |
| Kyoto Sanga | -0.106 | -18 | 2026-11-21 |
| Montedio Yamagata | -0.128 | -22 | 2026-11-25 |
| Consadole Sapporo | -0.147 | -26 | 2026-11-25 |
| Kataller Toyama | -0.155 | -27 | 2026-11-25 |
| Tochigi City | -0.181 | -31 | 2026-11-25 |
| Cerezo Osaka | -0.209 | -36 | 2026-11-25 |
| V-Varen Nagasaki | -0.21 | -36 | 2026-11-25 |
| Shonan Bellmare | -0.23 | -40 | 2026-11-25 |
| Fujieda MYFC | -0.259 | -45 | 2026-11-25 |
| Blaublitz Akita | -0.352 | -61 | 2026-11-25 |
| Ventforet Kofu | -0.37 | -64 | 2026-11-25 |
| Albirex Niigata | -0.492 | -85 | 2026-11-25 |

## Largest price changes, unplayed fixtures

| competition | date | home | away | odds1_live | oddsX_live | odds2_live | odds1_shadow | oddsX_shadow | odds2_shadow | prob1_diff | state_home | state_away |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| j2 | 2027-02-20 | Sagan Tosu | Albirex Niigata | 2.45 | 3.25 | 2.75 | 1.75 | 3.5 | 4.6 | 0.1667 | 0.478 | -0.492 |
| j2 | 2027-05-03 | Sagan Tosu | Blaublitz Akita | 2.15 | 3.25 | 3.2 | 1.65 | 3.6 | 5.25 | 0.1491 | 0.478 | -0.352 |
| j1 | 2027-03-13 | Tokyo | Cerezo Osaka | 1.8 | 3.75 | 4.0 | 1.44 | 4.5 | 7.0 | 0.1469 | 0.703 | -0.209 |
| j2 | 2027-05-03 | Vegalta Sendai | Albirex Niigata | 2.15 | 3.3 | 3.25 | 1.67 | 3.6 | 5.25 | 0.1399 | 0.279 | -0.492 |
| j1 | 2027-04-24 | Tokyo | Yokohama F. Marinos | 1.95 | 3.6 | 3.5 | 1.55 | 4.15 | 5.75 | 0.1353 | 0.703 | -0.095 |
| j1 | 2027-02-20 | Tokyo | V-Varen Nagasaki | 1.61 | 4.0 | 5.0 | 1.33 | 5.0 | 9.5 | 0.134 | 0.703 | -0.21 |
| j2 | 2027-05-15 | Sagan Tosu | Shonan Bellmare | 2.5 | 3.25 | 2.75 | 1.9 | 3.4 | 3.85 | 0.1223 | 0.478 | -0.23 |
| j1 | 2026-11-29 | Yokohama F. Marinos | Tokyo | 2.9 | 3.6 | 2.2 | 4.5 | 3.85 | 1.7 | -0.117 | -0.095 | 0.703 |
| j2 | 2027-03-06 | Sagan Tosu | Montedio Yamagata | 2.0 | 3.35 | 3.6 | 1.62 | 3.75 | 5.5 | 0.1163 | 0.478 | -0.128 |
| j2 | 2026-12-13 | Jubilo Iwata | Albirex Niigata | 2.55 | 3.25 | 2.62 | 2.0 | 3.35 | 3.6 | 0.1139 | 0.177 | -0.492 |
