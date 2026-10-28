# Shadow run: hierarchical engine v live prices

Run 2026-10-28 08:22 UTC. The published outputs one folder up are the live engine (`elo.engine: elo`); nothing here changes them.

## Score so far

Each engine's last prediction before kick-off, 1X2 log loss on 90-minute results (negative change = shadow better; logging started with the first shadow run).

_none_

## Price differences, unplayed fixtures

| competition | fixtures | mean |prob1 change| | max |
|---|---|---|---|
| emperors_cup | 8 | 0.008 | 0.013 |
| j1 | 260 | 0.008 | 0.028 |
| j2 | 250 | 0.014 | 0.044 |
| j3 | 250 | 0.018 | 0.051 |
| jfl | 170 | 0.044 | 0.272 |
| league_cup | 4 | 0.008 | 0.016 |

Largest:

| competition | date | home | away | odds1_live | oddsX_live | odds2_live | odds1_shadow | oddsX_shadow | odds2_shadow | prob1_diff |
|---|---|---|---|---|---|---|---|---|---|---|
| jfl | 2026-11-22 | Grulla Morioka | J-Lease FC | 1.66 | 3.85 | 4.75 | 3.0 | 3.5 | 2.15 | -0.2717 |
| jfl | 2027-04-03 | TIAMO Hirakata | J-Lease FC | 1.72 | 3.8 | 4.35 | 3.2 | 3.5 | 2.1 | -0.2715 |
| jfl | 2027-03-07 | Azul Claro Numazu | J-Lease FC | 1.53 | 4.15 | 5.75 | 2.6 | 3.45 | 2.45 | -0.2703 |
| jfl | 2026-12-06 | Criacao Shinjuku | J-Lease FC | 1.57 | 4.0 | 5.25 | 2.65 | 3.45 | 2.4 | -0.2611 |
| jfl | 2026-11-04 | J-Lease FC | Tokyo Musashino City | 3.05 | 3.5 | 2.15 | 1.7 | 3.8 | 4.5 | 0.2608 |
| jfl | 2027-05-01 | Maruyasu Okazaki | J-Lease FC | 1.85 | 3.6 | 3.85 | 3.5 | 3.6 | 1.95 | -0.2604 |
| jfl | 2026-11-29 | J-Lease FC | Maruyasu Okazaki | 3.45 | 3.6 | 1.95 | 1.83 | 3.6 | 3.85 | 0.2604 |
| jfl | 2027-03-21 | Veertien Mie | J-Lease FC | 1.5 | 4.15 | 6.0 | 2.45 | 3.45 | 2.6 | -0.2599 |
| jfl | 2027-05-23 | Honda | J-Lease FC | 1.42 | 4.5 | 7.5 | 2.15 | 3.5 | 3.0 | -0.2566 |
| jfl | 2026-11-01 | YSCC | J-Lease FC | 1.8 | 3.75 | 4.0 | 3.35 | 3.6 | 2.0 | -0.2565 |

## Ratings

Average by division (clubs that have played this year):

| last played in | elo_live | elo_shadow | diff |
|---|---|---|---|
| Emperor Cup | 768 | 782 | 14 |
| J1 League | 1465 | 1427 | -38 |
| J2 League | 1233 | 1203 | -29 |
| J3 League | 1109 | 1142 | 34 |
| JFL | 902 | 1000 | 99 |

Largest differences:

| team | last_tournament | last_match | elo_live | elo_shadow | diff |
|---|---|---|---|---|---|
| Tsukuba University | Emperor Cup | 2026-08-19 | 744 | 1122 | 378 |
| J-Lease FC | JFL | 2026-10-24 | 728 | 1060 | 332 |
| Kyoto Sangyo University SC | Emperor Cup | 2026-10-07 | 656 | 911 | 255 |
| Senshu University | Emperor Cup | 2026-08-26 | 612 | 779 | 167 |
| SRC Hiroshima | Emperor Cup | 2026-08-19 | 816 | 681 | -135 |
| Suzuka Point Getters | Emperor Cup | 2026-08-26 | 820 | 929 | 109 |
| Kawasoe Club | Emperor Cup | 2026-08-19 | 778 | 670 | -108 |
| Toyama Shinjo | Emperor Cup | 2026-08-19 | 797 | 689 | -108 |
| Oyama | Emperor Cup | 2026-08-19 | 781 | 674 | -107 |
| Honda Lock | JFL | 2026-10-24 | 833 | 939 | 106 |
