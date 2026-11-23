# Odds shadow: J1 League live prices corrected by past closing odds

Run 2026-11-23 20:26 UTC. Live engine `hierarchical`; nothing here changes the published outputs one folder up.

Closing odds to 2026-11-08 (4037 of 4050 J1 League matches priced by the market). Step k 0.3; tilt fitted on 3423 matches: a +0.048 (live logit), b 0.944 (club state difference).

## Score so far

Each side's last price before kick-off, 1X2 log loss on 90-minute results (negative change = shadow better; logging started with the first run).

| competition | matches | live | shadow | change % | se % |
|---|---|---|---|---|---|
| all | 8 | 1.21398 | 1.23996 | 2.14 | 4.78 |
| j1 | 8 | 1.21398 | 1.23996 | 2.14 | 4.78 |

## Club states

Positive: the closing market has rated the club above the model in its recent matches (logit; `elo_points` is the same in Elo points).

| team | state | elo_points | last_odds |
|---|---|---|---|
| Tokyo | 0.497 | 86 | 2026-11-07 |
| Sanfrecce Hiroshima | 0.366 | 64 | 2026-11-08 |
| Urawa Reds | 0.277 | 48 | 2026-11-08 |
| Kawasaki Frontale | 0.258 | 45 | 2026-11-07 |
| Kashiwa Reysol | 0.241 | 42 | 2026-11-07 |
| Fagiano Okayama | 0.21 | 36 | 2026-11-07 |
| Avispa Fukuoka | 0.193 | 34 | 2026-11-07 |
| Nagoya Grampus | 0.185 | 32 | 2026-11-07 |
| JEF United | 0.109 | 19 | 2026-11-08 |
| Kashima Antlers | 0.056 | 10 | 2026-11-07 |
| Vissel Kobe | 0.036 | 6 | 2026-11-08 |
| Gamba Osaka | -0.0 | 0 | 2026-11-08 |
| Machida Zelvia | -0.024 | -4 | 2026-11-08 |
| Tokyo Verdy | -0.109 | -19 | 2026-11-07 |
| Yokohama F. Marinos | -0.13 | -23 | 2026-11-07 |
| Shimizu S-Pulse | -0.147 | -26 | 2026-11-07 |
| Mito Hollyhock | -0.151 | -26 | 2026-11-08 |
| Cerezo Osaka | -0.246 | -43 | 2026-11-08 |
| Kyoto Sanga | -0.249 | -43 | 2026-11-07 |
| V-Varen Nagasaki | -0.358 | -62 | 2026-11-07 |

## Largest price changes, unplayed fixtures

| date | home | away | odds1_live | oddsX_live | odds2_live | odds1_shadow | oddsX_shadow | odds2_shadow | prob1_diff | state_home | state_away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2027-02-20 | Tokyo | V-Varen Nagasaki | 1.62 | 4.0 | 5.0 | 1.35 | 5.0 | 9.0 | 0.1329 | 0.497 | -0.358 |
| 2027-03-13 | Tokyo | Cerezo Osaka | 1.8 | 3.75 | 4.0 | 1.47 | 4.35 | 6.5 | 0.1272 | 0.497 | -0.246 |
| 2027-05-09 | Sanfrecce Hiroshima | V-Varen Nagasaki | 1.62 | 4.0 | 4.85 | 1.38 | 4.75 | 8.0 | 0.1169 | 0.366 | -0.358 |
| 2027-04-24 | Tokyo | Yokohama F. Marinos | 1.9 | 3.6 | 3.6 | 1.57 | 4.0 | 5.25 | 0.1125 | 0.497 | -0.13 |
| 2027-04-10 | Kawasaki Frontale | V-Varen Nagasaki | 1.95 | 3.6 | 3.5 | 1.6 | 4.0 | 5.0 | 0.1112 | 0.258 | -0.358 |
| 2027-03-06 | Kyoto Sanga | Tokyo | 3.35 | 3.6 | 2.0 | 5.25 | 4.0 | 1.57 | -0.1043 | -0.249 | 0.497 |
| 2027-05-03 | Tokyo | Shimizu S-Pulse | 1.57 | 4.0 | 5.25 | 1.36 | 4.85 | 8.5 | 0.1027 | 0.497 | -0.147 |
| 2027-05-09 | Tokyo | Tokyo Verdy | 1.62 | 4.0 | 5.0 | 1.4 | 4.6 | 7.5 | 0.1005 | 0.497 | -0.109 |
| 2027-03-10 | Nagoya Grampus | V-Varen Nagasaki | 2.15 | 3.6 | 2.95 | 1.8 | 3.75 | 4.0 | 0.0998 | 0.185 | -0.358 |
| 2026-12-19 | V-Varen Nagasaki | Urawa Reds | 2.95 | 3.6 | 2.2 | 4.15 | 3.8 | 1.75 | -0.0985 | -0.358 | 0.277 |
