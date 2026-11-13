# Odds shadow: J1 League live prices corrected by past closing odds

Run 2026-11-13 22:45 UTC. Live engine `hierarchical`; nothing here changes the published outputs one folder up.

Closing odds to 2026-11-08 (4037 of 4040 J1 League matches priced by the market). Step k 0.3; tilt fitted on 3423 matches: a +0.046 (live logit), b 0.945 (club state difference).

## Score so far

Each side's last price before kick-off, 1X2 log loss on 90-minute results (negative change = shadow better; logging started with the first run).

_none_

## Club states

Positive: the closing market has rated the club above the model in its recent matches (logit; `elo_points` is the same in Elo points).

| team | state | elo_points | last_odds |
|---|---|---|---|
| Tokyo | 0.496 | 86 | 2026-11-07 |
| Sanfrecce Hiroshima | 0.367 | 64 | 2026-11-08 |
| Urawa Reds | 0.277 | 48 | 2026-11-08 |
| Kawasaki Frontale | 0.259 | 45 | 2026-11-07 |
| Kashiwa Reysol | 0.24 | 42 | 2026-11-07 |
| Fagiano Okayama | 0.209 | 36 | 2026-11-07 |
| Avispa Fukuoka | 0.193 | 34 | 2026-11-07 |
| Nagoya Grampus | 0.185 | 32 | 2026-11-07 |
| JEF United | 0.108 | 19 | 2026-11-08 |
| Kashima Antlers | 0.056 | 10 | 2026-11-07 |
| Vissel Kobe | 0.036 | 6 | 2026-11-08 |
| Gamba Osaka | 0.0 | 0 | 2026-11-08 |
| Machida Zelvia | -0.024 | -4 | 2026-11-08 |
| Tokyo Verdy | -0.108 | -19 | 2026-11-07 |
| Yokohama F. Marinos | -0.131 | -23 | 2026-11-07 |
| Shimizu S-Pulse | -0.147 | -26 | 2026-11-07 |
| Mito Hollyhock | -0.151 | -26 | 2026-11-08 |
| Cerezo Osaka | -0.245 | -43 | 2026-11-08 |
| Kyoto Sanga | -0.248 | -43 | 2026-11-07 |
| V-Varen Nagasaki | -0.358 | -62 | 2026-11-07 |

## Largest price changes, unplayed fixtures

| date | home | away | odds1_live | oddsX_live | odds2_live | odds1_shadow | oddsX_shadow | odds2_shadow | prob1_diff | state_home | state_away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2027-02-20 | Tokyo | V-Varen Nagasaki | 1.62 | 4.0 | 4.85 | 1.35 | 4.85 | 9.0 | 0.1338 | 0.496 | -0.358 |
| 2027-03-13 | Tokyo | Cerezo Osaka | 1.87 | 3.75 | 3.75 | 1.5 | 4.2 | 6.0 | 0.1295 | 0.496 | -0.245 |
| 2027-05-09 | Sanfrecce Hiroshima | V-Varen Nagasaki | 1.61 | 4.0 | 5.0 | 1.37 | 4.75 | 8.5 | 0.1157 | 0.367 | -0.358 |
| 2027-04-24 | Tokyo | Yokohama F. Marinos | 1.95 | 3.6 | 3.45 | 1.61 | 4.0 | 5.0 | 0.1133 | 0.496 | -0.131 |
| 2027-04-10 | Kawasaki Frontale | V-Varen Nagasaki | 1.9 | 3.6 | 3.5 | 1.6 | 4.0 | 5.25 | 0.111 | 0.259 | -0.358 |
| 2026-11-21 | Cerezo Osaka | Tokyo | 3.05 | 3.6 | 2.1 | 4.6 | 3.85 | 1.66 | -0.1101 | -0.245 | 0.496 |
| 2027-03-06 | Kyoto Sanga | Tokyo | 3.35 | 3.6 | 2.0 | 5.25 | 4.0 | 1.6 | -0.1047 | -0.248 | 0.496 |
| 2027-05-03 | Tokyo | Shimizu S-Pulse | 1.6 | 4.0 | 5.25 | 1.37 | 4.75 | 8.5 | 0.1033 | 0.496 | -0.147 |
| 2027-05-09 | Tokyo | Tokyo Verdy | 1.65 | 4.0 | 4.75 | 1.42 | 4.5 | 7.25 | 0.1014 | 0.496 | -0.108 |
| 2027-03-10 | Nagoya Grampus | V-Varen Nagasaki | 2.15 | 3.6 | 3.0 | 1.75 | 3.8 | 4.0 | 0.0998 | 0.185 | -0.358 |
