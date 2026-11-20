# GTA VI Trophy Forecast

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
"Forecast how many GTA VI players earn each trophy in week one" ($250, 20 places).

`main.py` (Python 3.13, standard library only) reads a case file of Exophase
trophy earn rates and forecasts each trophy's earn rate at 2026-11-26 00:00 UTC
as five quantiles (5%, 25%, 50%, 75%, 95%). It runs offline, is deterministic
(two runs on the same case give byte-identical output), and takes ~0.3 s per
case (limit 8 s).

## Method

A trophy's earn rate is earners / players-tracked, and the two move for
different reasons: new players joining Exophase dilute every rate, while the
players already tracked earn trophies as they play. A single trend on the rate
itself cannot represent the dip-then-rise that makes, so the program fits the
two counts separately at run time and divides:

- Players tracked N(t): power-law fit vs a saturating log-logistic fit; the
  better in-sample fit wins. N(T) is clamped to [N_last, 5,000,000].
- Earners E_i(t) per trophy: power-law fit vs a saturating Weibull fit; the
  better in-sample fit wins. E(T) is floored at the most earners seen and
  capped at 0.98 N(T).
- Median rate = 100 E(T) / N(T), unshrunk.
- Spread (logit) = 0.5 * sqrt(random-walk^2 + (1.2 * fit-residual)^2),
  floored at 0.4 and capped at 3.0.

## Measured results

Synthetic backtests (final-like cases, cut 21 Nov 11:30 UTC, target 26 Nov
00:00 UTC; worlds calibrated to the real launch-night rates and the live
Exophase page of 20 Nov 07:34 UTC). Mean pinball loss on the logit scale,
lower is better:

| regime | flat | trend | this entry |
|---|---|---|---|
| base | 0.209 | 0.119 | 0.087 |
| slowN | 0.284 | 0.095 | 0.077 |
| fastN | 0.184 | 0.139 | 0.090 |
| fastEarn | 0.262 | 0.125 | 0.077 |
| slowEarn | 0.208 | 0.166 | 0.093 |
| mean | 0.229 | 0.129 | 0.085 |

Beats the trend baseline in 10/10 runs and the flat baseline in 10/10
(34% better than trend, 63% better than flat on average). Under a
misspecified world (earners log-logistic instead of Weibull) it still wins
4/5 regimes (mean 0.099 vs trend 0.117 vs flat 0.256).

Practice cases p00/p01 (kit): output valid (55 trophies, 5 quantiles each),
deterministic, 0.06 s for both. Practice loss cannot be measured locally —
the answers stay on the server — and practice pays nothing.

Honest limits: the backtests are synthetic worlds, not the real final
history (19–21 Nov), which does not exist yet; and the prize bar is the
lowest of the three house entries, including a house AI-model forecast that
cannot be measured in advance. The final is decided on the real rates at
26 Nov 00:00 UTC.
