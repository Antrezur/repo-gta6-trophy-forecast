"""Forecast how many GTA VI players earn each trophy in week one (Python 3.13, stdlib only).

Model: a trophy's earn rate is earners / players-tracked, and the two move for
different reasons. New players joining Exophase dilute every rate; the players
already tracked earn trophies as they play. A single trend on the rate itself
cannot represent the dip-then-rise that makes, so this program fits the two
counts separately at run time from the case's history, extrapolates both to the
target, and divides:
  - players tracked N(t): power-law fit ln N = a + b ln(1+t) and a saturating
    log-logistic fit N = N0 + (Ninf-N0)/(1+(t50/t)^k); the better in-sample fit
    (log space) wins. N(T) is clamped to [N_last, 5,000,000].
  - earners E_i(t) per trophy: power-law fit ln(E+0.5) = a + b ln(1+t) and a
    saturating Weibull fit E = A (1-exp(-((t+2)/tau)^k)); the better in-sample
    fit (log space) wins. E(T) is floored at the most earners seen (earners do
    not decrease) and capped at 0.98 N(T).
  - median rate = 100 E(T)/N(T), unshrunk (a shrink toward the last rate was
    tested and cost ~0.01 loss in every backtest world).
  - spread (logit) = 0.5 * sqrt(random-walk^2 + (1.2*fit-residual)^2),
    floored at 0.4 and capped at 3.0. The 0.5 multiplier is a hedge: the
    synthetic optimum is ~0.4, and the extra width covers model-form error
    the backtests cannot see (the fit-residual term widens it automatically
    when the real history fits worse than the synthetic ones). Extrapolation
    disagreement terms were tested and removed (they cost loss in every world).
    A trophy at ~100% in >=90% of reads gets a 0.15 spread around 100.
History reads repeat while Exophase has not refreshed, so fitting uses the
distinct states only (first appearance time each), one vote per state.
Deterministic: pure functions of the case file, fixed grids, no randomness,
no clock. Reads $MAKEFUN_INPUT_DIR else /input, writes $MAKEFUN_OUTPUT_DIR
else /output, one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SHRINK = 0
SPREAD_MULT = 0.5
DIS_E_COEF = 0
DIS_N_COEF = 0
N_CAP = 5_000_000   # Exophase cannot track more PSN players than this in week one
E_OFFSET = 2.0      # hours of head start in the Weibull earners fit
SPREAD_FLOOR = 0.4
SPREAD_CAP = 3.0


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def linfit(xs, ys):
    """Least squares y = a + b x. Returns (a, b, rmse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    rmse = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rmse


def distinct_states(case):
    """Collapse consecutive identical reads to one state each (first-appearance
    time). Returns (times_h, Ns, rates_per_trophy) with times in hours since
    the first read, plus the last read's rates."""
    hist = case["history"]
    t0 = parse_iso(hist[0]["at"])
    times, ns, rates = [], [], []
    prev = None
    for r in hist:
        key = (r["players_tracked"], tuple(r["rates"]))
        if key == prev:
            continue
        prev = key
        times.append((parse_iso(r["at"]) - t0).total_seconds() / 3600.0)
        ns.append(float(r["players_tracked"]))
        rates.append(list(r["rates"]))
    return times, ns, rates, t0


def fit_n(times, ns, t_target):
    """Forecast players tracked at t_target.
    Returns (N_main, N_alt, rmse_log): the better in-sample fit's forecast, the
    other candidate's forecast (for the disagreement spread), and the log RMSE."""
    if len(times) < 3 or max(ns) <= min(ns) * 1.001:
        return ns[-1], ns[-1], 0.0
    xs = [math.log1p(max(t, 0.0)) for t in times]
    ys = [math.log(n) for n in ns]
    a, b, rmse_pw = linfit(xs, ys)
    n_pw = math.exp(a + b * math.log1p(max(t_target, 0.0)))
    # Saturating log-logistic: N = N0 + (Ninf-N0)/(1+(t50/t)^k), grid (k, t50).
    best = (float("inf"), None)
    n0 = ns[0]
    for k in (1.0, 1.3, 1.69, 2.0, 2.5):
        for t50 in (20, 40, 80, 123, 180, 260, 400):
            gs = [1.0 / (1.0 + (t50 / max(t, 0.05)) ** k) for t in times]
            num = sum((n - n0) * g / (n * n) for n, g in zip(ns, gs))
            den = sum(g * g / (n * n) for n, g in zip(ns, gs))
            if den <= 0:
                continue
            ninf = n0 + num / den
            if ninf <= n0:
                continue
            errs = [math.log((n0 + (ninf - n0) * g) / n) for n, g in zip(ns, gs)]
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
            if rmse < best[0]:
                best = (rmse, (k, t50, ninf))
    n_sat = None
    if best[1] is not None:
        _, (k, t50, ninf) = best
        g = 1.0 / (1.0 + (t50 / max(t_target, 0.05)) ** k)
        n_sat = n0 + (ninf - n0) * g
    n_pw_c = min(max(n_pw, ns[-1]), N_CAP)
    if n_sat is not None and best[0] < rmse_pw:
        return min(max(n_sat, ns[-1]), N_CAP), n_pw_c, best[0]
    return n_pw_c, (min(max(n_sat, ns[-1]), N_CAP) if n_sat is not None else n_pw_c), rmse_pw


def fit_e(times, es, t_target):
    """Forecast earners at t_target.
    Returns (E_main, E_alt, rmse_log): the better in-sample fit's forecast, the
    other candidate's forecast, and the log RMSE."""
    e_max = max(es)
    if len(times) < 3 or (max(es) - min(es)) <= 0.0002 * max(es):
        return e_max, e_max, 0.0
    xs = [math.log1p(max(t, 0.0)) for t in times]
    ys = [math.log(e + 0.5) for e in es]
    a, b, rmse_pw = linfit(xs, ys)
    e_pw = math.exp(a + b * math.log1p(max(t_target, 0.0))) - 0.5
    # Saturating Weibull: E = A (1-exp(-((t+c)/tau)^k)), grid (k, tau).
    best = (float("inf"), None)
    for k in (0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        tau = 2.0
        while tau <= 3000.0:
            fs = [1.0 - math.exp(-((max(t, 0.0) + E_OFFSET) / tau) ** k) for t in times]
            num = sum(e * f / ((e + 1.0) ** 2) for e, f in zip(es, fs))
            den = sum(f * f / ((e + 1.0) ** 2) for e, f in zip(es, fs))
            if den > 0:
                amp = num / den
                if amp > 0:
                    errs = [math.log((amp * f + 0.5) / (e + 0.5)) for e, f in zip(es, fs)]
                    rmse = math.sqrt(sum(v * v for v in errs) / len(errs))
                    if rmse < best[0]:
                        best = (rmse, (k, tau, amp))
            tau *= 1.6
    e_sat = None
    if best[1] is not None:
        _, (k, tau, amp) = best
        e_sat = amp * (1.0 - math.exp(-((max(t_target, 0.0) + E_OFFSET) / tau) ** k))
    e_pw_c = max(e_pw, e_max)
    if e_sat is not None and best[0] < rmse_pw:
        return max(e_sat, e_max), e_pw_c, best[0]
    return e_pw_c, (max(e_sat, e_max) if e_sat is not None else e_pw_c), rmse_pw


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    times, ns, rates, t0 = distinct_states(case)
    t_target = (parse_iso(case["target"]) - t0).total_seconds() / 3600.0
    horizon_d = case["horizon_minutes"] / 1440.0
    base_spread = 0.5 * math.sqrt(horizon_d)
    n_t, n_alt, n_rmse = fit_n(times, ns, t_target)
    dis_n = abs(math.log(max(n_t, 1.0) / max(n_alt, 1.0))) if n_alt > 0 else 0.0
    last_rates = case["history"][-1]["rates"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        es = []
        for j, r in enumerate(rates):
            v = r[i]
            es.append(None if v is None else v / 100.0 * ns[j])
        known = [(t, e) for t, e in zip(times, es) if e is not None]
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        # Universal trophy: (nearly) every tracked player has earned it in
        # (nearly) every read. Its rate stays ~100%; only a small spread.
        seen = [r[i] for r in rates if r[i] is not None]
        if seen and sum(1 for v in seen if v >= 99.99) >= 0.9 * len(seen):
            rows.append({"id": trophy["id"], "q": [
                round4(100 * sigmoid(logit(100.0) + z * 0.15)) for z in Z]})
            continue
        if len(known) < 3:
            e_t = e_alt = (known[-1][1] if known else 0.0)
            e_rmse = 0.0
        else:
            kt = [t for t, _ in known]
            ke = [e for _, e in known]
            e_t, e_alt, e_rmse = fit_e(kt, ke, t_target)
        e_t = min(max(e_t, max([e for _, e in known]) if known else 0.0), 0.98 * n_t)
        e_alt = min(max(e_alt, max([e for _, e in known]) if known else 0.0), 0.98 * n_t)
        r_en = 100.0 * e_t / n_t if n_t > 0 else last_rate
        med_logit = (1 - SHRINK) * logit(r_en) + SHRINK * logit(last_rate)
        # Spread: random walk + in-sample fit residual. The fit-residual term
        # widens the band automatically when the real history fits worse than
        # the synthetic ones did. (Extrapolation-disagreement terms were
        # tested and set to zero weight: they cost loss in every world.)
        fit_rmse = max(min(e_rmse, 1.2), 0.0)
        dis_e = abs(math.log(max(e_t, 0.5) / max(e_alt, 0.5)))
        spread = math.sqrt(base_spread ** 2 + (1.2 * fit_rmse) ** 2
                           + (DIS_E_COEF * dis_e) ** 2 + (DIS_N_COEF * dis_n) ** 2) * SPREAD_MULT
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        qs = [round4(100 * sigmoid(med_logit + z * spread)) for z in Z]
        qs = [max(0.01, min(100.0, q)) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
