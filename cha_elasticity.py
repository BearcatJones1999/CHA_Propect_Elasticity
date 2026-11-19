"""
CHA Labor-to-Volume Elasticity
==============================

Estimates how worked hours respond to volume across PROSPECT departments,
using CHA benchmarking data (hospital x cost center x quarter).

Outputs (one Excel workbook, three sheets):
    pooled         Elasticity pooled across all departments (All CHA, Peer set)
    by_department  Elasticity per PROSPECT department (peer set)
    cin_vs_peer    Cincinnati elasticity vs. peers, per department

Every model is run under two specifications:
    cross_section  department/quarter fixed effects only. Compares bigger
                   units to smaller units ("structural" scale economies).
    within_unit    cost-center + quarter fixed effects. Compares each unit
                   to itself over time ("flex" elasticity: what happens to
                   a unit's hours when its own volume moves).

Usage:
    python cha_elasticity.py
    python cha_elasticity.py --input "path/to/Prospect Data Clean.xlsx" --output results.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_INPUT = (
    r"Y:\Nursing_CABIReports\FY27 Budget Files\FY27 Prospect Benchmarking"
    r"\Prospect Data Clean.xlsx"
)
DEFAULT_OUTPUT = "cha_elasticity_results.xlsx"
SHEET_NAME = "Skill Mix - Spreadsheet - clean"

HOME_PREFIX = "CIN-"
PEER_PREFIXES = ("BOS-", "COL", "PHI-", "CIN-", "AUR-", "LAC", "WAS", "SEA", "HOU-")

# Set to None to run every department with enough data
TARGET_DEPTS = ["1010", "1020", "1035", "1041", "1045", "1051",
                "1061", "1071", "1142", "1143", "1151", "1250", "1230"]

MIN_OBS = 50            # minimum rows for a department-level model
MIN_CLUSTERS = 5        # minimum cost centers for clustered SEs to be usable
ALPHA = 0.05            # significance level for CIs and verdicts
SCENARIO_CHANGE = 0.10  # volume change used to translate elasticities into hours

# Raw column names in the CHA extract
RAW = {
    "hospital": "Hospital Code",
    "quarter": "Calendar Year and Quarter",
    "dept": "PROSPECT Dept",
    "dept_desc": "Department Description",
    "ccref": "CCRef Code",
    "volume": "Volume",
    "hours": "Worked Hours in Period",
}

SPECS = {
    # name: fixed effects used when pooling across depts / within one dept
    "cross_section": {"pooled": ["dept", "period"], "dept": ["period"]},
    "within_unit":   {"pooled": ["unit", "period"], "dept": ["unit", "period"]},
}


# =============================================================================
# LOAD + CLEAN
# =============================================================================

def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CHA extract. Excel is cached to parquet next to it for speed."""
    path = Path(path)
    cache = path.with_suffix(".parquet")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_parquet(cache)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, dtype=str, engine="openpyxl")
        try:
            df.to_parquet(cache, index=False)
        except Exception as e:  # caching is a convenience, never a failure
            print(f"[warn] could not write parquet cache: {e}")

    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    missing = [c for c in RAW.values() if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return df[list(RAW.values())].rename(columns={v: k for k, v in RAW.items()})


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _to_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the job-title-level extract to one row per
    hospital x cost center x quarter. Volume and worked hours are reported at
    the cost-center level (repeated on every job-title row), so take the first.
    """
    df = raw.copy()
    for c in ["hospital", "dept", "ccref", "quarter"]:
        df[c] = _to_key(df[c])
    df["volume"] = _to_number(df["volume"])
    df["hours"] = _to_number(df["hours"])
    df["period"] = df["quarter"].str[:4] + "Q" + df["quarter"].str[4:]

    panel = (
        df.groupby(["hospital", "ccref", "dept", "period"], dropna=False)
          .agg(dept_desc=("dept_desc", "first"),
               volume=("volume", "first"),
               hours=("hours", "first"))
          .reset_index()
    )

    # Logs require strictly positive values
    panel = panel[(panel["volume"] > 0) & (panel["hours"] > 0)].copy()
    panel["ln_volume"] = np.log(panel["volume"])
    panel["ln_hours"] = np.log(panel["hours"])
    panel["unit"] = panel["hospital"] + "|" + panel["ccref"]
    panel["is_cin"] = panel["hospital"].str.startswith(HOME_PREFIX).astype(float)
    panel["is_peer_set"] = panel["hospital"].str.startswith(PEER_PREFIXES)
    return panel.reset_index(drop=True)


# =============================================================================
# ESTIMATION ENGINE
# =============================================================================

def _demean(values: np.ndarray, groups: list[np.ndarray],
            tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """
    Absorb fixed effects by alternating projections (subtract group means for
    each FE in turn until stable). Equivalent to including the dummies, but
    fast and memory-light even with thousands of cost centers.
    """
    M = values.astype(float).copy()
    if not groups:
        return M - M.mean(axis=0)
    for _ in range(max_iter):
        prev = M.copy()
        for g in groups:
            counts = np.bincount(g)
            means = np.column_stack([np.bincount(g, weights=M[:, j]) / counts
                                     for j in range(M.shape[1])])
            M -= means[g]
        if len(groups) == 1 or np.abs(M - prev).max() < tol:
            break
    return M


def fe_ols(d: pd.DataFrame, y: str, xs: list[str], fe: list[str],
           cluster: str = "unit"):
    """
    OLS of y on xs with absorbed fixed effects and SEs clustered by cost
    center (the same unit appears every quarter, so its errors are correlated).
    Returns (results, n_obs, n_clusters) or None if the sample is too thin.
    """
    d = d.dropna(subset=[y, *xs, *fe, cluster])
    if "unit" in fe:  # singleton units carry no within-unit information
        d = d[d.groupby("unit")["unit"].transform("size") > 1]
    n_clusters = d[cluster].nunique()
    if len(d) < MIN_OBS or n_clusters < MIN_CLUSTERS:
        return None

    groups = [pd.factorize(d[c])[0] for c in fe]
    M = _demean(d[[y, *xs]].to_numpy(), groups)
    Y = pd.Series(M[:, 0], name=y)
    X = pd.DataFrame(M[:, 1:], columns=xs)
    X = X.loc[:, X.abs().max() > 1e-12]  # drop regressors absorbed by the FEs

    res = sm.OLS(Y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return res, len(d), n_clusters


def _coef(res, name: str) -> dict:
    lo, hi = res.conf_int(alpha=ALPHA).loc[name]
    return {"b": res.params[name], "se": res.bse[name],
            "p": res.pvalues[name], "lo": lo, "hi": hi}


def classify_scale(lo: float, hi: float) -> str:
    if hi < 1:
        return "Economies of scale"
    if lo > 1:
        return "Diseconomies of scale"
    return "Proportional (CI includes 1)"


def hours_response(beta: float, change: float = SCENARIO_CHANGE) -> float:
    """% change in hours for a given % change in volume: (1+g)^beta - 1."""
    return (1 + change) ** beta - 1


def _elasticity_row(res, n, n_clusters, **labels) -> dict:
    c = _coef(res, "ln_volume")
    pct = int(SCENARIO_CHANGE * 100)
    return {
        **labels,
        "n_obs": n,
        "n_units": n_clusters,
        "elasticity": c["b"],
        "se_clustered": c["se"],
        "ci_low": c["lo"],
        "ci_high": c["hi"],
        "scale_behavior": classify_scale(c["lo"], c["hi"]),
        f"hours_chg_if_vol_+{pct}%": hours_response(c["b"]),
        f"hpu_chg_if_vol_+{pct}%": (1 + hours_response(c["b"])) / (1 + SCENARIO_CHANGE) - 1,
        "r2_within": res.rsquared,
    }


# =============================================================================
# ANALYSES
# =============================================================================

def pooled_elasticities(panel: pd.DataFrame) -> pd.DataFrame:
    """One elasticity across all departments, for All CHA and the peer set."""
    samples = {"All CHA": panel, "Peer set": panel[panel["is_peer_set"]]}
    rows = []
    for sample_name, d in samples.items():
        for spec, fe in SPECS.items():
            out = fe_ols(d, "ln_hours", ["ln_volume"], fe["pooled"])
            if out:
                rows.append(_elasticity_row(*out, sample=sample_name, spec=spec))
    return pd.DataFrame(rows)


def _dept_list(panel: pd.DataFrame) -> list[str]:
    return [str(x) for x in TARGET_DEPTS] if TARGET_DEPTS else sorted(panel["dept"].unique())


def _dept_desc(d: pd.DataFrame) -> str:
    m = d["dept_desc"].dropna().mode()
    return m.iloc[0] if len(m) else ""


def dept_elasticities(peer: pd.DataFrame) -> pd.DataFrame:
    """Elasticity per PROSPECT department, both specifications."""
    rows = []
    for dept in _dept_list(peer):
        d = peer[peer["dept"] == dept]
        for spec, fe in SPECS.items():
            out = fe_ols(d, "ln_hours", ["ln_volume"], fe["dept"])
            if out is None:
                print(f"[skip] dept {dept} / {spec}: insufficient data")
                continue
            rows.append(_elasticity_row(*out, dept=dept, dept_desc=_dept_desc(d), spec=spec))
    return pd.DataFrame(rows)


def cin_vs_peer(peer: pd.DataFrame) -> pd.DataFrame:
    """
    ln_hours = a + b*ln_volume + c*CIN + g*(ln_volume x CIN) + FE
      peer elasticity = b, CIN elasticity = b + g, test g = 0.
    The CIN main effect (c) lets Cincinnati have its own staffing level, so
    g measures only a difference in slope. In the within-unit spec c is
    absorbed by the cost-center FE.
    """
    rows = []
    pct = int(SCENARIO_CHANGE * 100)
    for dept in _dept_list(peer):
        d = peer[peer["dept"] == dept].copy()
        if d["is_cin"].sum() == 0 or (d["is_cin"] == 0).sum() == 0:
            print(f"[skip] dept {dept}: needs both CIN and peer rows")
            continue
        d["cin_x_ln_volume"] = d["ln_volume"] * d["is_cin"]
        cin_units = d.loc[d["is_cin"] == 1, "unit"].nunique()
        cin_avg_hours = d.loc[d["is_cin"] == 1, "hours"].mean()

        for spec, fe in SPECS.items():
            xs = ["ln_volume", "cin_x_ln_volume"]
            if spec == "cross_section":
                xs.insert(1, "is_cin")
            out = fe_ols(d, "ln_hours", xs, fe["dept"])
            if out is None or "cin_x_ln_volume" not in out[0].params:
                print(f"[skip] dept {dept} / {spec}: insufficient data")
                continue
            res, n, n_clusters = out
            b = _coef(res, "ln_volume")["b"]
            g = _coef(res, "cin_x_ln_volume")

            if g["p"] >= ALPHA:
                verdict = "No significant difference"
            elif g["b"] < 0:
                verdict = "CIN hours less volume-sensitive than peers"
            else:
                verdict = "CIN hours more volume-sensitive than peers"

            rows.append({
                "dept": dept,
                "dept_desc": _dept_desc(d),
                "spec": spec,
                "n_obs": n,
                "n_units": n_clusters,
                "n_cin_units": cin_units,
                "peer_elasticity": b,
                "cin_elasticity": b + g["b"],
                "diff_cin_minus_peer": g["b"],
                "se_clustered": g["se"],
                "p_value": g["p"],
                "ci_low": g["lo"],
                "ci_high": g["hi"],
                "verdict": verdict,
                # Materiality: extra hours per quarter for an average CIN unit
                # if its volume rose 10%, vs. what peer behavior would imply
                f"cin_excess_hours_per_qtr_if_vol_+{pct}%":
                    cin_avg_hours * (hours_response(b + g["b"]) - hours_response(b)),
                "low_cin_units_warning": cin_units < 3,
            })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    panel = build_panel(load_raw(args.input))
    peer = panel[panel["is_peer_set"]]
    print(f"Panel: {len(panel):,} unit-quarters | {panel['hospital'].nunique()} hospitals | "
          f"peer set: {len(peer):,} rows, {peer['hospital'].nunique()} hospitals")

    results = {
        "pooled": pooled_elasticities(panel),
        "by_department": dept_elasticities(peer),
        "cin_vs_peer": cin_vs_peer(peer),
    }

    with pd.ExcelWriter(args.output, engine="openpyxl") as xw:
        for sheet, df in results.items():
            df.to_excel(xw, sheet_name=sheet, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    for sheet, df in results.items():
        print(f"\n=== {sheet} ===")
        print(df.round(4).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
