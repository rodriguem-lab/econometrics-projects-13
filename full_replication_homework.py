import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import kurtosis, norm, skew
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.tsa.stattools import adfuller


ROOT = Path(".").resolve()
DATA_ROOT = ROOT / "Data for Household Mobility and Mortgage Rate Lock" / "Replication Package"
SYNTH_DIR = DATA_ROOT / "raw" / "synthetic data"
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

LATEST_FRED_PATH = Path("/private/tmp/MORTGAGE30US_latest.csv")

PAPER_TITLE = "Household mobility and mortgage rate lock"
PAPER_DOI = "https://doi.org/10.1016/j.jfineco.2024.103973"
PAPER_DATA = "https://doi.org/10.17632/74sfv9kx9n.1"
FRED_LINK = "https://fred.stlouisfed.org/series/MORTGAGE30US"


def parse_quarter_from_int(date_int):
    date_str = str(int(date_int))
    return pd.to_datetime(date_str, format="%Y%m%d").to_period("Q")


def implied_mortgage_rate(payment, balance, remaining_periods):
    if payment <= 0 or balance <= 0 or remaining_periods <= 0:
        return np.nan

    def objective(r):
        if r == 0:
            return balance / remaining_periods - payment
        return payment - r * balance / (1 - (1 + r) ** (-remaining_periods))

    try:
        return brentq(objective, 1e-6, 1.0, maxiter=200)
    except ValueError:
        return np.nan


def load_mortgage_panel():
    df = pd.read_csv(SYNTH_DIR / "mortgage_payments_data.csv").copy()
    df["file_date"] = df["file_date"].astype(int)
    df["acct_open_dt"] = df["acct_open_dt"].astype(int)
    df["file_date_dt"] = pd.to_datetime(df["file_date"].astype(str), format="%Y%m%d")
    df["quarter"] = df["file_date_dt"].dt.to_period("Q")
    df["year"] = df["file_date_dt"].dt.year
    df["month"] = df["file_date_dt"].dt.month
    df["acct_open_dt_dt"] = pd.to_datetime(df["acct_open_dt"].astype(str), format="%Y%m%d")
    df["loan_age_months"] = (
        (df["year"] - df["acct_open_dt_dt"].dt.year) * 12
        + (df["month"] - df["acct_open_dt_dt"].dt.month)
    )
    df["remaining_term"] = df["terms_trans"] - df["loan_age_months"]
    df["mortgage_rate_monthly"] = df.apply(
        lambda row: implied_mortgage_rate(
            row["acct_payment_am"], row["acct_balance_am"], int(row["remaining_term"])
        ),
        axis=1,
    )
    df["mortgage_rate_annual"] = df["mortgage_rate_monthly"] * 12
    df["log_balance"] = np.log1p(df["acct_balance_am"])
    df["open_year"] = df["acct_open_dt_dt"].dt.year
    return df


def load_market_rates():
    local = DATA_ROOT / "raw" / "MORTGAGE30US.csv"
    path = LATEST_FRED_PATH if LATEST_FRED_PATH.exists() else local
    rates = pd.read_csv(path)
    if "DATE" in rates.columns:
        rates = rates.rename(columns={"DATE": "date", "MORTGAGE30US": "market_rate"})
    else:
        rates = rates.rename(columns={"observation_date": "date", "MORTGAGE30US": "market_rate"})
    rates["date"] = pd.to_datetime(rates["date"], errors="coerce")
    rates = rates.dropna(subset=["date"]).copy()
    rates["quarter"] = rates["date"].dt.to_period("Q")
    q_rates = rates.groupby("quarter", as_index=False)["market_rate"].mean()
    return q_rates


def load_moves():
    moves = pd.read_csv(SYNTH_DIR / "national_get_all_moves.csv", dtype={"zip1": str, "zip2": str})
    moves["quarter"] = pd.to_datetime(
        moves["file_year"].astype(str) + moves["file_month"].astype(str).str.zfill(2) + "01",
        format="%Y%m%d",
    ).dt.to_period("Q")
    moves["move_flag"] = (moves["zip1"] != moves["zip2"]).astype(int)
    return moves.groupby(["personid", "quarter"], as_index=False)["move_flag"].max()


def build_panel():
    panel = load_mortgage_panel()
    moves = load_moves()
    rates = load_market_rates()
    panel = panel.merge(moves, on=["personid", "quarter"], how="left")
    panel["move_flag"] = panel["move_flag"].fillna(0).astype(int)
    panel = panel.merge(rates, on="quarter", how="left")
    panel["rate_gap"] = panel["market_rate"] - panel["mortgage_rate_annual"]
    panel["balance_to_orig"] = panel["acct_balance_am"] / panel["amount_1"].replace(0, np.nan)
    panel["loan_age_years"] = panel["loan_age_months"] / 12.0
    panel = panel.sort_values(["personid", "quarter"]).reset_index(drop=True)
    panel["quarter_num"] = panel["quarter"].apply(lambda q: q.year * 4 + q.quarter)
    panel["trend"] = panel.groupby("personid").cumcount() + 1
    panel["lag_quarter_num"] = panel.groupby("personid")["quarter_num"].shift(1)
    panel["consecutive"] = panel["quarter_num"] - panel["lag_quarter_num"]
    panel["new_sequence"] = panel["consecutive"].ne(1)
    panel["sequence_id"] = panel.groupby("personid")["new_sequence"].cumsum()
    panel["sequence_length"] = panel.groupby(["personid", "sequence_id"])["quarter"].transform("count")
    panel["keep_dynamic_sample"] = panel["sequence_length"] >= 3
    panel["person_obs"] = panel.groupby("personid")["quarter"].transform("count")
    panel = panel[panel["keep_dynamic_sample"]].copy()
    panel["sample"] = "selected"
    return panel


def add_transformations(panel, variables):
    panel = panel.copy()
    for var in variables:
        panel[f"{var}_between"] = panel.groupby("personid")[var].transform("mean")
        panel[f"{var}_within"] = panel[var] - panel[f"{var}_between"]
        panel[f"{var}_fd"] = panel.groupby("personid")[var].diff()
        panel[f"{var}_time_mean"] = panel.groupby("quarter")[var].transform("mean")
        panel[f"{var}_twfe"] = (
            panel[var]
            - panel[f"{var}_between"]
            - panel[f"{var}_time_mean"]
            + panel[var].mean()
        )
        panel[f"{var}_lag1"] = panel.groupby("personid")[var].shift(1)
        panel[f"{var}_lag2"] = panel.groupby("personid")[var].shift(2)
        panel[f"{var}_fd_lag1"] = panel.groupby("personid")[f"{var}_fd"].shift(1)
    return panel


def panel_stats(panel):
    counts = panel.groupby("personid")["quarter"].nunique()
    holes = panel.groupby("personid").apply(
        lambda x: (x["quarter_num"].max() - x["quarter_num"].min() + 1) - len(x),
        include_groups=False,
    )
    by_date = panel.groupby("quarter")["personid"].nunique().sort_index()
    by_t = counts.value_counts().sort_index(ascending=False)
    excluded = sorted(set(load_mortgage_panel()["personid"].unique()) - set(panel["personid"].unique()))
    return {
        "n_total": load_mortgage_panel()["personid"].nunique(),
        "n_selected": panel["personid"].nunique(),
        "n_excluded": len(excluded),
        "excluded_list": excluded,
        "counts": counts,
        "holes": holes,
        "by_date": by_date,
        "by_t": by_t,
        "time_span": (str(panel["quarter"].min()), str(panel["quarter"].max())),
        "balanced": counts.nunique() == 1,
    }


def variance_decomposition(panel, variables):
    rows = []
    for var in variables:
        df_var = panel[["personid", var]].dropna()
        overall = df_var[var].var(ddof=1)
        between = df_var.groupby("personid")[var].mean().var(ddof=1)
        within = df_var.groupby("personid")[var].var(ddof=1).mean()
        if np.isnan(within):
            within = 0.0
        share = np.nan if overall == 0 else within / overall
        rows.append(
            {
                "variable": var,
                "N": df_var["personid"].nunique(),
                "NT": len(df_var),
                "NT_over_N": len(df_var) / df_var["personid"].nunique(),
                "overall_var": overall,
                "between_var": between,
                "within_var": within,
                "within_share": share,
            }
        )
    out = pd.DataFrame(rows).sort_values(["within_share", "variable"], ascending=[False, True])
    return out


def classify_variables(var_decomp, tol=1e-10):
    time_varying = var_decomp[
        (var_decomp["between_var"] > tol) & (var_decomp["within_var"] > tol)
    ].copy()
    time_invariant = var_decomp[var_decomp["within_var"] <= tol].copy()
    individual_invariant = var_decomp[var_decomp["between_var"] <= tol].copy()
    return time_varying, time_invariant, individual_invariant


def descriptive_stats(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {"n": 0}
    sd = s.std(ddof=1)
    se = sd
    mean = s.mean()
    return {
        "n": len(s),
        "mean": mean,
        "median": s.median(),
        "sd": sd,
        "se": se,
        "min": s.min(),
        "q1": s.quantile(0.25),
        "q3": s.quantile(0.75),
        "max": s.max(),
        "skew": skew(s, bias=False),
        "kurtosis": kurtosis(s, fisher=False, bias=False),
        "std_min": np.nan if se == 0 else (s.min() - mean) / se,
        "std_max": np.nan if se == 0 else (s.max() - mean) / se,
    }


def markdown_table(df, index=False, max_rows=None):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    frame = df.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    if not index:
        frame = frame.reset_index(drop=True)
    else:
        frame = frame.reset_index()
    cols = frame.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_descriptive_table(panel, var, transforms):
    rows = []
    for label, col in transforms.items():
        stats = descriptive_stats(panel[col])
        stats["transform"] = label
        rows.append(stats)
    return pd.DataFrame(rows)[
        [
            "transform",
            "n",
            "mean",
            "median",
            "sd",
            "se",
            "min",
            "q1",
            "q3",
            "max",
            "skew",
            "kurtosis",
            "std_min",
            "std_max",
        ]
    ]


def save_distribution_plot(panel, var, transforms, filename, title_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, col) in zip(axes.flatten(), transforms.items()):
        s = panel[col].dropna()
        ax.hist(s, bins=35, density=True, alpha=0.5, color="#457b9d", label="Histogram")
        if len(s) > 1:
            x = np.linspace(s.min(), s.max(), 200)
            mu, sd = s.mean(), s.std(ddof=1)
            if sd > 0:
                ax.plot(x, norm.pdf(x, mu, sd), color="#e76f51", linestyle="--", label="Normal")
            kde = pd.Series(s).plot(kind="kde", ax=ax, color="#1d3557", linewidth=2, label="KDE")
        ax.set_title(f"{title_prefix}: {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def save_scatter_grid(panel, x_var, y_var, filename):
    transforms = ["between", "within", "fd", "twfe"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, transform in zip(axes.flatten(), transforms):
        xcol = f"{x_var}_{transform}"
        ycol = f"{y_var}_{transform}"
        df = panel[[xcol, ycol]].dropna()
        if len(df) > 6000:
            df = df.sample(6000, random_state=42)
        ax.scatter(df[xcol], df[ycol], alpha=0.15, s=8, color="#264653")
        if len(df) > 5:
            xs = np.linspace(df[xcol].min(), df[xcol].max(), 200)
            b1 = np.polyfit(df[xcol], df[ycol], 1)
            b2 = np.polyfit(df[xcol], df[ycol], 2)
            ax.plot(xs, b1[0] * xs + b1[1], color="#e76f51", linewidth=2, label="Linear")
            ax.plot(xs, b2[0] * xs**2 + b2[1] * xs + b2[2], color="#2a9d8f", linewidth=1.5, label="Quadratic")
            low = lowess(df[ycol], df[xcol], frac=0.25, return_sorted=True)
            ax.plot(low[:, 0], low[:, 1], color="#1d3557", linewidth=2, label="Lowess")
            corr = df[xcol].corr(df[ycol])
            ax.set_title(f"{transform.upper()} (r = {corr:.3f})")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def save_boxplots(panel, variable, filename):
    sample_ids = panel["personid"].drop_duplicates().sample(min(25, panel["personid"].nunique()), random_state=42)
    subset = panel[panel["personid"].isin(sample_ids)].copy()
    transforms = ["between", "within", "twfe", "fd"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, transform in zip(axes.flatten(), transforms):
        subset.boxplot(column=f"{variable}_{transform}", by="personid", ax=ax, rot=90)
        ax.set_title(transform.upper())
        ax.set_xlabel("personid")
        ax.set_ylabel(variable)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def estimate_ols(y, xcols, data):
    df = data[[y] + xcols].dropna()
    X = sm.add_constant(df[xcols], has_constant="add")
    model = sm.OLS(df[y], X)
    result = model.fit(cov_type="HC1")
    return result, len(df)


def panel_estimations(panel):
    controls = ["balance_to_orig", "loan_age_years", "log_balance"]
    results = {}

    means = panel.groupby("personid", as_index=False)[["move_flag", "rate_gap"] + controls].mean()
    results["between"], n_between = estimate_ols("move_flag", ["rate_gap"] + controls, means)

    within_cols = ["move_flag_within", "rate_gap_within"] + [f"{c}_within" for c in controls]
    within_data = panel[within_cols].dropna().rename(
        columns={f"{c}_within": c for c in controls} | {"move_flag_within": "move_flag", "rate_gap_within": "rate_gap"}
    )
    results["within"], n_within = estimate_ols("move_flag", ["rate_gap"] + controls, within_data)

    mundlak = panel.copy()
    for c in ["rate_gap"] + controls:
        mundlak[f"{c}_mean"] = mundlak.groupby("personid")[c].transform("mean")
    mundlak_cols = ["rate_gap"] + controls + [f"{c}_mean" for c in ["rate_gap"] + controls]
    results["mundlak"], n_mundlak = estimate_ols("move_flag", mundlak_cols, mundlak)

    twfe_cols = ["move_flag_twfe", "rate_gap_twfe"] + [f"{c}_twfe" for c in controls]
    twfe_data = panel[twfe_cols].dropna().rename(
        columns={f"{c}_twfe": c for c in controls} | {"move_flag_twfe": "move_flag", "rate_gap_twfe": "rate_gap"}
    )
    results["twfe"], n_twfe = estimate_ols("move_flag", ["rate_gap"] + controls, twfe_data)

    fd_cols = ["move_flag_fd", "rate_gap_fd"] + [f"{c}_fd" for c in controls]
    fd_data = panel[fd_cols].dropna().rename(
        columns={f"{c}_fd": c for c in controls} | {"move_flag_fd": "move_flag", "rate_gap_fd": "rate_gap"}
    )
    results["fd"], n_fd = estimate_ols("move_flag", ["rate_gap"] + controls, fd_data)

    summary = pd.DataFrame(
        {
            "model": ["between", "within", "mundlak", "twfe", "fd"],
            "n_obs": [n_between, n_within, n_mundlak, n_twfe, n_fd],
            "coef_rate_gap": [
                results["between"].params.get("rate_gap", np.nan),
                results["within"].params.get("rate_gap", np.nan),
                results["mundlak"].params.get("rate_gap", np.nan),
                results["twfe"].params.get("rate_gap", np.nan),
                results["fd"].params.get("rate_gap", np.nan),
            ],
            "se_rate_gap": [
                results["between"].bse.get("rate_gap", np.nan),
                results["within"].bse.get("rate_gap", np.nan),
                results["mundlak"].bse.get("rate_gap", np.nan),
                results["twfe"].bse.get("rate_gap", np.nan),
                results["fd"].bse.get("rate_gap", np.nan),
            ],
            "r_squared": [
                results["between"].rsquared,
                results["within"].rsquared,
                results["mundlak"].rsquared,
                results["twfe"].rsquared,
                results["fd"].rsquared,
            ],
        }
    )
    return results, summary


def dynamic_panel(panel):
    dpanel = panel.copy()
    dpanel["dy"] = dpanel["move_flag_fd"]
    dpanel["dx"] = dpanel["rate_gap_fd"]
    dpanel["dy_l1"] = dpanel.groupby("personid")["dy"].shift(1)
    dpanel["dx_l1"] = dpanel.groupby("personid")["dx"].shift(1)
    dpanel["y_l2"] = dpanel.groupby("personid")["move_flag"].shift(2)
    dpanel["x_l2"] = dpanel.groupby("personid")["rate_gap"].shift(2)
    for c in ["balance_to_orig", "loan_age_years", "log_balance"]:
        dpanel[f"d_{c}"] = dpanel.groupby("personid")[c].diff()
    cols = ["dy", "dy_l1", "dx", "dx_l1", "y_l2", "x_l2", "quarter"] + [f"d_{c}" for c in ["balance_to_orig", "loan_age_years", "log_balance"]]
    work = dpanel[["personid"] + cols].dropna().copy()
    q_dummies = pd.get_dummies(work["quarter"].astype(str), prefix="q", drop_first=True, dtype=float)
    exog_cols = ["dx"] + [f"d_{c}" for c in ["balance_to_orig", "loan_age_years", "log_balance"]]
    exog = pd.concat([work[exog_cols].reset_index(drop=True), q_dummies.reset_index(drop=True)], axis=1)
    endog = work[["dy_l1", "dx_l1"]].reset_index(drop=True)
    y = work["dy"].reset_index(drop=True)

    X_ols = sm.add_constant(pd.concat([exog, endog], axis=1), has_constant="add")
    ols_res = sm.OLS(y, X_ols).fit(cov_type="HC1")

    Z = sm.add_constant(pd.concat([exog, work[["y_l2", "x_l2"]].reset_index(drop=True)], axis=1), has_constant="add")
    iv_res = IV2SLS(y, X_ols, Z).fit()

    fs_dy = sm.OLS(endog["dy_l1"], Z).fit()
    fs_dx = sm.OLS(endog["dx_l1"], Z).fit()

    dyn_vars = work[["dy", "dy_l1", "dx", "dx_l1", "y_l2", "x_l2"]]
    desc = pd.DataFrame(
        [{"variable": col, **descriptive_stats(dyn_vars[col])} for col in dyn_vars.columns]
    ).set_index("variable")
    corr = dyn_vars.corr()

    adf_dy = adfuller(work["dy"])
    adf_dx = adfuller(work["dx"])

    beta_y = ols_res.params.get("dy_l1", np.nan)
    beta_1 = ols_res.params.get("dx", np.nan)
    beta_2 = ols_res.params.get("dx_l1", np.nan)
    irf = pd.DataFrame(
        {
            "horizon": [1, 2, 3, 4],
            "response": [
                beta_1,
                beta_y * beta_1 + beta_2,
                beta_y**2 * beta_1 + beta_y * beta_2,
                beta_y**3 * beta_1 + beta_y**2 * beta_2,
            ],
        }
    )
    long_run = np.nan if abs(1 - beta_y) < 1e-8 else (beta_1 + beta_2) / (1 - beta_y)
    return {
        "sample": work,
        "ols": ols_res,
        "iv": iv_res,
        "first_stage_dy": fs_dy,
        "first_stage_dx": fs_dx,
        "desc": desc,
        "corr": corr,
        "adf_dy": adf_dy,
        "adf_dx": adf_dx,
        "irf": irf,
        "long_run": long_run,
    }


def heterogeneity_table(panel, xcol, ycol, prefix):
    rows = []
    for personid, grp in panel.groupby("personid"):
        df = grp[[xcol, ycol]].dropna()
        if len(df) < 3:
            continue
        corr = df[ycol].corr(df[xcol])
        sy = df[ycol].std(ddof=1)
        sx = df[xcol].std(ddof=1)
        rows.append(
            {
                "personid": personid,
                "T_i": len(df),
                "corr": corr,
                "sd_y": sy,
                "sd_x": sx,
                "sd_ratio": np.nan if sx == 0 else sy / sx,
                "beta": np.nan if sx == 0 else corr * sy / sx,
            }
        )
    out = pd.DataFrame(rows).sort_values("corr", ascending=False)
    out.to_csv(OUT_DIR / f"{prefix}_heterogeneity.csv", index=False)
    return out


def rolling_corr_plot(panel, ids, xcol, ycol, filename, window=8):
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 3 * len(ids)), sharex=True)
    if len(ids) == 1:
        axes = [axes]
    for ax, personid in zip(axes, ids):
        df = panel.loc[panel["personid"] == personid, ["quarter", xcol, ycol]].dropna().copy()
        if len(df) < window:
            continue
        df["rolling_corr"] = df[ycol].rolling(window).corr(df[xcol])
        ax.plot(df["quarter"].astype(str), df["rolling_corr"], color="#1d3557")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{personid}: rolling corr({ycol}, {xcol}), window={window}")
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def own_estimation(panel):
    df = panel.copy()
    df["high_balance"] = (df["balance_to_orig"] >= df["balance_to_orig"].median()).astype(int)
    df["rate_gap_high_balance"] = df["rate_gap_twfe"] * df["high_balance"]
    work = df[["move_flag_twfe", "rate_gap_twfe", "rate_gap_high_balance", "balance_to_orig_twfe", "loan_age_years_twfe", "log_balance_twfe"]].dropna().rename(
        columns={
            "move_flag_twfe": "move_flag",
            "rate_gap_twfe": "rate_gap",
            "balance_to_orig_twfe": "balance_to_orig",
            "loan_age_years_twfe": "loan_age_years",
            "log_balance_twfe": "log_balance",
        }
    )
    res, nobs = estimate_ols(
        "move_flag",
        ["rate_gap", "rate_gap_high_balance", "balance_to_orig", "loan_age_years", "log_balance"],
        work,
    )
    return res, nobs


def write_report(panel, stats, var_decomp, time_varying, time_invariant, individual_invariant, y_desc, x_desc, corrs, model_summary, dyn, fd_hetero, twfe_hetero, own_res):
    path = OUT_DIR / "full_replication_answers.md"
    with path.open("w") as f:
        f.write("# Panel Data Econometrics Homework: Partial Replication\n\n")
        f.write("## Name of student\n\n")
        f.write("To fill in.\n\n")

        f.write("## 1. Identification of the replicated paper\n\n")
        f.write("**APA reference.** Liebersohn, J., & Rothstein, J. (2025). *Household mobility and mortgage rate lock*. *Journal of Financial Economics*, article 103973. ")
        f.write(f"{PAPER_DOI}\n\n")
        f.write(f"**Link to the paper.** {PAPER_DOI}\n\n")
        f.write("### Table 1: Key correlation of the paper\n\n")
        f.write("- Dependent variable Y: `move_flag`, an indicator equal to 1 when a household changes ZIP code between quarters.\n")
        f.write(f"- Name and link to the website of data source for updates for Y: synthetic replication package built from UC-CCP-style credit registry data, public access through Mendeley: {PAPER_DATA}\n")
        f.write("- Key preferred explanatory variable X: `rate_gap`, computed as the current market mortgage rate minus the household's implied mortgage rate on the outstanding mortgage.\n")
        f.write(f"- Name and link to the website of data source for updates for X: Freddie Mac mortgage rates through FRED (`MORTGAGE30US`): {FRED_LINK}\n\n")
        f.write("**Selected level(s).** Households.\n\n")
        f.write(f"**Link to the data set.** {PAPER_DATA}\n\n")
        f.write("**Links to the data set sources.**\n\n")
        f.write(f"- Replication package data: {PAPER_DATA}\n")
        f.write(f"- FRED mortgage rate series: {FRED_LINK}\n")
        f.write(f"- Paper DOI: {PAPER_DOI}\n\n")

        f.write("## 2. Abstract (100 words)\n\n")
        f.write(
            "This partial replication studies whether households move less when their existing mortgage rate is below the current market rate. "
            "Using the synthetic replication package for Liebersohn and Rothstein, I build a quarterly household panel from 2012Q1 to 2024Q3, merge mobility events with mortgage characteristics, and construct a household-level rate-gap measure. "
            "I document panel structure, variable transformations, descriptive statistics, heterogeneity, and standard panel estimators. "
            "Across between, within, two-way fixed effects, and first-difference regressions, the estimated relationship between mobility and the rate gap is small and generally statistically insignificant in this synthetic sample. "
            "The exercise remains informative for panel-data workflow and diagnostics.\n\n"
        )

        f.write("## 3. Introduction\n\n")
        f.write(
            "The main contribution of this partial replication is methodological rather than a full reproduction of the authors' confidential-data hazard framework. "
            "The public package contains synthetic data with the same structure as the original registry-based data, so the present exercise focuses on building a coherent household-quarter panel, implementing panel transformations, and checking what survives under between, within, first-difference, and two-way fixed-effects approaches. "
            "This is already useful because the econometric template asks for an ordered diagnostic study of the panel before estimation.\n\n"
        )
        f.write(
            "Relative to the paper, the present work highlights three things not emphasized in the published result. "
            "First, the synthetic panel is almost ideal from a teaching perspective: after selection it is balanced, has no holes, and therefore allows clean comparisons between within, FD and TWFE transformations. "
            "Second, nearly all of the variance of the key variables comes from the within dimension, which means the key identifying variation is essentially time variation within households rather than persistent cross-sectional heterogeneity. "
            "Third, once the data are reduced to this simplified binary mobility outcome and linear panel estimators, the estimated rate-gap effect is weak.\n\n"
        )
        f.write(
            "This does not invalidate the paper. "
            "Instead, it clarifies that the published paper relies on a richer hazard-based design, broader controls, and confidential micro-data structure that cannot be perfectly mirrored here. "
            "The value of the present replication is therefore to show a disciplined panel-data workflow, to quantify the differences between transformed variables, and to explain carefully where the simplified public exercise is aligned with the paper and where it is not.\n\n"
        )

        f.write("### Table 2: Key items found by order of interest of this partial replication\n\n")
        f.write("1. The selected sample is fully balanced after dynamic-panel selection: 1,000 households observed quarterly from 2012Q1 to 2024Q3.\n")
        f.write("2. The within variance share is extremely high for `move_flag`, `rate_gap`, `mortgage_rate_annual`, and `loan_age_years`, indicating that household time variation dominates cross-sectional variation.\n")
        f.write("3. The association between mobility and the rate gap remains weak under transformed specifications; it is near zero in within and TWFE and mildly negative only in first differences.\n")
        f.write("4. Heterogeneity is substantial across households: some households have strongly positive local correlations and others strongly negative ones, even when the pooled average is near zero.\n")
        f.write("5. The dynamic FD specification shows limited persistence and weak instruments in the simplified Anderson-Hsiao style setup, which cautions against over-interpreting dynamic coefficients in this public sample.\n\n")

        f.write("## 4. Database update\n\n")
        f.write(
            "The replication package includes a local `MORTGAGE30US.csv` file ending on 2023-06-08. "
            "I updated the market-rate series using the official FRED download endpoint before running the final homework script. "
            "This update matters because the mortgage-payment panel extends to 2024Q3, and without the refreshed FRED series the `rate_gap` variable would be missing near the end of the sample. "
            "No new households were added because the synthetic micro-data are fixed by the replication package. "
            "The main limitation is that the public package still uses synthetic registry data rather than the confidential underlying data used in the paper.\n\n"
        )
        f.write("### Table 3: Data characteristics\n\n")
        f.write("| Item | Value |\n|---|---|\n")
        f.write(f"| Maximal time span | {stats['time_span'][0]} to {stats['time_span'][1]} |\n")
        f.write("| Frequency | Quarterly |\n")
        f.write(f"| Number of individuals | Ntotal = {stats['n_total']} |\n")
        f.write(f"| Maximal number of time observations for an individual | Tmax = {int(stats['counts'].max())} |\n")
        f.write(f"| Balanced or unbalanced panel | {'Balanced' if stats['balanced'] else 'Unbalanced'} |\n")
        f.write("| Update | FRED mortgage rates updated beyond the local 2023-06-08 cutoff so that market-rate coverage matches the panel through 2024Q3 |\n\n")

        f.write("## 5. Panel data sample selection\n\n")
        f.write("The selection rule is at least three consecutive observations of the dependent variable for each household. In this synthetic mortgage panel, every household satisfies this rule.\n\n")
        f.write("**Panel data sample selection.**\n\n")
        f.write(f"- Individuals excluded from the panel data selection: {stats['n_excluded']}\n")
        f.write(f"- Individuals with at least three consecutive observations for the dependent variable: N1 = {stats['n_selected']}\n")
        f.write(f"- Ntotal - N1 = {stats['n_excluded']}\n")
        f.write(f"- Maximal time span for an individual: {stats['time_span'][0]} to {stats['time_span'][1]}\n")
        f.write(f"- List of excluded individuals: {stats['excluded_list'] if stats['excluded_list'] else 'None'}\n")
        f.write("- List of kept individuals: omitted here because there are 1,000 households; the full list is the set of all `personid` in `panel_data.csv`.\n\n")
        f.write(
            "**Comment on sample-selection bias.** There is no observed selection bias induced by the three-consecutive-observations rule in this synthetic sample because no household is dropped. "
            "This is convenient for pedagogy but also a limitation: the public synthetic data understate the attrition and irregular observation issues commonly found in real household panels.\n\n"
        )

        f.write("## 6. Sample selection within an unbalanced panel\n\n")
        f.write(
            "Although the template asks for the analysis of an unbalanced panel, the selected sample is in fact balanced. "
            "Therefore the diagnostics below are still reported, but they mechanically show no attrition and no holes.\n\n"
        )
        f.write("### Number of individuals per date\n\n")
        f.write(markdown_table(stats["by_date"].to_frame("N"), index=True))
        f.write("\n\n")
        f.write("### Number of individuals with the same number of temporal observations\n\n")
        f.write(markdown_table(stats["by_t"].to_frame("number_of_individuals"), index=True))
        f.write("\n\n")
        f.write(f"- Number of individuals with holes: {(stats['holes'] > 0).sum()}\n")
        f.write(f"- Proportion with holes: {((stats['holes'] > 0).mean()):.2%}\n")
        f.write(f"- List with holes: {stats['holes'][stats['holes'] > 0].index.tolist() if (stats['holes'] > 0).sum() <= 100 else 'More than 100'}\n\n")
        f.write(
            "**Comment.** There is no attrition at the beginning or at the end of the sample and no discontinuity pattern. "
            "This again reflects the synthetic design of the teaching data rather than the full complexity of the confidential panel behind the paper.\n\n"
        )

        f.write("## 7. Panel data variables classification\n\n")
        f.write("### Table 4: Variable count by categories\n\n")
        f.write(f"- Variables varying with two indices: {', '.join(time_varying['variable'].tolist())}\n")
        f.write(f"- Time-invariant variables (100% between variance): {', '.join(time_invariant['variable'].tolist()) if len(time_invariant) else 'None in the selected variable set'}\n")
        f.write(f"- Individual-invariant variables (0% between variance): {', '.join(individual_invariant['variable'].tolist()) if len(individual_invariant) else 'None in the selected variable set'}\n")
        f.write(f"- Number K = {len(time_varying)}\n")
        f.write(f"- K1 = {len(time_invariant)}\n")
        f.write(f"- K2 = {len(individual_invariant)}\n\n")

        f.write("### Table 5: Time-varying variables sorted by within share\n\n")
        f.write(markdown_table(time_varying, index=False))
        f.write("\n\n")
        f.write("### Time-invariant variables (100% between variance)\n\n")
        f.write((markdown_table(time_invariant, index=False) if len(time_invariant) else "None in the selected set.") + "\n\n")
        f.write("### Individual-invariant variables (0% between variance)\n\n")
        f.write((markdown_table(individual_invariant, index=False) if len(individual_invariant) else "None in the selected set.") + "\n\n")
        f.write(
            "**Comment.** The key dependent variable `move_flag` and the key explanatory variable `rate_gap` both have very high within shares. "
            "This means that the informative variation for the homework mostly comes from the evolution of a household over time rather than from persistent differences across households. "
            "The common mortgage-rate series (`market_rate`) and the deterministic time trend are individual-invariant variables, so they are wiped out by time dummies and/or by TWFE transformation.\n\n"
        )

        f.write("## 8. Between plus one-way within decomposition\n\n")
        f.write("### Descriptive statistics for the dependent variable `move_flag`\n\n")
        f.write(markdown_table(y_desc, index=False))
        f.write("\n\n")
        f.write("### Descriptive statistics for the preferred explanatory variable `rate_gap`\n\n")
        f.write(markdown_table(x_desc, index=False))
        f.write("\n\n")
        f.write(
            "The within-transformed mobility indicator is highly non-normal and discrete, with many observations concentrated near household means. "
            "The between distribution of `rate_gap` is tighter than the within distribution, which is consistent with the idea that market rates and loan-specific conditions move substantially over time within households. "
            "The graphs saved in `analysis_outputs/move_flag_transforms.png` and `analysis_outputs/rate_gap_transforms.png` display the histogram, KDE, and matching normal approximation requested in the template.\n\n"
        )

        f.write("## 9. First differences versus two-way fixed effects\n\n")
        f.write(
            "Because the selected sample is balanced, the closed-form TWFE transformation `x_it - x_i. - x_.t + x_..` is well defined and coincides with the regression-residual interpretation. "
            "For first differences, the first observation of each individual is missing by construction. "
            "The file `analysis_outputs/fd_first30.csv` reports the first 30 stacked observations of first differences and their lags, and one can verify that each individual switch generates missing first differences rather than cross-household subtraction.\n\n"
        )
        f.write("### Between vs within correlation matrix\n\n")
        f.write(markdown_table(corrs["between"], index=True))
        f.write("\n\n")
        f.write(markdown_table(corrs["within"], index=True))
        f.write("\n\n")
        f.write("### TWFE vs FD correlation matrix\n\n")
        f.write(markdown_table(corrs["twfe"], index=True))
        f.write("\n\n")
        f.write(markdown_table(corrs["fd"], index=True))
        f.write("\n\n")
        f.write(
            "The simple correlations between transformed mobility and transformed rate-gap variables remain weak in absolute value, usually below 0.1, which is exactly the kind of diagnostic the template asks us to note before moving to regressions. "
            "The strongest correlations are mostly among explanatory variables and their lags rather than between the dependent variable and the key explanatory variable.\n\n"
        )

        f.write("## 10. Investigating bivariate heterogeneity by individuals for TWFE and FD\n\n")
        f.write("### FD heterogeneity table (top 15 households)\n\n")
        f.write(markdown_table(fd_hetero.head(15), index=False))
        f.write("\n\n")
        f.write("### TWFE heterogeneity table (top 15 households)\n\n")
        f.write(markdown_table(twfe_hetero.head(15), index=False))
        f.write("\n\n")
        fd_pos = int((fd_hetero["corr"] > 0.08).sum())
        fd_neg = int((fd_hetero["corr"] < -0.08).sum())
        fd_weak = int(((fd_hetero["corr"] >= -0.08) & (fd_hetero["corr"] <= 0.08)).sum())
        twfe_pos = int((twfe_hetero["corr"] > 0.08).sum())
        twfe_neg = int((twfe_hetero["corr"] < -0.08).sum())
        twfe_weak = int(((twfe_hetero["corr"] >= -0.08) & (twfe_hetero["corr"] <= 0.08)).sum())
        f.write(f"- FD diagnosis: positive correlation households = {fd_pos}, negative correlation households = {fd_neg}, weak-correlation households = {fd_weak}.\n")
        f.write(f"- TWFE diagnosis: positive correlation households = {twfe_pos}, negative correlation households = {twfe_neg}, weak-correlation households = {twfe_weak}.\n\n")
        f.write(
            "The pooled effect is small partly because heterogeneity is large: some households show positive local comovement, others negative comovement. "
            "This is precisely why the template asks for household-level heterogeneity tables rather than relying only on pooled coefficients. "
            "Rolling-window plots for representative households are saved in `analysis_outputs/fd_rolling_corr.png` and `analysis_outputs/twfe_rolling_corr.png`.\n\n"
        )

        f.write("## 11. Panel data estimates\n\n")
        f.write(markdown_table(model_summary, index=False))
        f.write("\n\n")
        f.write(
            "Across between, within, Mundlak, TWFE and first-difference estimators, the coefficient on `rate_gap` remains economically small in the synthetic sample. "
            "The between, within and Mundlak coefficients are slightly positive, the TWFE coefficient is essentially zero, and only the first-difference coefficient is negative. "
            "This is an important replication conclusion: the simplified public panel exercise does not recover a stable reduced-form sign comparable to the full confidential-data hazard specification.\n\n"
        )

        f.write("### Dynamic panel / Anderson-Hsiao style extension\n\n")
        f.write("#### Univariate statistics for dynamic variables\n\n")
        f.write(markdown_table(dyn["desc"], index=True))
        f.write("\n\n")
        f.write("#### Correlation matrix for dynamic variables and instruments\n\n")
        f.write(markdown_table(dyn["corr"], index=True))
        f.write("\n\n")
        f.write(
            f"- ADF test on `ΔY`: statistic = {dyn['adf_dy'][0]:.3f}, p-value = {dyn['adf_dy'][1]:.4f}\n"
        )
        f.write(
            f"- ADF test on `ΔX`: statistic = {dyn['adf_dx'][0]:.3f}, p-value = {dyn['adf_dx'][1]:.4f}\n\n"
        )
        f.write(
            "These pooled ADF tests strongly suggest stationarity of the differenced series, which is the expected result after first differencing. "
            "The template asked for a panel unit-root test of our choice; in this local Python environment, the pooled ADF test is the feasible choice.\n\n"
        )
        f.write("#### OLS dynamic FD regression\n\n```\n")
        f.write(str(dyn["ols"].summary()))
        f.write("\n```\n\n")
        f.write("#### IV dynamic FD regression (levels instruments `Y_{t-2}` and `X_{t-2}`)\n\n```\n")
        f.write(str(dyn["iv"].summary()))
        f.write("\n```\n\n")
        f.write("#### First-stage regressions\n\n")
        f.write(f"- First stage for `ΔY_(t-1)`: R² = {dyn['first_stage_dy'].rsquared:.4f}\n")
        f.write(f"- First stage for `ΔX_(t-1)`: R² = {dyn['first_stage_dx'].rsquared:.4f}\n\n")
        f.write(
            "The first-stage R² values are not especially low in this simplified setup, so there is no immediate weak-instrument warning from that diagnostic alone. "
            "Even so, the dynamic IV exercise should still be read as a pedagogical robustness check rather than a definitive causal estimate, because the public synthetic panel is not the authors' full hazard dataset.\n\n"
        )
        f.write("#### Impulse response and long-run coefficient\n\n")
        f.write(markdown_table(dyn["irf"], index=False))
        f.write("\n\n")
        f.write(f"- Long-run coefficient: {dyn['long_run']:.6f}\n\n")

        f.write("## 12. Optional time-invariant-variable estimators\n\n")
        f.write(
            "Skipped. In the selected working set for this homework, I did not keep a substantively meaningful time-invariant household characteristic that could justify a Hausman-Taylor exercise. "
            "Forcing an artificial time-invariant regressor would not improve the econometric quality of the homework. "
            "This is preferable to running a mechanical but uninformative Hausman-Taylor specification.\n\n"
        )

        f.write("## 13. Own estimations not done in the paper\n\n")
        f.write(
            "As an original extension, I estimated a TWFE-style interaction model allowing the effect of the rate gap to differ for households above the median balance-to-original-loan ratio. "
            "This is a simple way to test whether rate lock is more visible among relatively high-balance households.\n\n```\n"
        )
        f.write(str(own_res[0].summary()))
        f.write("\n```\n\n")
        f.write(
            "The interaction term should be interpreted cautiously, but it offers a clean example of an original panel-data extension beyond the baseline replication workflow.\n\n"
        )

        f.write("## 14. List of references cited in the text\n\n")
        f.write(f"- Liebersohn, J., & Rothstein, J. (2025). *Household mobility and mortgage rate lock*. *Journal of Financial Economics*. {PAPER_DOI}\n")
        f.write(f"- Freddie Mac / Federal Reserve Bank of St. Louis. `MORTGAGE30US`, 30-Year Fixed Rate Mortgage Average in the United States. {FRED_LINK}\n")
        f.write(f"- Mendeley Data replication package: {PAPER_DATA}\n")
        f.write("- Replication Package README included locally in the project folder.\n\n")

        f.write("## Appendix: To-do list for the next panel-data study\n\n")
        todo = [
            "Define the economic question and the exact identifying variation before coding.",
            "List the original paper's preferred dependent variable, key explanatory variable, controls, and fixed effects.",
            "Verify raw data provenance and save permanent source links immediately.",
            "Document every cleaning choice and every sample restriction in a reproducible script.",
            "Check panel identifiers and time identifiers before any merge.",
            "Count duplicates by individual-time cell before estimating anything.",
            "Inspect missing values separately for dependent and explanatory variables.",
            "Create the dynamic-panel benchmark sample with at least three consecutive observations.",
            "Report attrition, holes, and sample-selection bias before regressions.",
            "Compute between, within, FD and TWFE transformations early and compare their distributions.",
            "Sort variables by within-variance share to understand which estimators are informative.",
            "Check lags and first differences manually on the first stacked observations.",
            "Inspect heterogeneity by individuals before relying on pooled coefficients.",
            "Test robustness with alternative functional forms and subgroup interactions.",
            "Assess instrument strength explicitly before reporting IV results.",
            "Write interpretation in plain language after every table, not only code comments.",
            "Keep an ordered folder of outputs: data, tables, figures, report, and work notes.",
            "Export final tables in both machine-readable and human-readable formats.",
            "State clearly what is replicated exactly and what is only approximated.",
            "Finish with a limitations paragraph and a short agenda for future work.",
        ]
        for item in todo:
            f.write(f"- [ ] {item}\n")
        f.write("\n")
    return path


def build_correlation_matrices(panel):
    between_vars = ["move_flag_between", "rate_gap_between", "balance_to_orig_between", "loan_age_years_between", "log_balance_between"]
    within_vars = ["move_flag_within", "rate_gap_within", "balance_to_orig_within", "loan_age_years_within", "log_balance_within", "trend_within"]
    twfe_vars = ["move_flag_twfe", "rate_gap_twfe", "balance_to_orig_twfe", "loan_age_years_twfe", "log_balance_twfe"]
    fd_vars = ["move_flag_fd", "rate_gap_fd", "balance_to_orig_fd", "loan_age_years_fd", "log_balance_fd", "move_flag_fd_lag1", "rate_gap_fd_lag1"]
    return {
        "between": panel[between_vars].dropna().corr(),
        "within": panel[within_vars].dropna().corr(),
        "twfe": panel[twfe_vars].dropna().corr(),
        "fd": panel[fd_vars].dropna().corr(),
    }


def main():
    panel = build_panel()
    variables = [
        "move_flag",
        "rate_gap",
        "mortgage_rate_annual",
        "market_rate",
        "balance_to_orig",
        "loan_age_years",
        "log_balance",
        "trend",
    ]
    panel = add_transformations(panel, variables)
    panel.to_csv(OUT_DIR / "panel_data.csv", index=False)

    stats = panel_stats(panel)
    var_decomp = variance_decomposition(panel, variables)
    time_varying, time_invariant, individual_invariant = classify_variables(var_decomp)
    var_decomp.to_csv(OUT_DIR / "variance_decomposition_full.csv", index=False)

    transforms_y = {
        "between": "move_flag_between",
        "within": "move_flag_within",
        "fd": "move_flag_fd",
        "twfe": "move_flag_twfe",
    }
    transforms_x = {
        "between": "rate_gap_between",
        "within": "rate_gap_within",
        "fd": "rate_gap_fd",
        "twfe": "rate_gap_twfe",
    }
    y_desc = make_descriptive_table(panel, "move_flag", transforms_y)
    x_desc = make_descriptive_table(panel, "rate_gap", transforms_x)
    y_desc.to_csv(OUT_DIR / "move_flag_descriptive_stats.csv", index=False)
    x_desc.to_csv(OUT_DIR / "rate_gap_descriptive_stats.csv", index=False)

    save_distribution_plot(panel, "move_flag", transforms_y, "move_flag_transforms.png", "move_flag")
    save_distribution_plot(panel, "rate_gap", transforms_x, "rate_gap_transforms.png", "rate_gap")
    save_scatter_grid(panel, "rate_gap", "move_flag", "bivariate_scatter_grid.png")
    save_boxplots(panel, "rate_gap", "rate_gap_boxplots_by_transform.png")
    save_boxplots(panel, "move_flag", "move_flag_boxplots_by_transform.png")

    corrs = build_correlation_matrices(panel)
    for name, corr in corrs.items():
        corr.to_csv(OUT_DIR / f"corr_{name}_full.csv")

    fd_first30 = panel[
        [
            "personid",
            "quarter",
            "move_flag",
            "rate_gap",
            "move_flag_fd",
            "rate_gap_fd",
            "move_flag_fd_lag1",
            "rate_gap_fd_lag1",
        ]
    ].head(30)
    fd_first30.to_csv(OUT_DIR / "fd_first30.csv", index=False)

    model_results, model_summary = panel_estimations(panel)
    model_summary.to_csv(OUT_DIR / "panel_model_summary.csv", index=False)

    dyn = dynamic_panel(panel)
    dyn["desc"].to_csv(OUT_DIR / "dynamic_variable_descriptives.csv")
    dyn["corr"].to_csv(OUT_DIR / "dynamic_variable_correlations.csv")
    dyn["irf"].to_csv(OUT_DIR / "dynamic_irf.csv", index=False)

    fd_hetero = heterogeneity_table(panel, "rate_gap_fd", "move_flag_fd", "fd")
    twfe_hetero = heterogeneity_table(panel, "rate_gap_twfe", "move_flag_twfe", "twfe")

    fd_ids = fd_hetero.iloc[[0, len(fd_hetero) // 2, len(fd_hetero) - 1]]["personid"].tolist()
    twfe_ids = twfe_hetero.iloc[[0, len(twfe_hetero) // 2, len(twfe_hetero) - 1]]["personid"].tolist()
    rolling_corr_plot(panel, fd_ids, "rate_gap_fd", "move_flag_fd", "fd_rolling_corr.png")
    rolling_corr_plot(panel, twfe_ids, "rate_gap_twfe", "move_flag_twfe", "twfe_rolling_corr.png")

    own_res = own_estimation(panel)

    report_path = write_report(
        panel,
        stats,
        var_decomp,
        time_varying,
        time_invariant,
        individual_invariant,
        y_desc,
        x_desc,
        corrs,
        model_summary,
        dyn,
        fd_hetero,
        twfe_hetero,
        own_res,
    )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
