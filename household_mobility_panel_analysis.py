import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(".").resolve()
DATA_ROOT = ROOT / "Data for Household Mobility and Mortgage Rate Lock" / "Replication Package"
SYNTH_DIR = DATA_ROOT / "raw" / "synthetic data"
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)


def parse_quarter_from_int(date_int):
    date_str = str(int(date_int))
    return pd.to_datetime(date_str, format="%Y%m%d").to_period("Q")


def implied_mortgage_rate(payment, balance, remaining_periods):
    if payment <= 0 or balance <= 0 or remaining_periods <= 0:
        return np.nan

    def f(r):
        if r == 0:
            return balance / remaining_periods - payment
        return payment - r * balance / (1 - (1 + r) ** (-remaining_periods))

    try:
        return brentq(f, 1e-6, 1.0, maxiter=200)
    except ValueError:
        return np.nan


def load_mortgage_panel():
    path = SYNTH_DIR / "mortgage_payments_data.csv"
    df = pd.read_csv(path)
    df = df.copy()
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
    return df


def load_market_rates():
    path = DATA_ROOT / "raw" / "MORTGAGE30US.csv"
    rates = pd.read_csv(path)
    rates["DATE"] = pd.to_datetime(rates["DATE"], errors="coerce")
    rates = rates.dropna(subset=["DATE"]).copy()
    rates["quarter"] = rates["DATE"].dt.to_period("Q")
    q_rates = rates.groupby("quarter", as_index=False)["MORTGAGE30US"].mean().rename(
        columns={"MORTGAGE30US": "market_rate"}
    )
    return q_rates


def load_moves():
    path = SYNTH_DIR / "national_get_all_moves.csv"
    moves = pd.read_csv(path, dtype={"zip1": str, "zip2": str})
    moves["quarter"] = pd.to_datetime(
        moves["file_year"].astype(str) + moves["file_month"].astype(str).str.zfill(2) + "01",
        format="%Y%m%d",
    ).dt.to_period("Q")
    moves["move_flag"] = (moves["zip1"] != moves["zip2"]).astype(int)
    moves = moves.groupby(["personid", "quarter"], as_index=False)["move_flag"].max()
    return moves


def attach_variables(panel):
    rates = load_market_rates()
    panel = panel.merge(rates, on="quarter", how="left")
    panel["rate_gap"] = panel["market_rate"] - panel["mortgage_rate_annual"]
    panel["rate_gap"] = panel["rate_gap"].round(4)
    panel["loan_age_years"] = panel["loan_age_months"] / 12.0
    panel["balance_to_orig"] = panel["acct_balance_am"] / panel["amount_1"].replace(0, np.nan)
    return panel


def build_panel_data():
    panel = load_mortgage_panel()
    moves = load_moves()
    panel = panel.merge(moves, on=["personid", "quarter"], how="left")
    panel["move_flag"] = panel["move_flag"].fillna(0).astype(int)
    # Ajout des variables de taux de marché et rate_gap avant toute transformation
    panel = attach_variables(panel)
    panel = panel.sort_values(["personid", "quarter"]).reset_index(drop=True)
    panel["quarter_num"] = panel["quarter"].apply(lambda q: q.year * 4 + q.quarter)
    panel["lag_quarter_num"] = panel.groupby("personid")["quarter_num"].shift(1)
    panel["consecutive"] = panel["quarter_num"] - panel["lag_quarter_num"]
    panel["new_sequence"] = panel["consecutive"] != 1
    panel["sequence_id"] = panel.groupby("personid")["new_sequence"].cumsum()
    panel["sequence_length"] = panel.groupby(["personid", "sequence_id"])["quarter"].transform("count")
    panel["consecutive_run"] = panel["sequence_length"]
    panel = panel[panel["consecutive_run"] >= 3].copy()
    panel["obs_count"] = panel.groupby("personid")["quarter"].transform("count")
    panel["person_mean_rate_gap"] = panel.groupby("personid")["rate_gap"].transform("mean")
    panel["rate_gap_within"] = panel["rate_gap"] - panel["person_mean_rate_gap"]
    panel["move_flag_within"] = panel["move_flag"] - panel.groupby("personid")["move_flag"].transform("mean")
    panel["rate_gap_fd"] = panel.groupby("personid")["rate_gap"].diff()
    panel["move_flag_fd"] = panel.groupby("personid")["move_flag"].diff()
    panel["rate_gap_between"] = panel["person_mean_rate_gap"]
    panel["quarter_mean_rate_gap"] = panel.groupby("quarter")["rate_gap"].transform("mean")
    panel["rate_gap_twfe"] = panel["rate_gap"] - panel["person_mean_rate_gap"] - panel["quarter_mean_rate_gap"] + panel["rate_gap"].mean()
    panel["move_flag_twfe"] = panel["move_flag"] - panel.groupby("personid")["move_flag"].transform("mean") - panel.groupby("quarter")["move_flag"].transform("mean") + panel["move_flag"].mean()
    return panel


def panel_statistics(panel):
    print("=== Panel summary ===")
    n_persons = panel["personid"].nunique()
    n_periods = panel["quarter"].nunique()
    print(f"persons: {n_persons:,}")
    print(f"periods: {n_periods:,}")
    counts = panel.groupby("personid")["quarter"].nunique()
    print(f"mean obs per person: {counts.mean():.2f}")
    print(f"median obs per person: {counts.median():.0f}")
    print(f"min obs per person: {counts.min()}")
    print(f"max obs per person: {counts.max()}")
    balanced = counts.nunique() == 1
    print(f"balanced panel: {balanced}")
    holes = panel.groupby("personid").apply(
        lambda x: (x["quarter_num"].max() - x["quarter_num"].min() + 1) - len(x)
    )
    print(f"mean hole count per person: {holes.mean():.2f}")
    print(f"percent with hole(s): {(holes > 0).mean():.2%}")
    print("\nObservations by quarter:")
    period_counts = panel.groupby("quarter")["personid"].nunique().sort_index()
    print(period_counts.to_string())
    return {
        "persons": n_persons,
        "periods": n_periods,
        "obs_per_person_mean": counts.mean(),
        "obs_per_person_median": counts.median(),
        "obs_per_person_min": counts.min(),
        "obs_per_person_max": counts.max(),
        "balanced": balanced,
        "mean_holes": holes.mean(),
        "percent_with_holes": (holes > 0).mean(),
        "period_counts": period_counts,
    }


def variance_decomposition(panel, variables):
    results = []
    for var in variables:
        df_var = panel[["personid", var]].dropna()
        overall = df_var[var].var(ddof=1)
        between = df_var.groupby("personid")[var].mean().var(ddof=1)
        within = df_var.groupby("personid")[var].var(ddof=1).mean()
        results.append(
            {
                "variable": var,
                "overall_var": overall,
                "between_var": between,
                "within_var": within,
                "within_share": within / overall if overall else np.nan,
            }
        )
    result_df = pd.DataFrame(results).sort_values("within_share", ascending=False)
    return result_df


def estimate_ols(Y, X, data, add_const=True):
    X_mat = data[X].copy()
    if add_const:
        X_mat = sm.add_constant(X_mat)
    model = sm.OLS(data[Y], X_mat, missing="drop")
    return model.fit(cov_type="HC1")


def run_regressions(panel):
    outputs = {}
    print("\n=== Regression models ===")
    panel = panel.copy()
    # Between model on person means
    # Only use columns that exist
    cols = ["move_flag", "rate_gap", "mortgage_rate_annual"]
    agg_dict = {c: (c, "mean") for c in cols if c in panel.columns}
    person_means = panel.groupby("personid").agg(**agg_dict).reset_index()
    X_between = [c for c in ["rate_gap", "mortgage_rate_annual"] if c in person_means.columns]
    outputs["between"] = estimate_ols(
        "move_flag",
        X_between,
        person_means,
    )
    print("Between model:")
    print(outputs["between"].summary())

    # Within / Fixed effects via within transformation
    outputs["within"] = estimate_ols(
        "move_flag_within",
        ["rate_gap_within"],
        panel,
    )
    print("\nWithin model (individual FE):")
    print(outputs["within"].summary())

    # Mundlak / random effects style
    outputs["mundlak"] = estimate_ols(
        "move_flag",
        ["rate_gap", "rate_gap_between"],
        panel,
    )
    print("\nMundlak-style model:")
    print(outputs["mundlak"].summary())

    # Two-way fixed effects via double demeaning
    outputs["twfe"] = estimate_ols(
        "move_flag_twfe",
        ["rate_gap_twfe"],
        panel,
    )
    print("\nTwo-way fixed effects model (TWFE transformed variables):")
    print(outputs["twfe"].summary())

    # First differences
    outputs["fd"] = estimate_ols(
        "move_flag_fd",
        ["rate_gap_fd"],
        panel,
    )
    print("\nFirst differences model:")
    print(outputs["fd"].summary())

    return outputs


def plot_distributions(panel):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    variables = ["move_flag", "rate_gap", "mortgage_rate_annual", "balance_to_orig"]
    for ax, var in zip(axes.flatten(), variables):
        values = panel[var].dropna()
        ax.hist(values, bins=30, alpha=0.6, color="#1976d2", density=True)
        xmin, xmax = values.min(), values.max()
        x = np.linspace(xmin, xmax, 200)
        kde = values.plot(kind="kde", ax=ax, label="KDE", color="#d62728")
        if var != "move_flag":
            mu, sigma = values.mean(), values.std()
            ax.plot(x, norm.pdf(x, mu, sigma), color="#2ca02c", linestyle="--", label="Normal approx")
        ax.set_title(var)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distribution_plots.png", dpi=150)
    plt.close(fig)


def plot_scatter(panel):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter 1 : Move Flag vs Rate Gap
    df1 = panel[["rate_gap", "move_flag"]].dropna()
    axes[0].scatter(df1["rate_gap"], df1["move_flag"], alpha=0.2, s=10)
    axes[0].set_title("Move Flag vs Rate Gap")
    axes[0].set_xlabel("Rate Gap")
    axes[0].set_ylabel("Move Flag")
    if len(df1) > 1:
        coeff = np.polyfit(df1["rate_gap"], df1["move_flag"], 1)
        x = np.linspace(df1["rate_gap"].min(), df1["rate_gap"].max(), 100)
        axes[0].plot(x, coeff[0] * x + coeff[1], color="red")

    # Scatter 2 : Move Flag vs Mortgage Rate
    df2 = panel[["mortgage_rate_annual", "move_flag"]].dropna()
    axes[1].scatter(df2["mortgage_rate_annual"], df2["move_flag"], alpha=0.2, s=10)
    axes[1].set_title("Move Flag vs Mortgage Rate")
    axes[1].set_xlabel("Mortgage Rate Annual")
    axes[1].set_ylabel("Move Flag")
    if len(df2) > 1:
        coeff2 = np.polyfit(df2["mortgage_rate_annual"], df2["move_flag"], 1)
        x2 = np.linspace(df2["mortgage_rate_annual"].min(), df2["mortgage_rate_annual"].max(), 100)
        axes[1].plot(x2, coeff2[0] * x2 + coeff2[1], color="red")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_plots.png", dpi=150)
    plt.close(fig)


def plot_boxplot(panel):
    sample_persons = panel["personid"].drop_duplicates().sample(min(20, panel["personid"].nunique()), random_state=42)
    subset = panel[panel["personid"].isin(sample_persons)]
    fig, ax = plt.subplots(figsize=(14, 8))
    subset.boxplot(column="rate_gap", by="personid", ax=ax, rot=90)
    ax.set_title("Rate Gap distribution for a sample of households")
    ax.set_xlabel("Sample personid")
    ax.set_ylabel("Rate Gap")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rate_gap_boxplot_sample.png", dpi=150)
    plt.close(fig)


def correlation_tables(panel):
    base_vars = ["move_flag", "rate_gap", "mortgage_rate_annual"]
    corr_between = panel.groupby("personid")[base_vars].mean().corr()
    # Only use within variables that exist
    within_vars = [col for col in ["move_flag_within", "rate_gap_within"] if col in panel.columns]
    corr_within = panel[within_vars].corr() if within_vars else None
    corr_fd = panel[["move_flag_fd", "rate_gap_fd"]].corr()
    corr_twfe = panel[["move_flag_twfe", "rate_gap_twfe"]].corr()
    corr_between.to_csv(OUT_DIR / "corr_between.csv")
    corr_fd.to_csv(OUT_DIR / "corr_fd.csv")
    corr_twfe.to_csv(OUT_DIR / "corr_twfe.csv")
    return corr_between, corr_fd, corr_twfe


def save_report(panel, stats, variance_df, results):
    report_path = OUT_DIR / "analysis_report.md"
    with report_path.open("w") as f:
        f.write("# Household Mobility and Mortgage Rate Lock Analysis\n\n")
        f.write("## Dataset and panel construction\n")
        f.write("- Source: synthetic replication data from the package.\n")
        f.write("- Panel unit: `personid` (household).\n")
        f.write("- Time unit: quarterly observations based on `mortgage_payments_data.csv`.\n")
        f.write("- Dependent variable: `move_flag` indicates a mobility event in the quarter.\n")
        f.write("- Key explanatory variable: `rate_gap` = market mortgage rate minus existing mortgage rate.\n")
        f.write("- Sample restricted to households with at least 3 consecutive quarterly observations.\n\n")
        f.write("## Panel statistics\n")
        for key, value in stats.items():
            if key == "period_counts":
                f.write("\n### Observations per quarter\n")
                f.write(value.to_string())
                f.write("\n\n")
            else:
                f.write(f"- {key}: {value}\n")
        f.write("\n## Variance decomposition\n")
        f.write(variance_df.to_markdown(index=False))
        f.write("\n\n## Regression results\n")
        for name, model in results.items():
            f.write(f"### {name}\n\n")
            f.write("```\n")
            f.write(str(model.summary()))
            f.write("\n```\n\n")
    return report_path


def main():
    panel = build_panel_data()
    panel.to_csv(OUT_DIR / "panel_data.csv", index=False)
    stats = panel_statistics(panel)
    variables = ["move_flag", "rate_gap", "mortgage_rate_annual", "balance_to_orig"]
    variance_df = variance_decomposition(panel, variables)
    variance_df.to_csv(OUT_DIR / "variance_decomposition.csv", index=False)
    plot_distributions(panel)
    plot_scatter(panel)
    plot_boxplot(panel)
    corr_between, corr_fd, corr_twfe = correlation_tables(panel)
    corr_between.to_csv(OUT_DIR / "corr_between.csv")
    corr_fd.to_csv(OUT_DIR / "corr_fd.csv")
    corr_twfe.to_csv(OUT_DIR / "corr_twfe.csv")
    results = run_regressions(panel)
    report_path = save_report(panel, stats, variance_df, results)
    print(f"Analysis complete. Output files written to {OUT_DIR}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
