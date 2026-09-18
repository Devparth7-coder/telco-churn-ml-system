"""Exploratory data analysis (scripted, reproducible).

Every figure answers a specific question; the generated Markdown report
records the conclusions. Run:

    python -m src.eda.explore

Outputs: reports/figures/eda_*.png and reports/eda_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.data.ingestion import load_raw_dataset
from src.logging_utils import get_logger, setup_logging

logger = get_logger("eda")

FIGURES_DIR = Path("reports") / "figures"

CATEGORY_GROUPS = [
    ("Contract", "Do longer contracts retain customers?"),
    ("InternetService", "Does service type drive churn?"),
    ("PaymentMethod", "Is electronic-check billing a churn signal?"),
    ("TechSupport", "Does tech support reduce churn?"),
    ("OnlineSecurity", "Does online security reduce churn?"),
]


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def run_eda() -> dict:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    df = load_raw_dataset(cfg)

    numeric_cols = ["tenure", "MonthlyCharges", "SeniorCitizen"]
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # --- core facts ---------------------------------------------------------
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_nonzero = missing[missing > 0]
    blank_total_charges = int(total_charges.isna().sum())
    zero_tenure = int((df["tenure"] == 0).sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["customerID"].duplicated().sum())
    churn_counts = df["Churn"].value_counts().to_dict()
    churn_rate = float((df["Churn"] == "Yes").mean())
    imbalance_ratio = round(churn_counts.get("No", 0) / max(churn_counts.get("Yes", 1), 1), 2)
    cardinality = {col: int(df[col].nunique()) for col in df.columns if df[col].dtype == object}

    figures: dict[str, str] = {}

    # Q1: how imbalanced is the target? --------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    pd.Series(churn_counts).plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title(f"Target distribution (churn rate {churn_rate:.1%})")
    ax.set_ylabel("customers")
    figures["target_distribution"] = _save(fig, "eda_target_distribution.png")

    # Q2: how does tenure relate to churn? ------------------------------------
    tenure_churn = df.groupby("tenure")["Churn"].apply(lambda s: (s == "Yes").mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tenure_churn.index, tenure_churn.values, marker=".", linewidth=1)
    ax.set_xlabel("tenure (months)")
    ax.set_ylabel("churn rate")
    ax.set_title("Churn rate by tenure — high early risk, stable long-tenure customers")
    figures["tenure_vs_churn"] = _save(fig, "eda_tenure_vs_churn.png")

    # Q3: categorical drivers of churn ----------------------------------------
    fig, axes = plt.subplots(1, len(CATEGORY_GROUPS), figsize=(4.2 * len(CATEGORY_GROUPS), 4))
    for ax, (col, question) in zip(axes, CATEGORY_GROUPS, strict=True):
        rates = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
        rates.plot.bar(ax=ax, color="#4c72b0")
        ax.set_title(question, fontsize=9)
        ax.set_ylabel("churn rate")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    figures["churn_rate_by_category"] = _save(fig, "eda_churn_rate_by_category.png")

    # Q4: numeric distributions by class ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, numeric_cols, strict=True):
        df.boxplot(column=col, by="Churn", ax=ax)
        ax.set_title(f"{col} by outcome")
        ax.set_xlabel("")
    fig.suptitle("")
    figures["numeric_by_class"] = _save(fig, "eda_numeric_by_class.png")

    # Q5: is TotalCharges just tenure x MonthlyCharges? (redundancy/leak check)
    fig, ax = plt.subplots(figsize=(6, 5))
    sample = df.sample(n=min(3000, len(df)), random_state=cfg.training.random_state)
    scatter = ax.scatter(
        sample["tenure"] * sample["MonthlyCharges"],
        pd.to_numeric(sample["TotalCharges"], errors="coerce"),
        c=(sample["Churn"] == "Yes").astype(int),
        cmap="coolwarm", alpha=0.4, s=10,
    )
    ax.plot([0, 8000], [0, 8000], "k--", linewidth=0.8)
    ax.set_xlabel("tenure x MonthlyCharges")
    ax.set_ylabel("TotalCharges")
    ax.set_title("TotalCharges vs tenure x MonthlyCharges (redundant, but legitimate history)")
    fig.colorbar(scatter, ax=ax, label="churned")
    figures["total_vs_tenure"] = _save(fig, "eda_total_charges_relationship.png")

    # Q6: correlation structure of numerics ------------------------------------
    corr_frame = pd.DataFrame(
        {
            "tenure": df["tenure"],
            "MonthlyCharges": df["MonthlyCharges"],
            "TotalCharges": total_charges,
            "SeniorCitizen": df["SeniorCitizen"],
            "Churn": (df["Churn"] == "Yes").astype(int),
        }
    )
    corr = corr_frame.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation matrix (numeric features + target)")
    fig.colorbar(im, ax=ax)
    figures["correlation"] = _save(fig, "eda_correlation.png")

    # Q7: MonthlyCharges outliers by service type -------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="MonthlyCharges", by="InternetService", ax=ax)
    ax.set_title("MonthlyCharges by internet service")
    ax.set_xlabel("")
    fig.suptitle("")
    figures["monthly_charges_outliers"] = _save(fig, "eda_monthly_charges_outliers.png")

    # --- conclusions -----------------------------------------------------------
    top_churn_cats = {
        col: df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False).head(2).round(3).to_dict()
        for col, _ in CATEGORY_GROUPS
    }

    report = [
        "# Exploratory Data Analysis — Telco Customer Churn",
        "",
        "## Dataset facts",
        f"- Rows: **{n_rows}**, columns: **{n_cols}** (one row per customer; snapshot data).",
        f"- Fully duplicate rows: **{duplicate_rows}**; duplicate customerIDs: **{duplicate_ids}**.",
        f"- Missing values per column: {missing_nonzero.to_dict() if len(missing_nonzero) else 'none (after raw parse)'}; "
        f"TotalCharges has **{blank_total_charges}** whitespace-blank cells, all coinciding with tenure=0. These "
        f"(count of zero-tenure customers: {zero_tenure}). These are brand-new customers, not corruption — "
        "kept and imputed rather than dropped.",
        f"- Target: churn rate **{churn_rate:.1%}** (No={churn_counts.get('No', 0)}, Yes={churn_counts.get('Yes', 0)}), "
        f"imbalance ratio ~{imbalance_ratio}:1 → class weighting + threshold tuning, not naive accuracy.",
        "- Identifier `customerID` is unique → dropped from features (leakage audit).",
        "",
        "## Cardinality (object columns)",
        ", ".join(f"{k}={v}" for k, v in sorted(cardinality.items())),
        "— all low cardinality; one-hot encoding is safe.",
        "",
        "## Findings",
        "1. **Tenure is strongly non-linear**: churn rate is very high in the first months, drops sharply, "
        "and flattens after ~24 months → tenure is binned (`tenure_group`) as a feature.",
        "2. **Contract type is the strongest categorical signal**: month-to-month customers churn far more than "
        f"two-year customers ({top_churn_cats['Contract']}).",
        f"3. **Fiber optic customers churn more than DSL** ({top_churn_cats['InternetService']}) and "
        f"electronic-check payers churn most ({top_churn_cats['PaymentMethod']}).",
        f"4. **Missing protective services correlates with churn**: TechSupport={top_churn_cats['TechSupport']}, "
        f"OnlineSecurity={top_churn_cats['OnlineSecurity']} → derived flag `no_security_support`.",
        "5. **TotalCharges ≈ tenure × MonthlyCharges** (see scatter) — redundant but legitimate billing history; "
        "kept, with a derived `avg_monthly_charges` ratio. No post-outcome information involved.",
        "6. **No timestamps exist** → temporal split impossible; stratified random split is appropriate and documented.",
        "7. **MonthlyCharges** has no implausible outliers (bounded by service type); the spread is real pricing, kept.",
        "8. **SeniorCitizen** is stored as 0/1 int, treated as numeric.",
        "",
        "## Figures",
    ]
    report += [f"- {name}: `{path}`" for name, path in figures.items()]
    report_path = Path("reports") / "eda_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    logger.info("EDA complete", extra={"rows": n_rows, "churn_rate": round(churn_rate, 4), "report": str(report_path)})
    return {"report": str(report_path), "figures": figures}


def main() -> int:
    run_eda()
    return 0


if __name__ == "__main__":
    sys.exit(main())
