# Exploratory Data Analysis — Telco Customer Churn

## Dataset facts
- Rows: **7043**, columns: **21** (one row per customer; snapshot data).
- Fully duplicate rows: **0**; duplicate customerIDs: **0**.
- Missing values per column: none (after raw parse); TotalCharges has **11** whitespace-blank cells, all coinciding with tenure=0. These (count of zero-tenure customers: 11). These are brand-new customers, not corruption — kept and imputed rather than dropped.
- Target: churn rate **26.5%** (No=5174, Yes=1869), imbalance ratio ~2.77:1 → class weighting + threshold tuning, not naive accuracy.
- Identifier `customerID` is unique → dropped from features (leakage audit).

## Cardinality (object columns)
Churn=2, Contract=3, Dependents=2, DeviceProtection=3, InternetService=3, MultipleLines=3, OnlineBackup=3, OnlineSecurity=3, PaperlessBilling=2, Partner=2, PaymentMethod=4, PhoneService=2, StreamingMovies=3, StreamingTV=3, TechSupport=3, customerID=7043, gender=2
— all low cardinality; one-hot encoding is safe.

## Findings
1. **Tenure is strongly non-linear**: churn rate is very high in the first months, drops sharply, and flattens after ~24 months → tenure is binned (`tenure_group`) as a feature.
2. **Contract type is the strongest categorical signal**: month-to-month customers churn far more than two-year customers ({'Month-to-month': 0.427, 'One year': 0.113}).
3. **Fiber optic customers churn more than DSL** ({'Fiber optic': 0.419, 'DSL': 0.19}) and electronic-check payers churn most ({'Electronic check': 0.453, 'Mailed check': 0.191}).
4. **Missing protective services correlates with churn**: TechSupport={'No': 0.416, 'Yes': 0.152}, OnlineSecurity={'No': 0.418, 'Yes': 0.146} → derived flag `no_security_support`.
5. **TotalCharges ≈ tenure × MonthlyCharges** (see scatter) — redundant but legitimate billing history; kept, with a derived `avg_monthly_charges` ratio. No post-outcome information involved.
6. **No timestamps exist** → temporal split impossible; stratified random split is appropriate and documented.
7. **MonthlyCharges** has no implausible outliers (bounded by service type); the spread is real pricing, kept.
8. **SeniorCitizen** is stored as 0/1 int, treated as numeric.

## Figures
- target_distribution: `reports/figures/eda_target_distribution.png`
- tenure_vs_churn: `reports/figures/eda_tenure_vs_churn.png`
- churn_rate_by_category: `reports/figures/eda_churn_rate_by_category.png`
- numeric_by_class: `reports/figures/eda_numeric_by_class.png`
- total_vs_tenure: `reports/figures/eda_total_charges_relationship.png`
- correlation: `reports/figures/eda_correlation.png`
- monthly_charges_outliers: `reports/figures/eda_monthly_charges_outliers.png`
