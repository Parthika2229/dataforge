# 🔧 DataForge — AI Data Pipeline Platform

> **Automated data engineering · Privacy shield · ML predictions · Anomaly detection**
> Free, open-source, runs entirely on your machine. No API key. No cloud required.

---

## 📸 What it does

DataForge is a full end-to-end data pipeline tool built with Streamlit. Upload any CSV and get:

- **Automated cleaning** — duplicates, missing values, encoding, outliers
- **Deep statistical profiling** — skewness, kurtosis, entropy, IQR per column
- **PII & PHI detection** — scans for sensitive data, scores GDPR / HIPAA / GLBA compliance
- **Smart insights** — quality score (0–100), automated warnings
- **Interactive visualizations** — histograms, bar charts, correlation heatmap
- **Anomaly detection** — Isolation Forest flags suspicious rows
- **ML predictions** — 7 models, auto leaderboard, SHAP explainability, live prediction
- **PDF compliance report** — professional audit document, one click

---

## 🚀 Quick Start

```bash
# 1. Extract the zip and navigate in
cd dataforge

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

Opens at **http://localhost:8501**

**If `streamlit` isn't found:**
```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
dataforge/
├── app.py                        # Main Streamlit application (9 steps)
├── requirements.txt              # All dependencies
├── .streamlit/
│   └── config.toml               # Dark teal theme
└── modules/
    ├── __init__.py
    ├── ingestion.py              # CSV loading + column type profiling
    ├── cleaning.py               # Automated cleaning pipeline
    ├── data_profiler.py          # Deep statistical profiling (NEW)
    ├── pii_detection.py          # PII/PHI detection + masking
    ├── insights.py               # Quality score + smart warnings
    ├── visualization.py          # Charts (histograms, heatmap, bar)
    ├── anomaly_detection.py      # Isolation Forest anomaly detection (NEW)
    ├── ml_advanced.py            # Full ML suite (7 models + SHAP + leaderboard)
    └── report_generator.py       # PDF compliance report
```

---

## 🗺️ The 9-Step Pipeline

| Step | Name | What it does |
|------|------|-------------|
| 1 | **Upload & Ingest** | Load CSV, detect types, preview, memory usage |
| 2 | **Clean & Transform** | Remove dupes, impute missing, cap outliers, one-hot encode |
| 3 | **Deep Profiler** | Skewness, kurtosis, IQR, entropy, outlier %, shape per column |
| 4 | **Privacy Shield** | PII/PHI scan, GDPR/HIPAA/GLBA scores, data masking |
| 5 | **Smart Insights** | Quality score (0–100), automated data quality warnings |
| 6 | **Visualize** | Histograms + KDE, bar charts, correlation heatmap |
| 7 | **Anomaly Detection** | Isolation Forest + Z-Score flags suspicious rows |
| 8 | **ML Prediction** | 7-model leaderboard, SHAP, hyperparameter tuning, live prediction |
| 9 | **Export Report** | Professional PDF compliance + audit report |

---

## ⚙️ Step-by-Step Settings Guide

### Step 1 · Upload & Ingest
- Supports CSV files up to 200MB
- Displays: rows, columns, missing cells, duplicates, memory usage
- **Column Profile** shows detected type, missing %, unique count, and a sample value
- Check column types are correct before proceeding

### Step 2 · Clean & Transform

| Option | When to use | Default |
|--------|-------------|---------|
| Remove duplicates | Always | ON |
| Imputation → Mean | Normal distributions (age, temperature) | — |
| Imputation → Median | Skewed distributions (prices, charges, salary) | ✅ Recommended |
| Imputation → Drop | Huge datasets, low missing % | — |
| Cap outliers (IQR) | Sensor errors, data entry mistakes | OFF |
| One-hot encode | Before running ML | ON |

> ⚠️ **Keep outlier capping OFF** for real-world financial columns (charges, salary, revenue). High values are real data, not errors.

### Step 3 · Deep Profiler
New in DataForge. Computes per column:
- **Skewness** — how asymmetric the distribution is
- **Kurtosis** — how heavy-tailed the distribution is
- **IQR** — interquartile range
- **CV %** — coefficient of variation
- **Outliers / Outliers %** — count of IQR outliers
- **Shape** — Normal / Mild skew / Heavy skew
- **Entropy** (categorical) — how diverse the values are

Automatically flags issues: high missing %, constant columns, ID-like columns, heavily skewed distributions.

### Step 4 · Privacy Shield

**Detection methods:**
- Name-based: column named `ssn`, `patient_name`, `diagnosis` → instantly flagged
- Content-based: scans actual values for SSN patterns, email, ICD codes, phone numbers, etc.

**Compliance scores:**

| Score | Status | Action |
|-------|--------|--------|
| 80–100 | ✅ COMPLIANT | Safe to share |
| 50–79 | ⚠️ REVIEW | Mask flagged columns |
| 0–49 | 🔴 AT RISK | Must mask before any sharing |

**Masking methods:**

| Method | Example output | Best for |
|--------|---------------|----------|
| Redact | `[REDACTED]` | External sharing |
| Hash (SHA-256) | `94e634ac63...` | Analytics, consistent tokens |
| Tokenize | `TOKEN_0042` | Internal use, re-identification possible |
| Partial Mask | `Jo****oe` | UI display |

> To reach HIPAA/GDPR 80+: mask PHI columns (ICD, NPI, diagnosis) with Hash, and PII columns (name, age, email) with Redact.

### Step 5 · Smart Insights

Quality score measures 4 dimensions:

| Dimension | Points | Measures |
|-----------|--------|----------|
| Completeness | 40 | % non-null values |
| Uniqueness | 20 | % non-duplicate rows |
| Consistency | 20 | % snake_case column names |
| Validity | 20 | % non-constant columns |

**Common warnings:**
- `sex_male & sex_female correlated 1.00` → Normal, always happens with one-hot encoding. Ignore.
- `smoker_yes & smoker_no correlated 1.00` → Same. Ignore.
- `Column has X extreme outliers` → Only worry if >10% of rows
- `Column had 70%+ missing` → Consider dropping the column

### Step 6 · Visualize

| Tab | Chart | What to look for |
|-----|-------|-----------------|
| Distributions | Histogram + KDE | Skew direction, bimodal peaks |
| Categoricals | Bar chart | Class imbalance (one bar 10x taller = problem) |
| Heatmap | Correlation matrix | Features correlated with your target column |

### Step 7 · Anomaly Detection

New in DataForge. Uses **Isolation Forest** (unsupervised ML) to find rows that don't fit the normal data pattern.

Settings:
- **Expected anomaly rate** — what % of your data you expect to be anomalous. Default: 5%
- **Z-Score threshold** — secondary filter. Default: 3.0 (flags values >3 standard deviations from mean)

Output:
- `anomaly_score` column — higher = more suspicious
- `is_anomaly` column — 1 = flagged, 0 = normal
- Visual: score distribution + scatter plots per feature
- Download anomaly report CSV

### Step 8 · ML Prediction

**Choose task type:**
- **Regression** → predicting a number (charges, salary, price, revenue)
- **Classification** → predicting a category (yes/no, fraud/legitimate, churn/retain)

**The 7 available models:**

| Model | Best for | Typical R² | Speed |
|-------|----------|------------|-------|
| XGBoost | Insurance, fraud, tabular data | 0.85–0.92 | Fast |
| LightGBM | Large datasets | 0.84–0.91 | Fastest |
| Random Forest | First baseline, mixed data | 0.82–0.88 | Medium |
| Gradient Boosting | Medium datasets | 0.81–0.87 | Slow |
| Ridge Regression | Linear relationships | 0.70–0.78 | Fastest |
| Decision Tree | Full interpretability | 0.68–0.75 | Fast |
| KNN | Small datasets | 0.60–0.72 | Slow |

**Workflow:**
1. Run **Leaderboard** first → identifies best model automatically
2. Go to **Single Model** tab → select leaderboard winner
3. Check **SHAP tab** → understand which features drive predictions
4. Test **Live Predict** tab → enter any values, get instant output
5. Download trained model as `.pkl` for deployment

**Reading your results:**

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|-----------|------|-----------|
| R² (regression) | <0.50 | 0.50–0.70 | 0.70–0.85 | 0.85+ |
| Accuracy (classification) | <0.60 | 0.60–0.75 | 0.75–0.90 | 0.90+ |
| CV Std | >0.10 | 0.05–0.10 | 0.02–0.05 | <0.02 |

---

## 📊 Supported Data Types

### Insurance Data
```
Target: charges → Regression → XGBoost
Imputation: Median
Outlier cap: OFF
Expected R²: 0.85–0.92
```

### Healthcare / Clinical
```
Target: readmission, outcome → Classification → Random Forest
Imputation: Median
Must mask: diagnosis, icd_code, npi, patient_name, dob
Required compliance: HIPAA 80+
```

### Financial / Banking
```
Target: fraud, default, approved → Classification → XGBoost
Imputation: Median
Must mask: account_no, ssn, credit_card, routing_no
Note: Use F1/AUC for imbalanced fraud datasets
```

### HR / Employee
```
Target: Attrition → Classification → Random Forest
Imputation: Mean (age) + Median (salary)
Must mask: EmployeeID, email, age
Note: Mild imbalance (84% No / 16% Yes)
```

### Retail / E-commerce
```
Target: sales, churn → Regression or Classification → LightGBM
Imputation: Median
Must mask: customer_id, email, address
Note: High-cardinality columns (product_id) skipped in encoding
```

---

## 🔧 Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.35 | Web UI |
| pandas | ≥2.0 | Data processing |
| numpy | ≥1.24 | Numerical computing |
| scikit-learn | ≥1.3 | ML models + Isolation Forest |
| xgboost | ≥2.0 | XGBoost model |
| lightgbm | ≥4.0 | LightGBM model |
| shap | ≥0.44 | Model explainability |
| matplotlib | ≥3.7 | Charts |
| seaborn | ≥0.13 | Statistical charts |
| scipy | ≥1.11 | Statistics |
| reportlab | ≥4.0 | PDF reports |

---

## ☁️ Free Deployment

### Streamlit Community Cloud (Easiest)
1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select repo → set main file to `app.py`
4. Deploy → get a public URL like `yourapp.streamlit.app`

### Render (Alternative)
1. Push to GitHub
2. New Web Service on [render.com](https://render.com)
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `app.py not found` | Run from inside the inner `dataforge` folder |
| `streamlit not recognized` | Use `python -m streamlit run app.py` |
| ML error: Not enough rows | Need ≥30 rows |
| ML error: Too many classes | Switch Task Type to Regression |
| SHAP not available | Use Random Forest, XGBoost, or LightGBM |
| Report button greyed out | Complete Step 4 (Privacy Shield) first |
| R² is negative | Wrong task type or target column |
| XGBoost not in list | `pip install xgboost` |
| LightGBM not in list | `pip install lightgbm` |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*DataForge — Built with Streamlit · Open Source · Free forever*
