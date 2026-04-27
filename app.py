import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from modules.ingestion       import load_data, detect_data_profile
from modules.cleaning        import clean_pipeline
from modules.insights        import generate_insights, compute_quality_score
from modules.visualization   import plot_distributions, plot_correlation_heatmap, plot_categoricals
from modules.pii_detection   import detect_pii_phi, mask_column, compute_compliance_score
from modules.ml_advanced     import run_single_model, run_leaderboard, predict_single_row, get_available_models
from modules.anomaly_detection import run_isolation_forest, zscore_anomalies, anomaly_summary, plot_anomaly_overview
from modules.data_profiler   import profile_dataframe, flag_issues
from modules.report_generator import generate_report

matplotlib.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor":   "#0a0e1a",
    "axes.edgecolor":   "#2d3748",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e293b",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
})

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataForge — AI Data Pipeline",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #060910 !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main background */
.stApp { background: #0a0e1a; }
.main  { background: #0a0e1a; }
[data-testid="stAppViewContainer"] { background: #0a0e1a; }

/* Cards */
.forge-card {
    background: #111827;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    border: 1px solid #1e293b;
    border-top: 3px solid #00d4aa;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.forge-card h3 {
    margin-top: 0;
    color: #00d4aa;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Syne', sans-serif !important;
}

/* Metric overrides */
[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size:0.78rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4aa !important; font-family:'Syne',sans-serif; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00a896) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,212,170,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0f172a; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #1e293b !important; color: #00d4aa !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #1e293b !important; border-radius: 10px; }

/* Expander */
.streamlit-expanderHeader { color: #00d4aa !important; }

/* Download button */
.stDownloadButton > button {
    background: #1e293b !important;
    color: #00d4aa !important;
    border: 1px solid #00d4aa !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Pipeline steps */
.forge-step {
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    margin: 0.25rem 0;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Inter', sans-serif;
}
.fs-done   { background: #0d2e27; color: #00d4aa !important; border-left: 3px solid #00d4aa; }
.fs-active { background: #1a1f35; color: #818cf8 !important; border-left: 3px solid #818cf8; }
.fs-idle   { background: #0f172a; color: #475569 !important; border-left: 3px solid #1e293b; }

/* Quality badge */
.qbadge {
    display: inline-block;
    padding: 0.4rem 1.4rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.qb-high { background: #0d2e27; color: #00d4aa; border: 1px solid #00d4aa; }
.qb-mid  { background: #2a1f0a; color: #fbbf24; border: 1px solid #fbbf24; }
.qb-low  { background: #2a0a0a; color: #f87171; border: 1px solid #f87171; }

/* Insight pills */
.ipill      { display:inline-block; background:#0f2d3d; color:#67e8f9;
               border:1px solid #0e7490; border-radius:6px; padding:3px 10px;
               font-size:0.8rem; margin:2px 3px; }
.ipill-warn { background:#2a1f0a; color:#fbbf24; border-color:#d97706; }
.ipill-err  { background:#2a0a0a; color:#f87171; border-color:#dc2626; }

/* Section divider */
.forge-divider { border:none; border-top:1px solid #1e293b; margin:1.5rem 0; }

/* Hero gradient text */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["raw_df","clean_df","profile","insights","ml_results","pii_df",
          "compliance_scores","quality_score","current_step","dataset_name",
          "cleaning_report","leaderboard_df","anomaly_df","deep_profile"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.current_step is None:
    st.session_state.current_step = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                    background:linear-gradient(135deg,#00d4aa,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🔧 DataForge
        </div>
        <div style='color:#475569;font-size:0.78rem;margin-top:2px;'>
            AI Data Pipeline Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #1e293b;margin:0.5rem 0;'>", unsafe_allow_html=True)

    steps = [
        ("📂","Upload & Ingest"),
        ("🧹","Clean & Transform"),
        ("🔬","Deep Profiler"),
        ("🔐","Privacy Shield"),
        ("💡","Smart Insights"),
        ("📊","Visualize"),
        ("🚨","Anomaly Detection"),
        ("🤖","ML Prediction"),
        ("📋","Export Report"),
    ]
    for i,(ic,s) in enumerate(steps):
        cs  = st.session_state.current_step
        cls = "fs-done" if i < cs else ("fs-active" if i == cs else "fs-idle")
        chk = "✓" if i < cs else ("▶" if i == cs else "○")
        st.markdown(f'<div class="forge-step {cls}">{chk} {ic} {s}</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #1e293b;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#334155;font-size:0.75rem;'>Upload any CSV to begin</div>",
                unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
col_h, col_badge = st.columns([4, 1])
with col_h:
    st.markdown('<div class="hero-title">🔧 DataForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automated data pipeline · Privacy shield · ML predictions · Anomaly detection</div>',
                unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem;text-align:center;'>
        <div style='color:#00d4aa;font-family:Syne,sans-serif;font-size:0.7rem;font-weight:700;'>
            v2.0 PUBLIC
        </div>
        <div style='color:#475569;font-size:0.65rem;'>Free & Open Source</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='forge-divider'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1 · UPLOAD
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="forge-card"><h3>📂 Step 1 · Upload & Ingest</h3>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")

if uploaded:
    df, err = load_data(uploaded)
    if err:
        st.error(f"Could not read file: {err}")
    else:
        st.session_state.raw_df       = df
        st.session_state.dataset_name = uploaded.name
        st.session_state.current_step = max(st.session_state.current_step, 1)
        profile = detect_data_profile(df)
        st.session_state.profile = profile

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Rows",           f"{df.shape[0]:,}")
        c2.metric("Columns",        df.shape[1])
        c3.metric("Missing cells",  f"{df.isnull().sum().sum():,}")
        c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")
        c5.metric("Memory",         f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.dataframe(df.head(8), use_container_width=True)

        with st.expander("🔍 Column Profile"):
            prof_df = pd.DataFrame({
                "Detected Type": profile["col_types"],
                "Missing":       df.isnull().sum(),
                "Missing %":    (df.isnull().mean()*100).round(1),
                "Unique":        df.nunique(),
                "Sample":       [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0]>0 else "—" for c in df.columns],
            })
            st.dataframe(prof_df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 2 · CLEANING
# ═══════════════════════════════════════════════════════════════
if st.session_state.raw_df is not None:
    st.markdown('<div class="forge-card"><h3>🧹 Step 2 · Clean & Transform</h3>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        remove_dupes    = st.checkbox("Remove duplicate rows", value=True)
        impute_num      = st.selectbox("Numerical imputation", ["mean","median","drop"])
    with c2:
        handle_outliers = st.checkbox("Cap outliers (IQR method)", value=False)
        encode_cat      = st.checkbox("One-hot encode categoricals", value=True)
    with c3:
        st.markdown("**Imputation guide**")
        st.markdown("<small style='color:#64748b'>Mean → normal distributions<br>"
                    "Median → skewed data, prices<br>"
                    "Drop → large datasets only</small>", unsafe_allow_html=True)

    if st.button("⚡ Run Cleaning Pipeline", type="primary"):
        with st.spinner("Running cleaning engine…"):
            clean_df, report = clean_pipeline(
                st.session_state.raw_df.copy(),
                remove_dupes=remove_dupes, num_strategy=impute_num,
                handle_outliers=handle_outliers, encode_cat=encode_cat,
            )
            st.session_state.clean_df        = clean_df
            st.session_state.cleaning_report = report
            st.session_state.current_step    = max(st.session_state.current_step, 2)

        st.success("✅ Cleaning complete!")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",     f"{clean_df.shape[0]:,}", delta=f"{clean_df.shape[0]-st.session_state.raw_df.shape[0]}")
        c2.metric("Columns",  clean_df.shape[1],        delta=f"+{clean_df.shape[1]-st.session_state.raw_df.shape[1]}")
        c3.metric("Missing",  f"{clean_df.isnull().sum().sum():,}")
        c4.metric("Transforms", len(report))

        with st.expander("📋 Cleaning Log"):
            for item in report:
                st.markdown(f"<span style='color:#64748b;font-size:0.85rem;'>→ {item}</span>", unsafe_allow_html=True)
        st.dataframe(clean_df.head(6), use_container_width=True)
        st.download_button("⬇️ Download Cleaned CSV", clean_df.to_csv(index=False).encode(),
                           "cleaned_data.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 3 · DEEP PROFILER (NEW)
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>🔬 Step 3 · Deep Data Profiler</h3>', unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b'>Complete statistical profile — skewness, kurtosis, IQR, entropy, outlier counts per column.</small>",
                unsafe_allow_html=True)

    if st.button("🔬 Run Deep Profile", type="primary"):
        with st.spinner("Profiling every column…"):
            dp = profile_dataframe(st.session_state.clean_df)
            issues = flag_issues(dp)
            st.session_state.deep_profile = (dp, issues)
            st.session_state.current_step = max(st.session_state.current_step, 3)

    if st.session_state.deep_profile is not None:
        dp, issues = st.session_state.deep_profile
        st.success(f"✅ Profile complete — {len(dp)} columns analysed")

        # Issues
        if issues:
            st.markdown("**⚠️ Flagged issues:**")
            for iss in issues:
                color = "#f87171" if iss["level"]=="error" else ("#fbbf24" if iss["level"]=="warn" else "#67e8f9")
                icon  = "🔴" if iss["level"]=="error" else ("🟡" if iss["level"]=="warn" else "🔵")
                st.markdown(f"<span style='color:{color};font-size:0.83rem;'>{icon} {iss['msg']}</span>",
                            unsafe_allow_html=True)
            st.markdown("")

        # Full profile table — show numerical stats
        num_cols_in_profile = [c for c in dp.columns if c not in ("Mode","Mode freq %","Entropy")]
        st.dataframe(dp, use_container_width=True)

        # Quick charts — top 3 skewed columns
        num_df = dp[dp["Skewness"].notna()].sort_values("Skewness", key=abs, ascending=False)
        if len(num_df) > 0:
            with st.expander("📊 Distribution Shape Explorer"):
                top_skewed = num_df.head(4).index.tolist()
                fig_sk, axes = plt.subplots(1, len(top_skewed), figsize=(4*len(top_skewed), 3))
                if len(top_skewed) == 1: axes = [axes]
                for ax, col in zip(axes, top_skewed):
                    data = st.session_state.clean_df[col].dropna()
                    ax.hist(data, bins=25, color="#00d4aa", alpha=0.8, edgecolor="none")
                    skew_val = dp.loc[col, "Skewness"]
                    ax.set_title(f"{col}\nskew={skew_val}", color="white", fontsize=9, fontweight="bold")
                    ax.tick_params(labelsize=7)
                fig_sk.tight_layout()
                st.pyplot(fig_sk, use_container_width=True)

        st.download_button("⬇️ Download Profile CSV", dp.reset_index().to_csv(index=False).encode(),
                           "data_profile.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 4 · PRIVACY SHIELD
# ═══════════════════════════════════════════════════════════════
if st.session_state.raw_df is not None:
    st.markdown('<div class="forge-card"><h3>🔐 Step 4 · Privacy Shield</h3>', unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b'>Detects PII & PHI by column name and content patterns. "
                "Scores GDPR / HIPAA / GLBA compliance readiness.</small>", unsafe_allow_html=True)

    if st.button("🔍 Scan for PII & PHI", type="primary"):
        with st.spinner("Scanning columns for sensitive data…"):
            pii_df = detect_pii_phi(st.session_state.raw_df)
            comp   = compute_compliance_score(pii_df, len(st.session_state.raw_df.columns))
            st.session_state.pii_df            = pii_df
            st.session_state.compliance_scores = comp
            st.session_state.current_step      = max(st.session_state.current_step, 4)

    if st.session_state.pii_df is not None:
        pii_df = st.session_state.pii_df
        comp   = st.session_state.compliance_scores

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Columns", len(pii_df))
        c2.metric("PHI Detected",  int((pii_df["type"]=="PHI").sum()))
        c3.metric("PII Detected",  int((pii_df["type"]=="PII").sum()))
        c4.metric("Safe Columns",  int((pii_df["type"]=="Safe").sum()))

        st.markdown("#### Compliance Readiness")
        cc1,cc2,cc3,cc4 = st.columns(4)
        for cw, reg in zip([cc1,cc2,cc3,cc4], ["HIPAA","GDPR","GLBA","overall"]):
            s     = comp.get(reg, 100)
            label = reg if reg != "overall" else "Overall"
            status = "✅ COMPLIANT" if s>=80 else ("⚠️ REVIEW" if s>=50 else "🔴 AT RISK")
            cw.metric(label, f"{s}/100", status)

        def _style_pii(row):
            if row["type"]=="PHI":  return ["background-color:#2a0a0a"]*len(row)
            elif row["type"]=="PII": return ["background-color:#2a1f0a"]*len(row)
            return [""]*len(row)
        st.dataframe(pii_df.style.apply(_style_pii, axis=1), use_container_width=True)

        flagged_cols = pii_df[pii_df["type"] != "Safe"]["column"].tolist()
        mask_src = st.session_state.raw_df
        if flagged_cols:
            st.markdown("#### 🛡️ Apply Data Masking")
            mc1, mc2 = st.columns(2)
            with mc1: mask_col_sel = st.selectbox("Column to mask", flagged_cols)
            with mc2: mask_method  = st.selectbox("Masking method",
                                                   ["Redact","Hash (SHA-256)","Tokenize","Partial Mask"])

            before_series = mask_src[mask_col_sel].head(8).reset_index(drop=True)
            after_series  = mask_column(mask_src[mask_col_sel], mask_method).head(8).reset_index(drop=True)
            after_series.name = mask_col_sel

            preview_df = pd.DataFrame({
                f"BEFORE — {mask_col_sel}": before_series,
                f"AFTER  — {mask_method}":  after_series,
            })
            st.dataframe(preview_df, use_container_width=True)

            if st.button("Apply Mask & Download Protected Dataset"):
                protected = mask_src.copy()
                protected[mask_col_sel] = mask_column(protected[mask_col_sel], mask_method)
                st.download_button("⬇️ Download Privacy-Safe CSV",
                                   protected.to_csv(index=False).encode(),
                                   "protected_data.csv","text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 5 · INSIGHTS
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>💡 Step 5 · Smart Insights</h3>', unsafe_allow_html=True)
    insights = generate_insights(st.session_state.clean_df, st.session_state.raw_df)
    score    = compute_quality_score(st.session_state.clean_df)
    st.session_state.insights      = insights
    st.session_state.quality_score = score
    st.session_state.current_step  = max(st.session_state.current_step, 5)

    cls = "qb-high" if score>=80 else ("qb-mid" if score>=50 else "qb-low")
    st.markdown(f'<span class="qbadge {cls}">Quality Score: {score} / 100</span>', unsafe_allow_html=True)

    for ins in insights:
        pill = "ipill ipill-err" if ins["level"]=="error" else \
               ("ipill ipill-warn" if ins["level"]=="warn" else "ipill")
        icon = "🔴" if ins["level"]=="error" else ("🟡" if ins["level"]=="warn" else "🟢")
        st.markdown(f'<span class="{pill}">{icon} {ins["msg"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 6 · VISUALIZATION
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>📊 Step 6 · Visualize</h3>', unsafe_allow_html=True)
    df_vis   = st.session_state.clean_df
    num_cols = df_vis.select_dtypes(include=np.number).columns.tolist()
    cat_cols_raw = st.session_state.raw_df.select_dtypes(include="object").columns.tolist()

    tab1,tab2,tab3 = st.tabs(["📈 Distributions","🗂️ Categoricals","🔥 Correlation Heatmap"])
    with tab1:
        if num_cols:
            sel = st.multiselect("Columns", num_cols, default=num_cols[:min(4,len(num_cols))])
            if sel: st.pyplot(plot_distributions(df_vis, sel), use_container_width=True)
    with tab2:
        if cat_cols_raw:
            sel2 = st.multiselect("Columns", cat_cols_raw, default=cat_cols_raw[:min(3,len(cat_cols_raw))])
            if sel2: st.pyplot(plot_categoricals(st.session_state.raw_df, sel2), use_container_width=True)
    with tab3:
        if len(num_cols) > 1:
            st.pyplot(plot_correlation_heatmap(df_vis, num_cols[:15]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 7 · ANOMALY DETECTION (NEW)
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>🚨 Step 7 · Anomaly Detection</h3>', unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b'>Uses Isolation Forest + Z-Score to detect suspicious rows. "
                "Flags outliers that don't fit the normal data pattern.</small>", unsafe_allow_html=True)

    contamination = st.slider("Expected anomaly rate (%)", 1, 20, 5, 1) / 100
    zscore_thresh = st.slider("Z-Score threshold", 2.0, 5.0, 3.0, 0.5)

    if st.button("🚨 Run Anomaly Detection", type="primary"):
        with st.spinner("Running Isolation Forest…"):
            df_anom = run_isolation_forest(st.session_state.clean_df, contamination=contamination)
            df_anom = zscore_anomalies(df_anom, threshold=zscore_thresh)
            summary = anomaly_summary(df_anom)
            st.session_state.anomaly_df   = df_anom
            st.session_state.current_step = max(st.session_state.current_step, 7)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total rows",     f"{summary['total_rows']:,}")
        c2.metric("Anomalies found",f"{summary['anomaly_count']:,}")
        c3.metric("Anomaly rate",   f"{summary['anomaly_pct']}%")
        c4.metric("Normal rows",    f"{summary['total_rows']-summary['anomaly_count']:,}")

        # Visual
        fig_anom = plot_anomaly_overview(df_anom)
        st.pyplot(fig_anom, use_container_width=True)

        # Table of anomalous rows
        anom_rows = df_anom[df_anom["is_anomaly"]==1].drop(columns=["zscore_anomaly"], errors="ignore")
        with st.expander(f"🔍 View {len(anom_rows)} anomalous rows"):
            st.dataframe(anom_rows.head(50), use_container_width=True)

        st.download_button("⬇️ Download Anomaly Report CSV",
                           df_anom.to_csv(index=False).encode(),
                           "anomaly_report.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 8 · ADVANCED ML
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>🤖 Step 8 · ML Prediction</h3>', unsafe_allow_html=True)
    df_ml       = st.session_state.clean_df
    num_cols_ml = df_ml.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols_ml) < 2:
        st.warning("Need at least 2 numerical columns.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1: target_col = st.selectbox("Target column (what to predict)", num_cols_ml)
        with c2: task_type  = st.radio("Task type", ["Classification","Regression"], horizontal=True)
        with c3: test_size  = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        available_models = list(get_available_models(task_type).keys())

        ml_tab1, ml_tab2 = st.tabs([f"🏆 Leaderboard — All {len(available_models)} Models",
                                     "🎯 Single Model + SHAP + Live Predict"])

        with ml_tab1:
            if st.button("🚀 Run Leaderboard", type="primary"):
                with st.spinner(f"Training {len(available_models)} models…"):
                    lb_df, lb_err = run_leaderboard(df_ml, target_col, task_type, test_size, cv_folds)
                if lb_err:
                    st.error(lb_err)
                else:
                    st.session_state.leaderboard_df = lb_df
                    best = lb_df.iloc[0]
                    st.success(f"🥇 Best model: **{best['Model']}** — Score: {best['Primary Score']}")

                    def _hl_top(row):
                        if row.name == 1: return ["background:#0d2e27;font-weight:bold"]*len(row)
                        return [""]*len(row)
                    st.dataframe(lb_df.style.apply(_hl_top, axis=1), use_container_width=True)

                    valid = lb_df[lb_df["Primary Score"] > -900]
                    fig_lb, ax = plt.subplots(figsize=(9, 3.5))
                    colors_lb = ["#00d4aa" if i==0 else "#1e3a35" for i in range(len(valid))]
                    ax.barh(valid["Model"][::-1], valid["Primary Score"][::-1],
                            color=colors_lb[::-1], edgecolor="none", height=0.6)
                    ax.set_xlabel("Primary Score")
                    ax.set_title("Model Leaderboard", fontweight="bold", color="white")
                    fig_lb.tight_layout()
                    st.pyplot(fig_lb, use_container_width=True)

        with ml_tab2:
            model_name = st.selectbox("Model", available_models)
            with st.expander("⚙️ Hyperparameters"):
                hparams = {}
                if any(x in model_name for x in ["Random Forest","XGBoost","LightGBM","Gradient"]):
                    hparams["n_estimators"] = st.slider("n_estimators", 50, 500, 100, 50)
                    hparams["max_depth"]    = st.slider("max_depth", 2, 20, 6)
                elif "Ridge" in model_name:
                    hparams["alpha"] = st.number_input("alpha", 0.01, 100.0, 1.0)
                elif "KNN" in model_name:
                    hparams["n_neighbors"] = st.slider("n_neighbors", 2, 20, 5)
                else:
                    st.caption("No tunable params.")

            if st.button("🎯 Train Model", type="primary"):
                with st.spinner(f"Training {model_name}…"):
                    results = run_single_model(df_ml, target_col, model_name, task_type,
                                              test_size, cv_folds, hparams or None)
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state.ml_results   = results
                    st.session_state.current_step = max(st.session_state.current_step, 8)
                    st.success(f"✅ {model_name} trained!")

                    metrics = results["metrics"]
                    mc = st.columns(min(len(metrics), 4))
                    for i,(k,v) in enumerate(metrics.items()):
                        mc[i%len(mc)].metric(k, v)

                    ta,tb,tc,td = st.tabs(["📊 Feature Importance","🧠 SHAP","📈 Predictions","🔮 Live Predict"])

                    with ta:
                        fi = results.get("feature_importance")
                        if fi is not None:
                            fig_fi, ax = plt.subplots(figsize=(7, min(5, len(fi.head(12))*0.45+1.5)))
                            ax.barh(fi["feature"].head(12)[::-1], fi["importance"].head(12)[::-1],
                                    color="#00d4aa", edgecolor="none")
                            ax.set_title("Feature Importances", fontweight="bold", color="white")
                            fig_fi.tight_layout()
                            st.pyplot(fig_fi, use_container_width=True)
                        else:
                            st.info("Not available for this model.")

                    with tb:
                        shap_vals = results.get("shap_values")
                        if shap_vals is not None:
                            try:
                                feat_names = results.get("shap_features",[])
                                arr = np.abs(shap_vals) if shap_vals.ndim==2 else shap_vals
                                shap_mean = pd.DataFrame(arr, columns=feat_names).mean().sort_values(ascending=False).head(10)
                                fig_sh, ax = plt.subplots(figsize=(7, 3.5))
                                ax.barh(shap_mean.index[::-1], shap_mean.values[::-1], color="#818cf8", edgecolor="none")
                                ax.set_xlabel("Mean |SHAP value|")
                                ax.set_title("SHAP Feature Impact", fontweight="bold", color="white")
                                fig_sh.tight_layout()
                                st.pyplot(fig_sh, use_container_width=True)
                                st.caption("Higher = more influence on model predictions.")
                            except Exception as e:
                                st.info(f"SHAP unavailable: {e}")
                        else:
                            st.info("SHAP not available for this model.")

                    with tc:
                        pred_df = results["predictions_df"]
                        st.dataframe(pred_df.head(25), use_container_width=True)
                        if task_type == "Regression":
                            fig_r, ax = plt.subplots(figsize=(6,4))
                            ax.scatter(pred_df["actual"], pred_df["predicted"],
                                       alpha=0.6, color="#00d4aa", edgecolors="none", s=30)
                            mn = min(pred_df["actual"].min(), pred_df["predicted"].min())
                            mx = max(pred_df["actual"].max(), pred_df["predicted"].max())
                            ax.plot([mn,mx],[mn,mx],"--",color="#ff6b6b",alpha=0.7,label="Perfect fit",linewidth=1.5)
                            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                            ax.set_title("Actual vs Predicted", fontweight="bold", color="white")
                            ax.legend(labelcolor="white",framealpha=0.2)
                            fig_r.tight_layout()
                            st.pyplot(fig_r, use_container_width=True)
                        elif results.get("confusion_matrix") is not None:
                            cm_data = results["confusion_matrix"]
                            fig_cm, ax = plt.subplots(figsize=(4,3))
                            sns.heatmap(cm_data, annot=True, fmt="d",
                                        cmap=sns.light_palette("#00d4aa", as_cmap=True), ax=ax)
                            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                            ax.set_title("Confusion Matrix", fontweight="bold", color="white")
                            fig_cm.tight_layout()
                            st.pyplot(fig_cm, use_container_width=True)
                        st.download_button("⬇️ Download Predictions", pred_df.to_csv(index=False).encode(),
                                           "predictions.csv","text/csv")

                    with td:
                        st.markdown("**Enter values for a live prediction:**")
                        feat_names = results.get("shap_features") or \
                                     df_ml.drop(columns=[target_col]).select_dtypes(include=np.number).columns.tolist()
                        input_vals = {}
                        lp_cols = st.columns(3)
                        for i, feat in enumerate(feat_names[:12]):
                            mn = float(df_ml[feat].min()) if feat in df_ml else 0.0
                            mx = float(df_ml[feat].max()) if feat in df_ml else 100.0
                            mu = float(df_ml[feat].mean()) if feat in df_ml else 0.0
                            input_vals[feat] = lp_cols[i%3].number_input(feat, min_value=mn, max_value=mx, value=round(mu,2))
                        if st.button("🔮 Predict"):
                            try:
                                pred = predict_single_row(results["model_bytes"], input_vals)
                                st.markdown(f"""
                                <div style='background:#0d2e27;border:1px solid #00d4aa;border-radius:10px;
                                            padding:1rem 1.5rem;margin-top:0.5rem;'>
                                    <div style='color:#64748b;font-size:0.8rem;'>Predicted value for <b>{target_col}</b></div>
                                    <div style='color:#00d4aa;font-family:Syne,sans-serif;font-size:2rem;font-weight:800;'>
                                        {round(float(pred), 4)}
                                    </div>
                                </div>""", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Prediction error: {e}")

                    st.download_button("⬇️ Download Trained Model (.pkl)",
                                       results["model_bytes"],
                                       f"{model_name.lower().replace(' ','_')}.pkl",
                                       "application/octet-stream")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 9 · EXPORT REPORT
# ═══════════════════════════════════════════════════════════════
if st.session_state.clean_df is not None:
    st.markdown('<div class="forge-card"><h3>📋 Step 9 · Export Compliance Report</h3>', unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b'>Generate a professional PDF covering the full pipeline — "
                "quality score, cleaning log, PII scan, compliance scores, and ML results.</small>",
                unsafe_allow_html=True)

    can_gen = st.session_state.pii_df is not None and st.session_state.insights is not None
    if not can_gen:
        st.info("💡 Complete Step 4 (Privacy Shield scan) to unlock the full report.")

    if st.button("📄 Generate PDF Report", type="primary", disabled=not can_gen):
        with st.spinner("Generating report…"):
            pdf_bytes = generate_report(
                dataset_name      = st.session_state.dataset_name or "dataset.csv",
                raw_shape         = st.session_state.raw_df.shape,
                clean_shape       = st.session_state.clean_df.shape,
                cleaning_report   = st.session_state.cleaning_report or [],
                quality_score     = st.session_state.quality_score or 0,
                insights          = st.session_state.insights or [],
                pii_df            = st.session_state.pii_df,
                compliance_scores = st.session_state.compliance_scores or {},
                ml_results        = st.session_state.ml_results,
            )
        st.success("✅ Report ready!")
        st.download_button("⬇️ Download PDF Report", pdf_bytes,
                           "dataforge_report.pdf","application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='forge-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
    <span style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                 background:linear-gradient(135deg,#00d4aa,#818cf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🔧 DataForge
    </span>
    <span style='color:#334155;font-size:0.8rem;margin-left:1rem;'>
        Open Source · Free to use · Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
