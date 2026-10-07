
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Renal Health Intelligence Dashboard", layout="wide", page_icon="🩺")

# ---------- Styling ----------
st.markdown('''
<style>
.main-header {
    background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.2rem;
}
.main-header h1 { margin: 0; font-size: 1.9rem; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.92; }

.metric-card {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}
.metric-card h2 { margin: 0; font-size: 1.8rem; }
.metric-card p { margin: 0.2rem 0 0 0; font-size: 0.85rem; opacity: 0.9; }

.c-blue   { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
.c-red    { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
.c-green  { background: linear-gradient(135deg, #11998e, #38ef7d); }
.c-purple { background: linear-gradient(135deg, #8e2de2, #4a00e0); }
.c-orange { background: linear-gradient(135deg, #f7971e, #ffd200); }

.section-title {
    border-left: 5px solid #4e54c8;
    padding-left: 0.6rem;
    margin: 0.8rem 0 0.6rem 0;
}
</style>
''', unsafe_allow_html=True)


def colored_table(dframe, color_col, scale_name="Greens"):
    # Plotly-native colored table — no matplotlib / pandas Styler involved,
    # so it can't fail with the "requires matplotlib" error.
    values = dframe[color_col].astype(float).to_numpy()
    vmin, vmax = values.min(), values.max()
    norm = np.full_like(values, 0.5) if vmax == vmin else (values - vmin) / (vmax - vmin)
    row_colors = px.colors.sample_colorscale(scale_name, norm.tolist())

    cell_fill = []
    for col in dframe.columns:
        cell_fill.append(row_colors if col == color_col else ["white"] * len(dframe))

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dframe.columns), fill_color="#4e54c8",
                    font=dict(color="white", size=13), align="left"),
        cells=dict(values=[dframe[c] for c in dframe.columns],
                   fill_color=cell_fill, align="left", height=28)
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=46 + 30 * len(dframe))
    return fig


def metric_card(col, label, value, css_class):
    col.markdown(f'''
    <div class="metric-card {css_class}">
        <h2>{value}</h2>
        <p>{label}</p>
    </div>
    ''', unsafe_allow_html=True)


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


df = load_csv("ckd_processed.csv")
if df is None:
    st.error("ckd_processed.csv not found. Run the notebook's export cell first.")
    st.stop()

df["Classification"] = df["classification"].map({0: "Not CKD", 1: "CKD"})

corr_df = load_csv("correlation_matrix.csv")
reg_results = load_csv("regression_results.csv")
reg_metrics = load_csv("regression_metrics.csv")
clf_metrics = load_csv("classification_metrics.csv")
roc_df = load_csv("roc_data.csv")
feat_imp = load_csv("feature_importance.csv")
sentiment_df = load_csv("text_sentiment.csv")
top_words_df = load_csv("text_top_words.csv")

st.markdown('''
<div class="main-header">
<h1>🩺 Renal Health Intelligence System</h1>
<p>Chronic Kidney Disease monitoring, risk scoring, and patient-feedback analytics</p>
</div>
''', unsafe_allow_html=True)

tabs = st.tabs([
    "📊 Overview", "🔍 EDA", "📈 Regression", "🧩 Clustering",
    "🤖 Classification", "💬 Text Analytics", "🧑‍⚕️ Patient Drill-down"
])

# ---------------- Overview ----------------
with tabs[0]:
    st.markdown('<h3 class="section-title">Key Numbers</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Patients", len(df), "c-blue")
    metric_card(c2, "CKD Cases", int(df["classification"].sum()), "c-red")
    metric_card(c3, "Non-CKD Cases", int((df["classification"] == 0).sum()), "c-green")
    metric_card(c4, "Average eGFR", round(df["eGFR"].mean(), 1), "c-purple")

    st.markdown('<h3 class="section-title">Distributions</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="Classification", title="CKD vs Not CKD",
                     color="Classification",
                     color_discrete_map={"CKD": "#ef473a", "Not CKD": "#38ef7d"}, hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        stage_counts = df["CKD_Stage"].value_counts().reset_index()
        stage_counts.columns = ["CKD_Stage", "Count"]
        fig = px.bar(stage_counts, x="CKD_Stage", y="Count", color="CKD_Stage",
                     title="Patients by CKD Stage", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- EDA ----------------
with tabs[1]:
    st.markdown('<h3 class="section-title">Exploratory Data Analysis</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Classification", y="sc", color="Classification",
                     title="Serum Creatinine by Class",
                     color_discrete_map={"CKD": "#ef473a", "Not CKD": "#38ef7d"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="age", color="Classification", barmode="overlay",
                            title="Age Distribution by Class", nbins=25,
                            color_discrete_map={"CKD": "#ef473a", "Not CKD": "#38ef7d"})
        st.plotly_chart(fig, use_container_width=True)

    if corr_df is not None:
        corr_df = corr_df.set_index(corr_df.columns[0])
        fig = px.imshow(corr_df, color_continuous_scale="RdBu_r", title="Correlation Heatmap", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Regression ----------------
with tabs[2]:
    st.markdown('<h3 class="section-title">Predicting eGFR (Kidney Function)</h3>', unsafe_allow_html=True)
    if reg_metrics is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(colored_table(reg_metrics, "R2", "Greens"), use_container_width=True)
        with c2:
            fig = px.bar(reg_metrics, x="Model", y="R2", color="Model", title="R² Score by Model",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

    if reg_results is not None:
        model_choice = st.radio("Show predictions from:", ["Random Forest", "Linear Regression"], horizontal=True)
        col = "Predicted_eGFR_RandomForest" if model_choice == "Random Forest" else "Predicted_eGFR_LinearRegression"
        fig = px.scatter(reg_results, x="Actual_eGFR", y=col, opacity=0.65,
                          title=f"Actual vs Predicted eGFR — {model_choice}",
                          color_discrete_sequence=["#4e54c8"])
        max_v = float(reg_results["Actual_eGFR"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Clustering ----------------
with tabs[3]:
    st.markdown('<h3 class="section-title">Patient Risk Clusters</h3>', unsafe_allow_html=True)
    if "PC1" in df.columns and "PC2" in df.columns:
        fig = px.scatter(df, x="PC1", y="PC2", color=df["Risk_Cluster"].astype(str),
                          title="Patient Risk Clusters (PCA Projection)",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          labels={"color": "Risk Cluster"})
        st.plotly_chart(fig, use_container_width=True)

    cluster_profile = df.groupby("Risk_Cluster")[["sc", "hemo", "bp", "eGFR"]].mean().round(2).reset_index()
    st.markdown("**Cluster profiles (average clinical values)**")
    st.plotly_chart(colored_table(cluster_profile, "eGFR", "Purples"), use_container_width=True)

# ---------------- Classification ----------------
with tabs[4]:
    st.markdown('<h3 class="section-title">CKD Detection — Model Comparison</h3>', unsafe_allow_html=True)
    if clf_metrics is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(colored_table(clf_metrics, "F1", "Blues"), use_container_width=True)
        with c2:
            melted = clf_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                         title="Model Performance Comparison", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

    if roc_df is not None:
        fig = px.line(roc_df, x="FPR", y="TPR", color="Model", title="ROC Curves",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    if feat_imp is not None:
        fig = px.bar(feat_imp.head(10).sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Top Feature Importances (Random Forest)",
                     color="Importance", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Text Analytics ----------------
with tabs[5]:
    st.markdown('<h3 class="section-title">Patient Review Sentiment</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if sentiment_df is not None:
            fig = px.bar(sentiment_df, x="Rating_Bin", y="Avg_Sentiment", color="Avg_Sentiment",
                         title="Average Sentiment by Rating Band", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if top_words_df is not None:
            fig = px.bar(top_words_df.sort_values("Count"), x="Count", y="Word", orientation="h",
                         title="Top Keywords in Kidney-Related Reviews",
                         color="Count", color_continuous_scale="Tealgrn")
            st.plotly_chart(fig, use_container_width=True)

    if os.path.exists("wordcloud_kidney.png"):
        st.markdown("**Word Cloud**")
        st.image("wordcloud_kidney.png", use_container_width=True)

# ---------------- Patient Drill-down ----------------
with tabs[6]:
    st.markdown('<h3 class="section-title">Individual Patient Record</h3>', unsafe_allow_html=True)
    idx = st.number_input("Patient row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    patient = df.iloc[int(idx)]

    risk_color = "c-red" if patient["Classification"] == "CKD" else "c-green"
    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Age", int(patient["age"]), "c-blue")
    metric_card(c2, "eGFR", round(patient["eGFR"], 1), "c-purple")
    metric_card(c3, "Status", patient["Classification"], risk_color)

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Serum Creatinine", round(patient["sc"], 2), "c-orange")
    metric_card(c2, "Hemoglobin", round(patient["hemo"], 2), "c-blue")
    metric_card(c3, "CKD Stage", patient["CKD_Stage"], "c-purple")

    st.markdown("**Full Patient Record**")
    st.dataframe(patient.to_frame(name="Value"), use_container_width=True)
