"""
ChurnGuard AI — Customer Churn Prediction
Modern dark glassmorphism Streamlit UI with SVM inference and interactive charts.
"""
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark glassmorphism design system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ───────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0a18 0%, #110820 40%, #0a1428 80%, #0a0a18 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ── Streamlit UI chrome controls ───────────────────── */
#MainMenu { visibility: hidden !important; display: none !important; }
footer { visibility: hidden !important; display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Keep header and toolbar visible and transparent so the top-left (>) button is visible */
header[data-testid="stHeader"], header {
    background: transparent !important;
    visibility: visible !important;
    display: block !important;
    height: 3.5rem !important;
    z-index: 99999 !important;
}

[data-testid="stToolbar"] {
    visibility: visible !important;
    display: flex !important;
    background: transparent !important;
}

/* Hide right-side toolbar buttons (deploy, menu actions) */
[data-testid="stToolbarActions"],
[data-testid="stToolbarActionButton"],
[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    display: none !important;
}

/* ── The top-left (>) sidebar button (Streamlit 1.30+ & legacy) ──────── */
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: inline-flex !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 1000000 !important;
    color: #c4b5fd !important;
    background: rgba(124, 58, 237, 0.35) !important;
    border: 1px solid rgba(124, 58, 237, 0.7) !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    margin: 8px 0 0 12px !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.5) !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

[data-testid="stExpandSidebarButton"]:hover,
[data-testid="collapsedControl"]:hover {
    background: rgba(124, 58, 237, 0.65) !important;
    border-color: #a78bfa !important;
    box-shadow: 0 0 22px rgba(124, 58, 237, 0.8) !important;
    transform: scale(1.08);
}

[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] span,
[data-testid="collapsedControl"] svg {
    color: #c4b5fd !important;
    fill: #c4b5fd !important;
}

/* ── Sidebar collapse button (inside sidebar) ───────── */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: inline-flex !important;
    color: #c4b5fd !important;
    background: rgba(124, 58, 237, 0.2) !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(124, 58, 237, 0.45) !important;
    color: #ffffff !important;
    box-shadow: 0 0 14px rgba(124, 58, 237, 0.5) !important;
}

.main .block-container {
    padding: 1rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0a20 0%, #090914 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.3) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.2rem 1.1rem; }

/* ── Hero Command Card ──────────────────────────────── */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 20, 65, 0.55) 0%, rgba(15, 23, 42, 0.65) 50%, rgba(15, 10, 35, 0.75) 100%) !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    border-radius: 20px !important;
    padding: 2.2rem 2.4rem 1.9rem 2.4rem !important;
    margin: 0.4rem 0 1.6rem 0 !important;
    box-shadow: 0 10px 35px -5px rgba(124, 58, 237, 0.22), 0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(14px) !important;
}
.hero-glow-orb {
    position: absolute !important;
    top: -70px !important;
    right: -70px !important;
    width: 250px !important;
    height: 250px !important;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.35) 0%, rgba(59, 130, 246, 0.15) 50%, transparent 70%) !important;
    border-radius: 50% !important;
    filter: blur(40px) !important;
    pointer-events: none !important;
}
.hero-top-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
    margin-bottom: 1.1rem !important;
}
.hero-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    background: rgba(124, 58, 237, 0.18) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: #c4b5fd !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
.hero-pulse-dot {
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    background: #10B981 !important;
    box-shadow: 0 0 10px #10B981 !important;
}
.hero-helper-chip {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    font-size: 0.78rem !important;
    color: #cbd5e1 !important;
    background: rgba(124, 58, 237, 0.12) !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
}
.hero-title {
    margin: 0 0 0.65rem 0 !important;
    font-size: 2.45rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.18 !important;
    background: linear-gradient(135deg, #ffffff 0%, #ddd6fe 30%, #a78bfa 70%, #60a5fa 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    filter: drop-shadow(0 2px 14px rgba(124, 58, 237, 0.25)) !important;
}
.hero-subtext {
    margin: 0 0 1.5rem 0 !important;
    color: #94a3b8 !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    max-width: 980px !important;
    font-weight: 400 !important;
}
.hero-metrics-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
    gap: 0.85rem !important;
    padding-top: 1.1rem !important;
    border-top: 1px solid rgba(139, 92, 246, 0.2) !important;
}
.hero-stat-card {
    background: rgba(124, 58, 237, 0.1) !important;
    border: 1px solid rgba(124, 58, 237, 0.26) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}
.hero-stat-val {
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    line-height: 1 !important;
}
.hero-stat-label {
    font-size: 0.7rem !important;
    color: #a5b4fc !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    margin-top: 4px !important;
    letter-spacing: 0.04em !important;
}

/* ── Form inputs — selectbox ────────────────────────── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.055) !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(124, 58, 237, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}
div[data-baseweb="select"] svg { color: #7C3AED !important; }

/* ── Dropdown popover ───────────────────────────────── */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: #160f28 !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 10px !important;
}

/* ── Number input ───────────────────────────────────── */
input[type="number"] {
    background: rgba(255,255,255,0.055) !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
input[type="number"]:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
    outline: none !important;
}

/* ── Form labels ────────────────────────────────────── */
label {
    color: #a5b4fc !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Submit button ──────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.5) !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.7) !important;
}

/* ── Divider ────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(124, 58, 237, 0.18) !important;
    margin: 1.2rem 0 !important;
}

/* ── Metric cards ───────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(124, 58, 237, 0.09) !important;
    border: 1px solid rgba(124, 58, 237, 0.24) !important;
    border-radius: 12px !important;
    padding: 0.65rem 0.85rem !important;
}
[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-radius: 14px !important;
    margin-top: 1rem !important;
}
[data-testid="stExpander"] summary { color: #a5b4fc !important; font-weight: 600 !important; }

/* ── Dataframe ──────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Headings ───────────────────────────────────────── */
h1, h2, h3, h4 { color: #e2e8f0 !important; }

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.45); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model + scaler  (same objects saved from the notebook, untouched)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("final_svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "❌ Could not find `final_svm_model.pkl` and/or `scaler.pkl`. "
        "Place both files in the same folder as `app.py` before running."
    )
    st.stop()

# Exact column order the scaler/model were fit on (pulled from scaler, not invented)
FEATURE_ORDER = list(scaler.feature_names_in_)

# ─────────────────────────────────────────────────────────────────────────────
# Feature-vector builder
# Reproduces: drop customerID → get_dummies(drop_first=True) → scaler
# with the SAME reference categories the training data had.
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_ORDER}

    # Numeric features
    row["SeniorCitizen"]  = int(inputs["SeniorCitizen"])
    row["tenure"]         = float(inputs["tenure"])
    row["MonthlyCharges"] = float(inputs["MonthlyCharges"])
    row["TotalCharges"]   = float(inputs["TotalCharges"])

    # Binary Yes/No  (reference category = "No" or "Female")
    if inputs["gender"]          == "Male": row["gender_Male"]          = 1
    if inputs["Partner"]         == "Yes":  row["Partner_Yes"]          = 1
    if inputs["Dependents"]      == "Yes":  row["Dependents_Yes"]       = 1
    if inputs["PhoneService"]    == "Yes":  row["PhoneService_Yes"]     = 1
    if inputs["PaperlessBilling"]== "Yes":  row["PaperlessBilling_Yes"] = 1

    # MultipleLines  (reference = "No")
    if inputs["MultipleLines"] == "No phone service":
        row["MultipleLines_No phone service"] = 1
    elif inputs["MultipleLines"] == "Yes":
        row["MultipleLines_Yes"] = 1

    # InternetService  (reference = "DSL")
    if inputs["InternetService"] == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif inputs["InternetService"] == "No":
        row["InternetService_No"] = 1

    # Internet add-ons  (reference = "No")
    for feat in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]:
        val = inputs[feat]
        if val == "No internet service":
            row[f"{feat}_No internet service"] = 1
        elif val == "Yes":
            row[f"{feat}_Yes"] = 1

    # Contract  (reference = "Month-to-month")
    if inputs["Contract"] == "One year":  row["Contract_One year"]  = 1
    elif inputs["Contract"] == "Two year": row["Contract_Two year"] = 1

    # PaymentMethod  (reference = "Bank transfer (automatic)")
    pm = inputs["PaymentMethod"]
    if pm == "Credit card (automatic)":  row["PaymentMethod_Credit card (automatic)"] = 1
    elif pm == "Electronic check":        row["PaymentMethod_Electronic check"]        = 1
    elif pm == "Mailed check":            row["PaymentMethod_Mailed check"]            = 1

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly layout tokens
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
)

def sigmoid_risk(score: float) -> float:
    """Map SVM decision_function → 0–100 churn-risk % via sigmoid."""
    return round((1 / (1 + math.exp(-score))) * 100, 1)


def fig_gauge(risk_pct: float, prediction: str) -> go.Figure:
    """Speedometer gauge showing churn risk score."""
    bar_color = "#EF4444" if prediction == "Yes" else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "CHURN RISK SCORE",
               "font": {"size": 12, "color": "#64748b"}},
        number={"suffix": "%",
                "font": {"size": 44, "color": bar_color, "family": "Inter"},
                "valueformat": ".1f"},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#1e293b",
                "tickfont": {"size": 10, "color": "#475569"},
                "nticks": 6,
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.13)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.13)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.13)"},
            ],
            "threshold": {
                "line": {"color": "#F59E0B", "width": 2},
                "thickness": 0.72,
                "value": 50,
            },
        },
    ))
    fig.update_layout(**_LAYOUT, height=290, margin=dict(l=20, r=20, t=48, b=20))
    return fig


def fig_feature_importance(model, feature_names: list, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-N SVM linear coefficients."""
    coefs = model.coef_[0]
    df = (pd.DataFrame({"feature": feature_names, "coef": coefs})
            .assign(abs_coef=lambda x: x["coef"].abs())
            .nlargest(top_n, "abs_coef")
            .sort_values("coef", ascending=True))

    colors = ["#EF4444" if c > 0 else "#10B981" for c in df["coef"]]

    fig = go.Figure(go.Bar(
        x=df["coef"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text="🔍  Top Feature Importance — SVM Linear Coefficients",
            font=dict(size=15, color="#e2e8f0"), x=0,
        ),
        xaxis=dict(
            title="Coefficient Value",
            titlefont=dict(size=11, color="#64748b"),
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#cbd5e1"),
            gridcolor="rgba(255,255,255,0.03)",
        ),
        height=470,
        margin=dict(l=20, r=20, t=48, b=20),
        showlegend=False,
        bargap=0.38,
    )
    return fig


def fig_churn_donut() -> go.Figure:
    """Small donut chart showing dataset churn distribution."""
    fig = go.Figure(go.Pie(
        labels=["No Churn", "Churn"],
        values=[73.4, 26.6],
        hole=0.68,
        marker=dict(
            colors=["#10B981", "#EF4444"],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b>: %{value}%<extra></extra>",
        sort=False,
    ))
    fig.add_annotation(
        text="26.6%", x=0.5, y=0.58,
        showarrow=False, font=dict(size=17, color="#EF4444", family="Inter"),
        align="center",
    )
    fig.add_annotation(
        text="churn rate", x=0.5, y=0.42,
        showarrow=False, font=dict(size=10, color="#64748b", family="Inter"),
        align="center",
    )
    fig.update_layout(
        **_LAYOUT,
        height=190,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.08,
            font=dict(size=11, color="#94a3b8"),
        ),
    )
    return fig


def fig_customer_profile(inputs: dict) -> go.Figure:
    """
    Radar (spider) chart comparing this customer's key numeric features
    against IBM Telco dataset averages for churned vs. retained customers.
    """
    # Approximate dataset averages from IBM Telco EDA
    AVG_CHURN    = {"tenure": 18.0, "MonthlyCharges": 74.4, "TotalCharges": 1435.0}
    AVG_RETAINED = {"tenure": 37.6, "MonthlyCharges": 61.3, "TotalCharges": 2556.0}
    MAX_VALS     = {"tenure": 72,   "MonthlyCharges": 120,  "TotalCharges": 9000}

    cats   = ["Tenure<br>(months)", "Monthly<br>Charges", "Total<br>Charges"]
    keys   = ["tenure", "MonthlyCharges", "TotalCharges"]

    def norm(d: dict) -> list:
        return [min(d[k] / MAX_VALS[k] * 100, 100) for k in keys]

    cust_vals   = norm(inputs)
    churn_vals  = norm(AVG_CHURN)
    retain_vals = norm(AVG_RETAINED)

    # Close the polygon by repeating the first point
    theta = cats + [cats[0]]
    def close(lst): return lst + [lst[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=close(churn_vals), theta=theta,
        fill="toself", name="Avg Churner",
        line=dict(color="#EF4444", width=2),
        fillcolor="rgba(239,68,68,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(retain_vals), theta=theta,
        fill="toself", name="Avg Retained",
        line=dict(color="#10B981", width=2),
        fillcolor="rgba(16,185,129,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(cust_vals), theta=theta,
        fill="toself", name="This Customer",
        line=dict(color="#818CF8", width=2.5, dash="dot"),
        fillcolor="rgba(124,58,237,0.18)",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text="📡  Customer Profile vs. Dataset Averages",
            font=dict(size=15, color="#e2e8f0"), x=0,
        ),
        polar=dict(
            bgcolor="rgba(255,255,255,0.025)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(255,255,255,0.07)",
                tickcolor="#334155",
                tickfont=dict(size=9, color="#475569"),
                linecolor="rgba(255,255,255,0.07)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
                tickfont=dict(size=11, color="#94a3b8"),
            ),
        ),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.12,
            font=dict(size=11, color="#94a3b8"),
        ),
        height=360,
        margin=dict(l=20, r=20, t=48, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.8rem;">
        <div style="font-size:2.6rem;margin-bottom:6px;">🛡️</div>
        <div style="font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,#a78bfa,#60a5fa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ChurnGuard AI
        </div>
        <div style="font-size:0.68rem;color:#334155;margin-top:5px;
                    letter-spacing:0.1em;text-transform:uppercase;">
            Customer Retention Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model performance metrics
    st.markdown(
        "<p style='font-size:0.75rem;color:#7C3AED;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;margin:0 0 8px;'>📊 Model Performance</p>",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy",  "80%")
        st.metric("Precision", "64%")
    with m2:
        st.metric("F1-Score", "0.58")
        st.metric("Recall",   "54%")

    st.divider()

    # Dataset churn donut
    st.markdown(
        "<p style='font-size:0.75rem;color:#7C3AED;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;margin:0 0 4px;'>🎯 Dataset Overview</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fig_churn_donut(),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown(
        "<div style='text-align:center;font-size:0.72rem;color:#334155;margin-top:-6px;'>"
        "IBM Telco · 7,032 customers</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Tech stack footer
    st.markdown(
        "<div style='font-size:0.7rem;color:#1e293b;line-height:2;'>"
        "⚙️ <span style='color:#475569;'>SVM · linear kernel · C=1</span><br>"
        "📐 <span style='color:#475569;'>30 one-hot encoded features</span><br>"
        "🔧 <span style='color:#475569;'>StandardScaler preprocessing</span><br>"
        "📦 <span style='color:#475569;'>scikit-learn 1.9.0</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main — Hero command center & retention overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
<div class="hero-glow-orb"></div>
<div class="hero-top-row">
<div class="hero-badge"><span class="hero-pulse-dot"></span> 🛡️ ChurnGuard AI · Enterprise Intelligence</div>
<div class="hero-helper-chip"><span style="color:#a78bfa;font-weight:900;">👈 &gt;</span> Check top-left <strong style="color:#c4b5fd;">(&gt;)</strong> button to view Model &amp; Dataset Analytics</div>
</div>
<h1 class="hero-title">Customer Churn Prediction &amp; Retention Analytics</h1>
<p class="hero-subtext">Evaluate customer retention risks with machine learning. Configure demographics, subscription services, and billing contracts below, then run AI inference to generate real-time churn probabilities, risk speedometer gauges, radar benchmarks, and key predictive factors.</p>
<div class="hero-metrics-grid">
<div class="hero-stat-card">
<div style="font-size:1.55rem;">🎯</div>
<div><div class="hero-stat-val">80% Accuracy</div><div class="hero-stat-label">Model Test Score</div></div>
</div>
<div class="hero-stat-card" style="background:rgba(59,130,246,0.08)!important;border-color:rgba(59,130,246,0.24)!important;">
<div style="font-size:1.55rem;">⚙️</div>
<div><div class="hero-stat-val">Linear SVM</div><div class="hero-stat-label" style="color:#93c5fd!important;">C=1 Balanced Classifier</div></div>
</div>
<div class="hero-stat-card" style="background:rgba(16,185,129,0.08)!important;border-color:rgba(16,185,129,0.24)!important;">
<div style="font-size:1.55rem;">📊</div>
<div><div class="hero-stat-val">30 Features</div><div class="hero-stat-label" style="color:#6ee7b7!important;">Scaled &amp; One-Hot Encoded</div></div>
</div>
<div class="hero-stat-card" style="background:rgba(245,158,11,0.08)!important;border-color:rgba(245,158,11,0.24)!important;">
<div style="font-size:1.55rem;">🔍</div>
<div><div class="hero-stat-val">Explainable AI</div><div class="hero-stat-label" style="color:#fcd34d!important;">Real-Time Charts</div></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────────────────────────────────────
with st.form("churn_form"):

    # ── Demographics ─────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;color:#a5b4fc;"
        "letter-spacing:0.04em;margin-bottom:12px;'>👤 Demographics</h3>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: gender     = st.selectbox("Gender",         ["Female", "Male"])
    with c2: senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3: partner    = st.selectbox("Has Partner",    ["No", "Yes"])
    with c4: dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.divider()

    # ── Phone & Internet ──────────────────────────────
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;color:#a5b4fc;"
        "letter-spacing:0.04em;margin-bottom:12px;'>📞 Phone & Internet Services</h3>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
            st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
    with c2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.divider()

    # ── Add-on Services ───────────────────────────────
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;color:#a5b4fc;"
        "letter-spacing:0.04em;margin-bottom:12px;'>🛡️ Add-on Services</h3>",
        unsafe_allow_html=True,
    )
    has_internet = internet_service != "No"
    addon_cols   = st.columns(3)
    addon_labels = ["Online Security", "Online Backup", "Device Protection",
                    "Tech Support", "Streaming TV", "Streaming Movies"]
    addon_keys   = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    addon_values = {}
    for i, (label, key) in enumerate(zip(addon_labels, addon_keys)):
        with addon_cols[i % 3]:
            if has_internet:
                addon_values[key] = st.selectbox(label, ["No", "Yes"], key=key)
            else:
                st.selectbox(label, ["No internet service"], disabled=True, key=key)
                addon_values[key] = "No internet service"

    st.divider()

    # ── Contract & Billing ────────────────────────────
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;color:#a5b4fc;"
        "letter-spacing:0.04em;margin-bottom:12px;'>📄 Contract & Billing</h3>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with c2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    st.divider()

    # ── Charges & Tenure ──────────────────────────────
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;color:#a5b4fc;"
        "letter-spacing:0.04em;margin-bottom:12px;'>💰 Charges & Tenure</h3>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12, step=1
        )
    with c2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5
        )
    with c3:
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0, step=1.0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮   Analyze Customer Risk", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction & results
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    # Build feature vector and run inference
    inputs = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   addon_values["OnlineSecurity"],
        "OnlineBackup":     addon_values["OnlineBackup"],
        "DeviceProtection": addon_values["DeviceProtection"],
        "TechSupport":      addon_values["TechSupport"],
        "StreamingTV":      addon_values["StreamingTV"],
        "StreamingMovies":  addon_values["StreamingMovies"],
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    X_row          = build_feature_row(inputs)
    X_scaled       = scaler.transform(X_row)
    prediction     = model.predict(X_scaled)[0]
    decision_score = model.decision_function(X_scaled)[0]
    risk_pct       = sigmoid_risk(decision_score)

    st.divider()

    # ── Prediction result banner ──────────────────────
    if prediction == "Yes":
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,rgba(239,68,68,0.18),rgba(239,68,68,0.06));
            border: 1px solid rgba(239,68,68,0.45);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.2rem;
        ">
            <div style="font-size:2.8rem;flex-shrink:0;">⚠️</div>
            <div style="flex:1;">
                <div style="font-size:1.35rem;font-weight:800;color:#FCA5A5;
                            letter-spacing:0.02em;">HIGH CHURN RISK</div>
                <div style="font-size:0.85rem;color:#64748b;margin-top:5px;">
                    This customer is likely to leave. Immediate retention action recommended.
                </div>
            </div>
            <div style="text-align:right;flex-shrink:0;">
                <div style="font-size:2.4rem;font-weight:900;color:#EF4444;
                            line-height:1;">{risk_pct}%</div>
                <div style="font-size:0.68rem;color:#64748b;letter-spacing:0.06em;
                            text-transform:uppercase;">risk score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,rgba(16,185,129,0.18),rgba(16,185,129,0.06));
            border: 1px solid rgba(16,185,129,0.45);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.2rem;
        ">
            <div style="font-size:2.8rem;flex-shrink:0;">✅</div>
            <div style="flex:1;">
                <div style="font-size:1.35rem;font-weight:800;color:#6EE7B7;
                            letter-spacing:0.02em;">LOW CHURN RISK</div>
                <div style="font-size:0.85rem;color:#64748b;margin-top:5px;">
                    This customer is likely to stay. Continue delivering excellent service.
                </div>
            </div>
            <div style="text-align:right;flex-shrink:0;">
                <div style="font-size:2.4rem;font-weight:900;color:#10B981;
                            line-height:1;">{risk_pct}%</div>
                <div style="font-size:0.68rem;color:#64748b;letter-spacing:0.06em;
                            text-transform:uppercase;">risk score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 1: Gauge chart + Radar profile ───────────
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.plotly_chart(
            fig_gauge(risk_pct, prediction),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_right:
        st.plotly_chart(
            fig_customer_profile(inputs),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Row 2: Feature importance (full width) ────────
    st.plotly_chart(
        fig_feature_importance(model, FEATURE_ORDER),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown(
        "<p style='font-size:0.78rem;color:#334155;text-align:center;margin-top:-8px;'>"
        "🔴 <span style='color:#475569;'>Red bars push toward <em>churn</em></span>"
        " &nbsp;·&nbsp; "
        "🟢 <span style='color:#475569;'>Green bars push toward <em>retention</em></span>"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Expandable: raw encoded vector ───────────────
    with st.expander("🔬  Inspect the encoded feature vector sent to the model"):
        st.dataframe(X_row.T.rename(columns={0: "value"}), use_container_width=True)
        st.caption(
            f"Decision-function distance from hyperplane: `{decision_score:.4f}` "
            "(positive = predicts Churn, negative = predicts No Churn). "
            "This SVM was trained with `probability=False`, so `decision_function()` "
            "is used as a rough confidence proxy, not a calibrated probability."
        )
