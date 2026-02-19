"""
app.py — Bank Customer Churn Predictor
Streamlit web application for real-time churn predictions

Run with:
    streamlit run app.py

Requirements (install with pip):
    pip install streamlit 
    pip install pandas numpy scikit-learn matplotlib seaborn plotly

Note: Probably You already have installed in your environment (pandas numpy scikit-learn matplotlib seaborn plotly):
    - imbalanced-learn (for SMOTE - used in training, not prediction)
    - joblib or pickle (for loading model - built into Python)
    
The app will load:
    - model_file.pkl  (trained Random Forest model)
    - Scaler_file.pkl (fitted CustomScaler for preprocessing)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark banking aesthetic with gold accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0D1117;
    color: #E8E8E8;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #C9D1D9 !important;
}

/* ── Header ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #F0C040;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.0rem;
    color: #8B949E;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── KPI Cards ── */
.kpi-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    flex: 1;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.gold::before  { background: #F0C040; }
.kpi-card.red::before   { background: #F85149; }
.kpi-card.green::before { background: #3FB950; }
.kpi-card.blue::before  { background: #58A6FF; }

.kpi-label {
    font-size: 0.72rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #F0C040;
    line-height: 1;
}
.kpi-value.red   { color: #F85149; }
.kpi-value.green { color: #3FB950; }
.kpi-value.blue  { color: #58A6FF; }

/* ── Section headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #C9D1D9;
    border-bottom: 1px solid #21262D;
    padding-bottom: 10px;
    margin-bottom: 18px;
    margin-top: 10px;
}

/* ── Prediction badge ── */
.badge-churn {
    display: inline-block;
    background: rgba(248,81,73,0.15);
    border: 1px solid #F85149;
    color: #F85149;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.badge-stay {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid #3FB950;
    color: #3FB950;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: #161B22 !important;
    border-color: #30363D !important;
    color: #E8E8E8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #F0C040;
    color: #0D1117;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    padding: 0.6rem 2rem;
    letter-spacing: 0.03em;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #E0B030;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(240,192,64,0.3);
}

/* ── DataFrames ── */
.dataframe { background: #161B22 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-bottom: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    color: #8B949E;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    color: #F0C040 !important;
    border-bottom: 2px solid #F0C040 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161B22 !important;
    color: #C9D1D9 !important;
    border-radius: 8px !important;
}

/* ── Info / Success / Warning boxes ── */
.stAlert {
    border-radius: 8px;
}

/* ── Divider ── */
hr { border-color: #21262D; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CustomScaler (must match training definition for pickle compatibility)
# ─────────────────────────────────────────────────────────────────────────────
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler    = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns   = columns
        self.with_mean = with_mean
        self.with_std  = with_std
        self.copy      = copy
        self.mean_     = None
        self.std_      = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.std_  = np.std(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled    = pd.DataFrame(
            self.scaler.transform(X[self.columns]),
            columns=self.columns,
            index=X.index
        )
        X_notscaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_notscaled, X_scaled], axis=1)[init_col_order]


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'HasCrCard', 'IsActiveMember', 'CreditScore', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score',
    'Point Earned', 'Geography_Germany', 'Geography_Spain', 'Gender_Male',
    'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER'
]
NUMERICAL_COLS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'EstimatedSalary', 'Satisfaction Score', 'Point Earned'
]
COLS_TO_DROP = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
CATEGORICAL_COLS = ['Geography', 'Gender', 'Card Type']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts(model_bytes, scaler_bytes):
    model  = pickle.loads(model_bytes)
    scaler = pickle.loads(scaler_bytes)
    return model, scaler


def preprocess(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply same preprocessing as training pipeline."""
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors='ignore')
    if 'Exited' in df.columns:
        df = df.drop(columns=['Exited'])
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype='int')
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def predict_dataframe(df_raw: pd.DataFrame, model, scaler):
    """Preprocess raw df and return predictions."""
    df_processed = preprocess(df_raw.copy(), scaler)
    preds = model.predict(df_processed)
    probs = model.predict_proba(df_processed)[:, 1]
    result = df_raw.copy().reset_index(drop=True)
    result['Churn Probability'] = (probs * 100).round(1)
    result['Predicted'] = ['🔴 Churn' if p == 1 else '🟢 Stay' for p in preds]
    result['Risk Level'] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    return result


def build_single_customer(inputs: dict):
    """Build a one-row DataFrame from the sidebar inputs."""
    return pd.DataFrame([{
        'CreditScore'       : inputs['credit_score'],
        'Geography'         : inputs['geography'],
        'Gender'            : inputs['gender'],
        'Age'               : inputs['age'],
        'Tenure'            : inputs['tenure'],
        'Balance'           : inputs['balance'],
        'NumOfProducts'     : inputs['num_products'],
        'HasCrCard'         : int(inputs['has_cr_card']),
        'IsActiveMember'    : int(inputs['is_active']),
        'EstimatedSalary'   : inputs['salary'],
        'Satisfaction Score': inputs['satisfaction'],
        'Card Type'         : inputs['card_type'],
        'Point Earned'      : inputs['points'],
    }])


def plot_gauge(probability: float):
    """Plotly gauge chart for single customer churn probability."""
    color = '#3FB950' if probability < 0.4 else '#F0C040' if probability < 0.65 else '#F85149'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 42, 'color': color, 'family': 'DM Serif Display'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B949E', 'tickfont': {'color': '#8B949E'}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#161B22',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],   'color': 'rgba(63,185,80,0.12)'},
                {'range': [40, 65],  'color': 'rgba(240,192,64,0.12)'},
                {'range': [65, 100], 'color': 'rgba(248,81,73,0.12)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': probability * 100
            }
        },
        title={'text': "Churn Probability", 'font': {'size': 13, 'color': '#8B949E', 'family': 'DM Sans'}}
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D1D9',
    )
    return fig


def plot_feature_importance(model):
    """Horizontal bar chart of top-10 feature importances."""
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    top10 = importances.nlargest(10).sort_values()

    fig = go.Figure(go.Bar(
        x=top10.values,
        y=top10.index,
        orientation='h',
        marker=dict(
            color=top10.values,
            colorscale=[[0, '#21262D'], [1, '#F0C040']],
            showscale=False
        ),
        text=[f'{v:.3f}' for v in top10.values],
        textposition='outside',
        textfont=dict(color='#8B949E', size=11),
    ))
    fig.update_layout(
        height=360,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(color='#C9D1D9', tickfont=dict(size=12)),
        font=dict(family='DM Sans', color='#C9D1D9'),
    )
    return fig


def plot_risk_distribution(result_df):
    """Donut chart of risk levels."""
    risk_counts = result_df['Risk Level'].value_counts()
    colors_map  = {'Low': '#3FB950', 'Medium': '#F0C040', 'High': '#F85149'}
    colors      = [colors_map.get(k, '#8B949E') for k in risk_counts.index]

    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker_colors=colors,
        textfont=dict(size=12, family='DM Sans'),
        hovertemplate='%{label}: %{value} customers<extra></extra>',
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#C9D1D9'),
        showlegend=True,
        legend=dict(font=dict(color='#C9D1D9')),
    )
    return fig


def plot_age_vs_churn(result_df):
    """Box plot: Age distribution by predicted churn."""
    stay_ages  = result_df[result_df['Predicted'] == '🟢 Stay']['Age'].dropna()
    churn_ages = result_df[result_df['Predicted'] == '🔴 Churn']['Age'].dropna()

    fig = go.Figure()
    fig.add_trace(go.Box(y=stay_ages,  name='Stay',  marker_color='#3FB950', line_color='#3FB950'))
    fig.add_trace(go.Box(y=churn_ages, name='Churn', marker_color='#F85149', line_color='#F85149'))
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(color='#C9D1D9', gridcolor='#21262D'),
        xaxis=dict(color='#C9D1D9'),
        font=dict(family='DM Sans', color='#C9D1D9'),
        showlegend=False,
    )
    return fig


def style_predictions_table(df):
    """Return a display-ready version of the results DataFrame."""
    display_cols = ['Predicted', 'Churn Probability', 'Risk Level']
    if 'CustomerId' in df.columns:
        display_cols = ['CustomerId'] + display_cols
    if 'Surname' in df.columns:
        display_cols = ['Surname'] + display_cols
    for c in ['Age', 'Geography', 'Gender', 'Balance', 'NumOfProducts']:
        if c in df.columns:
            display_cols.append(c)
    return df[[c for c in display_cols if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Model upload + Single Customer form
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 18px 0 10px'>
        <span style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#F0C040'>🏦 ChurnGuard-Alert</span><br>
        <span style='color:#8B949E; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em'>ML Prediction Suite</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model artifacts ──
    st.markdown("#### 📦 Load Model Artifacts")
    model_file  = st.file_uploader("model_file.pkl",  type=['pkl'], key='model')
    scaler_file = st.file_uploader("Scaler_file.pkl", type=['pkl'], key='scaler')

    artifacts_loaded = False
    model, scaler = None, None

    if model_file and scaler_file:
        try:
            model, scaler = load_artifacts(model_file.read(), scaler_file.read())
            st.success("✅ Artifacts loaded")
            artifacts_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading artifacts: {e}")
    else:
        st.info("Upload both .pkl files to begin")

    st.divider()

    # ── Single Customer Form ──
    st.markdown("#### 👤 Single Customer")
    with st.expander("Enter customer details", expanded=False):
        geo        = st.selectbox("Geography",     ["France", "Germany", "Spain"])
        gender     = st.selectbox("Gender",        ["Female", "Male"])
        card_type  = st.selectbox("Card Type",     ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
        age        = st.slider("Age",              18, 92, 38)
        credit     = st.slider("Credit Score",     300, 850, 650)
        tenure     = st.slider("Tenure (years)",   0, 10, 5)
        products   = st.slider("Num of Products",  1, 4, 1)
        satisf     = st.slider("Satisfaction",     1, 5, 3)
        balance    = st.number_input("Balance (€)",         0.0, 300000.0, 85000.0, step=1000.0)
        salary     = st.number_input("Est. Salary (€)",     0.0, 250000.0, 75000.0, step=1000.0)
        points     = st.number_input("Points Earned",       0, 1000, 400, step=10)
        has_card   = st.toggle("Has Credit Card",  value=True)
        is_active  = st.toggle("Is Active Member", value=True)

        single_inputs = dict(
            geography=geo, gender=gender, card_type=card_type, age=age,
            credit_score=credit, tenure=tenure, num_products=products,
            satisfaction=satisf, balance=balance, salary=salary,
            points=points, has_cr_card=has_card, is_active=is_active
        )

        predict_single_btn = st.button("⚡ Predict This Customer", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Bank Customer Churn Predictor</div>
<div class='hero-sub'>Machine Learning · Random Forest · End-to-End Pipeline</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂 Batch Prediction", "👤 Single Customer", "📊 Model Insights"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch CSV upload
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Upload Customer Data for Batch Prediction</div>", unsafe_allow_html=True)

    if not artifacts_loaded:
        st.warning("⚠️ Please upload both model artifacts in the sidebar first.")
    else:
        csv_file = st.file_uploader(
            "Drop your customer CSV here",
            type=['csv'],
            help="Should match the Customer-Churn-Records.csv schema. The 'Exited' column is optional."
        )

        if csv_file:
            try:
                df_raw = pd.read_csv(csv_file)
                st.caption(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

                with st.spinner("Running predictions..."):
                    result_df = predict_dataframe(df_raw, model, scaler)

                # ── KPI Cards ──
                total     = len(result_df)
                churners  = (result_df['Predicted'] == '🔴 Churn').sum()
                stayers   = total - churners
                avg_prob  = result_df['Churn Probability'].mean()
                high_risk = (result_df['Risk Level'] == 'High').sum()

                st.markdown(f"""
                <div class='kpi-row'>
                  <div class='kpi-card gold'>
                    <div class='kpi-label'>Total Customers</div>
                    <div class='kpi-value'>{total:,}</div>
                  </div>
                  <div class='kpi-card red'>
                    <div class='kpi-label'>Predicted Churners</div>
                    <div class='kpi-value red'>{churners:,}</div>
                  </div>
                  <div class='kpi-card green'>
                    <div class='kpi-label'>Predicted to Stay</div>
                    <div class='kpi-value green'>{stayers:,}</div>
                  </div>
                  <div class='kpi-card blue'>
                    <div class='kpi-label'>High-Risk Customers</div>
                    <div class='kpi-value blue'>{high_risk:,}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Charts row ──
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Risk Level Distribution**")
                    st.plotly_chart(plot_risk_distribution(result_df), use_container_width=True, config={'displayModeBar': False})
                with col_b:
                    st.markdown("**Age Distribution by Prediction**")
                    if 'Age' in result_df.columns:
                        st.plotly_chart(plot_age_vs_churn(result_df), use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("No 'Age' column in uploaded file.")
                with col_c:
                    st.markdown("**Churn Probability Histogram**")
                    fig_hist = px.histogram(
                        result_df, x='Churn Probability', nbins=30,
                        color_discrete_sequence=['#F0C040']
                    )
                    fig_hist.update_layout(
                        height=280, margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(color='#C9D1D9', gridcolor='#21262D'),
                        yaxis=dict(color='#C9D1D9', gridcolor='#21262D'),
                        font=dict(family='DM Sans', color='#C9D1D9'),
                        bargap=0.05
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

                st.divider()

                # ── Predictions Table ──
                st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)

                # Filter controls
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    filter_pred = st.multiselect("Filter by Prediction", ['🔴 Churn', '🟢 Stay'],
                                                  default=['🔴 Churn', '🟢 Stay'])
                with fc2:
                    filter_risk = st.multiselect("Filter by Risk Level", ['Low', 'Medium', 'High'],
                                                  default=['Low', 'Medium', 'High'])
                with fc3:
                    min_prob = st.slider("Min Churn Probability (%)", 0, 100, 0)

                filtered = result_df[
                    result_df['Predicted'].isin(filter_pred) &
                    result_df['Risk Level'].isin(filter_risk) &
                    (result_df['Churn Probability'] >= min_prob)
                ]
                st.caption(f"Showing {len(filtered):,} of {total:,} customers")
                st.dataframe(
                    style_predictions_table(filtered),
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                )

                # ── Download ──
                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Predictions CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.exception(e)
        else:
            st.markdown("""
            <div style='background:#161B22; border:1px dashed #30363D; border-radius:12px;
                        padding:40px; text-align:center; color:#8B949E; margin-top:20px;'>
                <div style='font-size:2.5rem; margin-bottom:12px'>📁</div>
                <div style='font-size:1.0rem'>Upload a CSV file to generate batch predictions</div>
                <div style='font-size:0.82rem; margin-top:6px'>Expected format: Customer-Churn-Records.csv schema</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Customer
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Single Customer Prediction</div>", unsafe_allow_html=True)

    if not artifacts_loaded:
        st.warning("⚠️ Please upload both model artifacts in the sidebar first.")
    elif predict_single_btn:
        single_df = build_single_customer(single_inputs)

        try:
            result_single = predict_dataframe(single_df, model, scaler)
            prob     = result_single['Churn Probability'].iloc[0] / 100
            pred_val = result_single['Predicted'].iloc[0]
            risk     = result_single['Risk Level'].iloc[0]

            col_gauge, col_info = st.columns([1, 1])

            with col_gauge:
                st.plotly_chart(plot_gauge(prob), use_container_width=True, config={'displayModeBar': False})

            with col_info:
                st.markdown("<br>", unsafe_allow_html=True)
                badge_class = 'badge-churn' if '🔴' in pred_val else 'badge-stay'
                label       = 'HIGH CHURN RISK' if '🔴' in pred_val else 'LIKELY TO STAY'
                st.markdown(f"<span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                risk_color = {'Low': '#3FB950', 'Medium': '#F0C040', 'High': '#F85149'}.get(str(risk), '#8B949E')
                st.markdown(f"""
                <div style='background:#161B22; border:1px solid #21262D; border-radius:10px; padding:16px 20px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                        <span style='color:#8B949E; font-size:0.82rem'>CHURN PROBABILITY</span>
                        <span style='color:#F0C040; font-weight:600'>{prob*100:.1f}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                        <span style='color:#8B949E; font-size:0.82rem'>RISK LEVEL</span>
                        <span style='color:{risk_color}; font-weight:600'>{risk}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between'>
                        <span style='color:#8B949E; font-size:0.82rem'>STAY PROBABILITY</span>
                        <span style='color:#3FB950; font-weight:600'>{(1-prob)*100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if prob >= 0.65:
                    st.markdown("""
                    <div style='background:rgba(248,81,73,0.08); border:1px solid rgba(248,81,73,0.3);
                                border-radius:8px; padding:14px; margin-top:14px; font-size:0.88rem; color:#C9D1D9'>
                        <b style='color:#F85149'>⚠️ Retention Action Recommended</b><br>
                        Consider a personalised retention offer: loyalty bonus, fee waiver, or dedicated account manager outreach.
                    </div>
                    """, unsafe_allow_html=True)
                elif prob >= 0.40:
                    st.markdown("""
                    <div style='background:rgba(240,192,64,0.08); border:1px solid rgba(240,192,64,0.3);
                                border-radius:8px; padding:14px; margin-top:14px; font-size:0.88rem; color:#C9D1D9'>
                        <b style='color:#F0C040'>👁 Monitor This Customer</b><br>
                        Medium risk — schedule a proactive check-in or satisfaction survey.
                    </div>
                    """, unsafe_allow_html=True)

            # ── Input Summary ──
            with st.expander("View submitted input values"):
                st.dataframe(single_df.T.rename(columns={0: 'Value'}), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

    else:
        st.markdown("""
        <div style='background:#161B22; border:1px dashed #30363D; border-radius:12px;
                    padding:60px; text-align:center; color:#8B949E;'>
            <div style='font-size:2.5rem; margin-bottom:12px'>👤</div>
            <div style='font-size:1.0rem'>Fill in the customer details in the sidebar</div>
            <div style='font-size:0.82rem; margin-top:6px; color:#58A6FF'>then click ⚡ Predict This Customer</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Model Insights & Pipeline Overview</div>", unsafe_allow_html=True)

    if not artifacts_loaded:
        st.warning("⚠️ Upload model artifacts to view feature importances.")
    else:
        col_fi, col_info = st.columns([1.4, 1])

        with col_fi:
            st.markdown("**Top 10 Feature Importances** (Random Forest — Mean Decrease in Impurity)")
            st.plotly_chart(plot_feature_importance(model), use_container_width=True, config={'displayModeBar': False})

        with col_info:
            st.markdown("**Model Card**")
            st.markdown(f"""
            <div style='background:#161B22; border:1px solid #21262D; border-radius:10px; padding:18px; font-size:0.88rem'>
                <div style='color:#8B949E; margin-bottom:4px'>Algorithm</div>
                <div style='color:#C9D1D9; margin-bottom:12px'>Random Forest Classifier</div>

                <div style='color:#8B949E; margin-bottom:4px'>Trees in Ensemble</div>
                <div style='color:#C9D1D9; margin-bottom:12px'>{getattr(model, 'n_estimators', 'N/A')}</div>

                <div style='color:#8B949E; margin-bottom:4px'>Input Features</div>
                <div style='color:#C9D1D9; margin-bottom:12px'>{getattr(model, 'n_features_in_', 'N/A')}</div>

                <div style='color:#8B949E; margin-bottom:4px'>Training Resampling</div>
                <div style='color:#C9D1D9; margin-bottom:12px'>SMOTE (balanced 50/50)</div>

                <div style='color:#8B949E; margin-bottom:4px'>Scaling Method</div>
                <div style='color:#C9D1D9; margin-bottom:12px'>CustomScaler (numerical cols only)</div>

                <div style='color:#8B949E; margin-bottom:4px'>Categorical Encoding</div>
                <div style='color:#C9D1D9'>One-Hot (drop_first=True)</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Pipeline walkthrough ──
    st.markdown("<div class='section-header'>Pipeline Walkthrough</div>", unsafe_allow_html=True)

    steps = [
        ("N1", "Data Upload",          "Load raw CSV · Check shape, dtypes · Data dictionary"),
        ("N2", "EDA",                  "Distributions · Correlation matrix · Leakage detection"),
        ("N3", "Data Cleaning",        "Drop identifiers & Complain · Outlier rationale"),
        ("N4", "Feature Engineering",  "StandardScaler · One-Hot Encoding · SMOTE balancing"),
        ("N5", "Model Selection",      "Train 6 models · Compare Accuracy / F1 / ROC-AUC"),
        ("N6", "Model Saving",         "Retrain on 100% data · Pickle model & scaler"),
        ("N7", "Inference Module",     "CustomerChurn class · Production-ready API wrapper"),
    ]

    cols = st.columns(len(steps))
    for col, (nb, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:#161B22; border:1px solid #21262D; border-radius:10px;
                    padding:14px 12px; text-align:center; height:140px'>
            <div style='color:#F0C040; font-family:DM Serif Display,serif; font-size:1.1rem'>{nb}</div>
            <div style='color:#C9D1D9; font-weight:600; font-size:0.82rem; margin:6px 0'>{title}</div>
            <div style='color:#8B949E; font-size:0.72rem; line-height:1.5'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── How to run ──
    with st.expander("📖 How to Run This App Locally"):
        st.code("""
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn imbalanced-learn plotly

# 2. Make sure these files are in the same folder:
#    app.py   model_file.pkl   Scaler_file.pkl

# 3. Run
streamlit run app.py

# 4. Open your browser at  http://localhost:8501
        """, language="bash")

    with st.expander("📖 How the Inference Pipeline Works"):
        st.markdown("""
**Step-by-step for each prediction request:**

1. **Load artifacts** — `model_file.pkl` and `Scaler_file.pkl` are loaded once on startup (cached).
2. **Drop identifiers** — `RowNumber`, `CustomerId`, `Surname`, `Complain` are removed.
3. **Scale numerical features** — The pre-fitted `CustomScaler` applies `transform()` (never `fit_transform()`).
4. **One-hot encode** — `Geography`, `Gender`, `Card Type` are encoded with `pd.get_dummies(drop_first=True)`.
5. **Reindex** — Columns are reordered to exactly match the model's expected input using `reindex(fill_value=0)`.
6. **Predict** — `model.predict()` returns 0/1 · `model.predict_proba()` returns the churn probability.
7. **Attach to original data** — Predictions are appended to the raw DataFrame for readability.

**Why `transform()` and not `fit_transform()`?**
`fit_transform()` would re-learn mean and standard deviation from the new input data, producing different scaling than training. Using `transform()` with the saved scaler guarantees the same numerical transformation as during model training.
        """)
