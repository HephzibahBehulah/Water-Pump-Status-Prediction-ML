"""
Hydro Dominion — Water Pump Status Predictor
ReDI School Data Circle 2026
Models: XGBoost + SMOTE  |  Random Forest + SMOTE
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ===== CONSTANTS =====
TIER1_FEATURES = [
    'quantity', 'lga', 'extraction_type_group', 'waterpoint_type',
    'longitude', 'latitude', 'funder', 'installer', 'pump_age',
    'region', 'water_quality', 'source', 'payment_binary',
    'management', 'gps_height'
]
FEATURE_LABELS = {
    'quantity': 'Water Quantity', 'lga': 'District (LGA)',
    'extraction_type_group': 'Extraction Type', 'waterpoint_type': 'Waterpoint Type',
    'longitude': 'Longitude', 'latitude': 'Latitude', 'funder': 'Funder',
    'installer': 'Installer', 'pump_age': 'Pump Age (yrs)', 'region': 'Region',
    'water_quality': 'Water Quality', 'source': 'Water Source',
    'payment_binary': 'Payment (0=never, 1=yes)', 'management': 'Management',
    'gps_height': 'GPS Height (m)',
}
ORDINAL_COLS = ['water_quality', 'quantity', 'population_bin']
NOMINAL_COLS = ['lga', 'region', 'extraction_type_group', 'waterpoint_type',
                'source', 'installer', 'funder', 'management']
STATUS_COLORS = {
    'functional': '#28a745',
    'functional needs repair': '#ffc107',
    'non functional': '#dc3545',
}
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Hydro Dominion",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
  .functional    {background:#d4edda;border-left:5px solid #28a745;padding:1rem;border-radius:8px;margin-bottom:.5rem}
  .nonfunctional {background:#f8d7da;border-left:5px solid #dc3545;padding:1rem;border-radius:8px;margin-bottom:.5rem}
  .needsrepair   {background:#fff3cd;border-left:5px solid #ffc107;padding:1rem;border-radius:8px;margin-bottom:.5rem}
  .metric-card   {background:#f8f9fa;border:1px solid #dee2e6;padding:1rem;border-radius:8px;text-align:center}
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    base   = Path(__file__).parent
    xgb    = joblib.load(base / 'xgboost_smote_notebook.pkl')
    rf     = joblib.load(base / 'rf_smote_notebook.pkl')
    le     = joblib.load(base / 'label_encoder.pkl')
    oe_ord = joblib.load(base / 'oe_ordinal_fitted.pkl')
    oe_nom = joblib.load(base / 'oe_nominal_fitted.pkl')
    return xgb, rf, le, oe_ord, oe_nom

@st.cache_resource
def get_explainers(_xgb, _rf):
    return shap.TreeExplainer(_xgb), shap.TreeExplainer(_rf)

xgb_model, rf_model, le, oe_ordinal, oe_nominal = load_models()
explainer_xgb, explainer_rf = get_explainers(xgb_model, rf_model)
CLASS_NAMES = list(le.classes_)

# ===== LOAD TRAINING DATA =====
@st.cache_data(show_spinner=False)
def load_training_data():
    values = pd.read_csv(DATA_DIR / 'training_values.csv')
    labels = pd.read_csv(DATA_DIR / 'training_labels.csv')
    df = values.merge(labels, on='id')
    df['year_recorded'] = pd.to_datetime(df['date_recorded'], errors='coerce').dt.year.fillna(2013)
    df['pump_age'] = (df['year_recorded'] - df['construction_year']).clip(lower=0)
    df['pump_age'] = df['pump_age'].fillna(df['pump_age'].median())
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
    df['payment_binary'] = df['payment'].apply(
        lambda x: 0 if str(x).strip().lower() == 'never pay' else 1)
    for col in NOMINAL_COLS + ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()
    df['population_bin'] = pd.cut(df['population'], bins=[-1,0,25,215,680,np.inf],
        labels=['no_pop','small','medium','large','very_large']).astype(str)
    df['gps_height'] = pd.to_numeric(df['gps_height'], errors='coerce').fillna(0)
    return df

# ===== VALIDATION METRICS (computed once, cached) =====
@st.cache_data(show_spinner=False)
def compute_val_metrics():
    df  = load_training_data()
    y   = le.transform(df['status_group'])
    enc = df.copy()
    enc[ORDINAL_COLS] = oe_ordinal.transform(enc[ORDINAL_COLS].astype(str))
    enc[NOMINAL_COLS] = oe_nominal.transform(enc[NOMINAL_COLS].astype(str))
    for col in TIER1_FEATURES:
        if col not in enc.columns:
            enc[col] = 0
    X = enc[TIER1_FEATURES].values
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    xgb_p = xgb_model.predict(X_val)
    rf_p  = rf_model.predict(X_val)
    return (confusion_matrix(y_val, xgb_p),
            confusion_matrix(y_val, rf_p),
            y_val, xgb_p, rf_p)

# ===== PREPROCESSING =====
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['population'] = pd.to_numeric(df.get('population', 300), errors='coerce').fillna(0)
    df['population_log']  = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    df['population_bin']  = pd.cut(df['population'], bins=[-1,0,25,215,680,np.inf],
        labels=['no_pop','small','medium','large','very_large']).astype(str)
    df['gps_height'] = pd.to_numeric(df.get('gps_height', 669), errors='coerce').fillna(0)
    df['gps_height_zero'] = (df['gps_height'] == 0).astype(int)
    df['payment_binary'] = df['payment'].apply(
        lambda x: 0 if str(x).strip().lower() == 'never pay' else 1)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().fillna('unknown')
        else:
            df[col] = 'unknown'
    for col in ORDINAL_COLS:
        if col not in df.columns:
            df[col] = 'unknown'
    df[ORDINAL_COLS] = oe_ordinal.transform(df[ORDINAL_COLS].astype(str))
    df[NOMINAL_COLS] = oe_nominal.transform(df[NOMINAL_COLS].astype(str))
    return df

def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = engineer_features(df)
    for col, val in {'longitude':33.94,'latitude':-6.50,'pump_age':10.0,'gps_height':669.0}.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)
        else:
            df[col] = val
    df = encode_features(df)
    for col in TIER1_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[TIER1_FEATURES].values

def predict(df_raw: pd.DataFrame):
    X = preprocess(df_raw)
    return (le.inverse_transform(xgb_model.predict(X)), xgb_model.predict_proba(X),
            le.inverse_transform(rf_model.predict(X)),  rf_model.predict_proba(X))

# ===== SHARED FORM =====
# Real pumps from the dataset — used to seed the form with a random example
# on every fresh page load instead of one frozen default.
PRESET_PUMPS = [
    {'region': 'Mwanza', 'basin': 'Lake Victoria', 'extraction_type_group': 'gravity',
     'water_quality': 'soft', 'quantity': 'insufficient', 'construction_year': 1999,
     'population': 1200, 'gps_height': 1226, 'management': 'vwc', 'payment': 'never pay'},
    {'region': 'Ruvuma', 'basin': 'Ruvuma / Southern Coast', 'extraction_type_group': 'gravity',
     'water_quality': 'soft', 'quantity': 'enough', 'construction_year': 1986,
     'population': 1, 'gps_height': 462, 'management': 'vwc', 'payment': 'monthly'},
    {'region': 'Manyara', 'basin': 'Internal', 'extraction_type_group': 'other',
     'water_quality': 'soft', 'quantity': 'seasonal', 'construction_year': 2012,
     'population': 321, 'gps_height': 1996, 'management': 'parastatal', 'payment': 'never pay'},
    {'region': 'Singida', 'basin': 'Internal', 'extraction_type_group': 'other',
     'water_quality': 'soft', 'quantity': 'insufficient', 'construction_year': 2010,
     'population': 500, 'gps_height': 1567, 'management': 'vwc', 'payment': 'never pay'},
    {'region': 'Arusha', 'basin': 'Pangani', 'extraction_type_group': 'gravity',
     'water_quality': 'soft', 'quantity': 'insufficient', 'construction_year': 2000,
     'population': 300, 'gps_height': 1569, 'management': 'vwc', 'payment': 'never pay'},
    {'region': 'Ruvuma', 'basin': 'Ruvuma / Southern Coast', 'extraction_type_group': 'gravity',
     'water_quality': 'soft', 'quantity': 'enough', 'construction_year': 2000,
     'population': 60, 'gps_height': 1260, 'management': 'water board', 'payment': 'monthly'},
]

PRESET_KEY_PREFIXES = ["single_", "shap_"]

def _seed_presets_once():
    """Pick ONE random preset per session and seed both the Single Pump
    Check and SHAP Explainer forms with it, so a refresh shows the same
    pump on both pages instead of two independent random picks."""
    if "_preset_seeded" in st.session_state:
        return
    preset = random.choice(PRESET_PUMPS)
    for prefix in PRESET_KEY_PREFIXES:
        st.session_state[f"{prefix}reg"]   = preset['region']
        st.session_state[f"{prefix}basin"] = preset['basin']
        st.session_state[f"{prefix}ext"]   = preset['extraction_type_group']
        st.session_state[f"{prefix}wq"]    = preset['water_quality']
        st.session_state[f"{prefix}qty"]   = preset['quantity']
        st.session_state[f"{prefix}cyear"] = preset['construction_year']
        st.session_state[f"{prefix}pop"]   = preset['population']
        st.session_state[f"{prefix}gps"]   = preset['gps_height']
        st.session_state[f"{prefix}mgmt"]  = preset['management']
        st.session_state[f"{prefix}pay"]   = preset['payment']
    st.session_state["_preset_seeded"] = True

def pump_input_form(key_prefix=""):
    REFERENCE_YEAR = 2013  # most training records were recorded in 2011-2013

    _seed_presets_once()

    region_options = sorted([
        'Arusha','Dar es Salaam','Dodoma','Geita','Iringa','Kagera','Katavi','Kigoma',
        'Kilimanjaro','Lindi','Manyara','Mara','Mbeya','Morogoro','Mtwara','Mwanza',
        'Njombe','Pwani','Rukwa','Ruvuma','Shinyanga','Simiyu','Singida','Songwe',
        'Tabora','Tanga'
    ])
    basin_options = ['Lake Victoria','Pangani','Rufiji','Internal','Wami / Ruvu',
        'Lake Tanganyika','Lake Nyasa','Ruvuma / Southern Coast','Lake Rukwa']
    extraction_options = ['gravity','hand pump','motorpump','submersible','rope pump',
        'nira/tanira','swn 80','mono','afridev','windmill','india mark ii','other','unknown']
    water_quality_options = ['soft','milky','salty','coloured','fluoride',
        'fluoride abandoned','salty abandoned','unknown']
    quantity_options = ['enough','insufficient','seasonal','dry','unknown']
    management_options = ['vwc','wug','water authority','wua','water board',
        'private operator','company','parastatal','other','unknown']
    payment_options = ['never pay','per bucket','monthly','on failure','annually','other','unknown']

    st.markdown("##### Pump Details")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        region = st.selectbox("Region", region_options, key=f"{key_prefix}reg")
    with c2:
        basin = st.selectbox("Basin", basin_options, key=f"{key_prefix}basin")
    with c3:
        extraction_type_group = st.selectbox("Extraction", extraction_options, key=f"{key_prefix}ext")
    with c4:
        water_quality = st.selectbox("Water Quality", water_quality_options, key=f"{key_prefix}wq")
    with c5:
        quantity = st.selectbox("Quantity", quantity_options, key=f"{key_prefix}qty")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        construction_year = st.number_input("Construction Year", min_value=1960, max_value=REFERENCE_YEAR,
            key=f"{key_prefix}cyear")
    with c2:
        population = st.number_input("Population", min_value=0, max_value=50000,
            step=50, key=f"{key_prefix}pop")
    with c3:
        gps_height = st.number_input("GPS Height (m)", min_value=-100, max_value=5000,
            key=f"{key_prefix}gps")
    with c4:
        management = st.selectbox("Management", management_options, key=f"{key_prefix}mgmt")
    with c5:
        payment = st.selectbox("Payment", payment_options, key=f"{key_prefix}pay")

    pump_age = max(0, REFERENCE_YEAR - construction_year)

    # Fields the model needs but aren't on this simplified form — held at the
    # most common value observed in training data so predictions stay stable.
    source = 'spring'
    waterpoint_type = 'communal standpipe'
    longitude = 35.15
    latitude = -5.71
    lga = 'njombe'
    installer = 'dwe'
    funder = 'government of tanzania'

    return pd.DataFrame([{
        'quantity': quantity, 'water_quality': water_quality, 'source': source,
        'waterpoint_type': waterpoint_type, 'extraction_type_group': extraction_type_group,
        'payment': payment, 'longitude': longitude, 'latitude': latitude,
        'gps_height': gps_height, 'region': region.lower(), 'basin': basin.lower(), 'lga': lga,
        'pump_age': pump_age, 'population': population, 'management': management.lower(),
        'installer': installer, 'funder': funder
    }])

# ===== DISPLAY HELPERS =====
def show_result(label, probas, model_name):
    css   = {'functional':'functional','non functional':'nonfunctional',
             'functional needs repair':'needsrepair'}.get(label.lower(),'nonfunctional')
    emoji = {'functional':'✅','non functional':'❌','functional needs repair':'⚠️'}.get(label.lower(),'❓')
    st.markdown(f"""<div class="{css}"><strong>{model_name}</strong><br>
        <span style="font-size:1.3rem">{emoji} <b>{label.upper()}</b></span><br>
        Confidence: <b>{probas.max()*100:.1f}%</b></div>""", unsafe_allow_html=True)
    for cls, prob in zip(le.classes_, probas):
        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

def shap_waterfall(sv, idx, X_row, model_name, label):
    try:
        base = sv.base_values[0, idx] if sv.base_values.ndim == 2 else sv.base_values[0]
        exp  = shap.Explanation(values=sv.values[0, :, idx], base_values=base,
                                data=X_row, feature_names=[FEATURE_LABELS.get(f,f) for f in TIER1_FEATURES])
        fig, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"{model_name} — {label}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Waterfall error ({model_name}): {e}")

def shap_bar(sv, idx, model_name):
    try:
        vals  = sv.values[0, :, idx]
        names = [FEATURE_LABELS.get(f, f) for f in TIER1_FEATURES]
        order = np.argsort(np.abs(vals))[::-1][:10]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh([names[i] for i in order[::-1]], vals[order[::-1]],
                color=['#d73027' if vals[i] > 0 else '#4575b4' for i in order[::-1]])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"{model_name} — top contributions", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Bar chart error ({model_name}): {e}")

# ===== SIDEBAR =====
st.sidebar.title("💧 Hydro Dominion")
st.sidebar.caption("Water Pump Status Predictor — Tanzania")
st.sidebar.divider()
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", [
    "Tanzania Map",
    "EDA Insights",
    "Model Performance",
    "Single Pump Check",
    "SHAP Explainer",
    "Batch Analysis",
    "About",
])
st.sidebar.divider()
st.sidebar.markdown("""
### Prediction Classes
- ✅ **Functional** — working, no action needed
- ⚠️ **Needs Repair** — working but requires maintenance
- ❌ **Non-functional** — not working, urgent action required
""")

# =============================================================
# PAGE: TANZANIA MAP
# =============================================================
if page == "Tanzania Map":
    st.subheader("Tanzania Pump Status — Interactive Map")
    st.caption("59,400 water pumps. Click a point for details. Use filters to narrow down.")

    with st.spinner("Loading pump data..."):
        df_map = load_training_data()

    # Filter out bad coordinates
    df_map = df_map[(df_map['longitude'].between(28, 42)) & (df_map['latitude'].between(-12, -1))].copy()

    # Filters (above the map)
    col1, col2, col3 = st.columns(3)
    with col1:
        regions = ['All'] + sorted(df_map['region'].str.title().unique().tolist())
        sel_region = st.selectbox("Filter by Region", regions)
    with col2:
        ext_types = ['All'] + sorted(df_map['extraction_type_group'].str.title().unique().tolist())
        sel_ext = st.selectbox("Filter by Extraction Type", ext_types)
    with col3:
        mgmt_types = ['All'] + sorted(df_map['management'].str.title().unique().tolist())
        sel_mgmt = st.selectbox("Filter by Management", mgmt_types)

    df_plot = df_map.copy()
    if sel_region != 'All':
        df_plot = df_plot[df_plot['region'].str.title() == sel_region]
    if sel_ext != 'All':
        df_plot = df_plot[df_plot['extraction_type_group'].str.title() == sel_ext]
    if sel_mgmt != 'All':
        df_plot = df_plot[df_plot['management'].str.title() == sel_mgmt]

    sample = df_plot.sample(min(15000, len(df_plot)), random_state=42)
    sample['status_label'] = sample['status_group'].str.title()

    fig_map = px.scatter_mapbox(
        sample,
        lat='latitude', lon='longitude',
        color='status_label',
        color_discrete_map={
            'Functional': '#28a745',
            'Functional Needs Repair': '#ffc107',
            'Non Functional': '#dc3545',
        },
        hover_data={
            'region': True, 'extraction_type_group': True,
            'management': True, 'pump_age': True,
            'latitude': False, 'longitude': False,
        },
        mapbox_style='open-street-map',
        zoom=5, center={'lat': -6.5, 'lon': 34.8},
        height=580,
        title=f"Showing {len(sample):,} of {len(df_plot):,} pumps"
    )
    fig_map.update_layout(legend_title_text='Status', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Summary stats for the current filter
    st.divider()
    counts = df_plot['status_group'].value_counts()
    total  = len(df_plot)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pumps", f"{total:,}")
    c2.metric("✅ Functional",
               f"{counts.get('functional', 0):,}",
               f"{counts.get('functional',0)/total*100:.1f}%")
    c3.metric("⚠️ Needs Repair",
               f"{counts.get('functional needs repair', 0):,}",
               f"{counts.get('functional needs repair',0)/total*100:.1f}%")
    c4.metric("❌ Non-functional",
               f"{counts.get('non functional', 0):,}",
               f"{counts.get('non functional',0)/total*100:.1f}%")

# =============================================================
# PAGE: EDA INSIGHTS
# =============================================================
elif page == "EDA Insights":
    st.subheader("EDA Insights — Sprint 1 Key Findings")

    with st.spinner("Loading dataset..."):
        df = load_training_data()

    # ---- KPI row ----
    total = len(df)
    counts = df['status_group'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pumps", f"{total:,}")
    c2.metric("✅ Functional", f"{counts.get('functional',0)/total*100:.1f}%")
    c3.metric("⚠️ Needs Repair", f"{counts.get('functional needs repair',0)/total*100:.1f}%")
    c4.metric("❌ Non-functional", f"{counts.get('non functional',0)/total*100:.1f}%")

    st.divider()

    # ---- Chart 1: Water Quantity vs Status ----
    st.markdown("### Finding 1 — Water Quantity is the Strongest Signal")
    qty_map = {'dry': 1, 'unknown': 2, 'seasonal': 3, 'insufficient': 4, 'enough': 5}
    qty_grp = (df.groupby(['quantity','status_group'])
                 .size().reset_index(name='count'))
    qty_total = qty_grp.groupby('quantity')['count'].transform('sum')
    qty_grp['pct'] = qty_grp['count'] / qty_total * 100
    qty_grp['order'] = qty_grp['quantity'].map(qty_map).fillna(6)
    qty_grp = qty_grp.sort_values('order')
    fig1 = px.bar(qty_grp, x='quantity', y='pct', color='status_group',
                  color_discrete_map=STATUS_COLORS, barmode='stack',
                  labels={'pct':'% of pumps','quantity':'Water Quantity','status_group':'Status'},
                  text_auto='.0f', height=380)
    fig1.update_traces(texttemplate='%{y:.0f}%', textposition='inside',
                       textfont_size=11, insidetextanchor='middle')
    fig1.update_layout(legend_title_text='Status', xaxis_title='Water Quantity',
                       yaxis_title='% of Pumps', uniformtext_minsize=8)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "**Takeaway:** Dry water = 97%+ non-functional. 'Enough' water = 65%+ functional. "
        "Water quantity is the single strongest predictor of pump status."
    )

    st.divider()

    # ---- Chart 2: Management Type vs Functional % ----
    st.markdown("### Finding 2 — Professional Management Keeps Pumps Running")
    mgmt_grp = (df.groupby(['management','status_group'])
                  .size().reset_index(name='count'))
    mgmt_total = mgmt_grp.groupby('management')['count'].transform('sum')
    mgmt_grp['pct'] = mgmt_grp['count'] / mgmt_total * 100
    func_pct = (mgmt_grp[mgmt_grp['status_group']=='functional']
                .set_index('management')['pct']
                .sort_values(ascending=True))
    func_pct = func_pct[func_pct.index != 'unknown']
    fig2 = px.bar(x=func_pct.values, y=func_pct.index, orientation='h',
                  color=func_pct.values,
                  color_continuous_scale=[[0,'#dc3545'],[0.5,'#ffc107'],[1,'#28a745']],
                  labels={'x':'% Functional','y':'Management Type'},
                  text=func_pct.values.round(1), height=420)
    fig2.update_traces(texttemplate='%{x:.0f}%', textposition='outside')
    fig2.update_layout(coloraxis_showscale=False, xaxis_range=[0,100])
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "**Takeaway:** Private operators, water boards, and WUAs keep 70–80% of pumps functional. "
        "Poorly supervised types (school, none) have much higher failure rates."
    )

    st.divider()

    # ---- Chart 3: Extraction Type vs Non-functional Rate ----
    st.markdown("### Finding 3 — Pump Type Influences Lifespan")
    ext_grp = (df.groupby(['extraction_type_group','status_group'])
                 .size().reset_index(name='count'))
    ext_total = ext_grp.groupby('extraction_type_group')['count'].transform('sum')
    ext_grp['pct'] = ext_grp['count'] / ext_total * 100
    nonfunc_pct = (ext_grp[ext_grp['status_group']=='non functional']
                   .set_index('extraction_type_group')['pct']
                   .sort_values(ascending=False))
    nonfunc_pct = nonfunc_pct[nonfunc_pct.index != 'other']
    fig3 = px.bar(x=nonfunc_pct.index, y=nonfunc_pct.values,
                  color=nonfunc_pct.values,
                  color_continuous_scale=[[0,'#28a745'],[0.4,'#ffc107'],[1,'#dc3545']],
                  labels={'x':'Extraction Type','y':'% Non-functional'},
                  text=nonfunc_pct.values.round(1), height=380)
    fig3.update_traces(texttemplate='%{y:.0f}%', textposition='outside')
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "**Takeaway:** Rope pumps and hand pumps fail most often. "
        "Gravity and mono pumps are the most reliable — "
        "they have fewer mechanical parts and lower maintenance needs."
    )

    st.divider()

    # ---- Chart 4: Pump Age vs Failure ----
    st.markdown("### Finding 4 — Pump Age and Sudden Failure Spikes")
    df_age = df[(df['pump_age'] >= 0) & (df['pump_age'] <= 60)].copy()
    df_age['age_bucket'] = pd.cut(df_age['pump_age'],
        bins=[0,5,10,15,20,25,30,35,40,50,60],
        labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-50','50-60'],
        include_lowest=True)
    age_grp = df_age.groupby(['age_bucket','status_group']).size().reset_index(name='count')
    age_total = age_grp.groupby('age_bucket')['count'].transform('sum')
    age_grp['pct'] = age_grp['count'] / age_total * 100

    fig4 = px.bar(age_grp, x='age_bucket', y='pct', color='status_group',
                  color_discrete_map=STATUS_COLORS, barmode='stack',
                  labels={'pct':'% of pumps','age_bucket':'Pump Age (years)','status_group':'Status'},
                  height=380)
    fig4.update_layout(legend_title_text='Status', xaxis_title='Pump Age (years)', yaxis_title='% of Pumps')
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "**Takeaway:** Non-functional rates increase steadily with age. "
        "Sharp spikes appear at 40–50+ years — these are high-priority replacement targets."
    )

    st.divider()

    # ---- Chart 5: Regional failure rate ----
    st.markdown("### Finding 5 — Regional Hotspots")
    reg_grp = (df.groupby(['region','status_group'])
                 .size().reset_index(name='count'))
    reg_total = reg_grp.groupby('region')['count'].transform('sum')
    reg_grp['pct'] = reg_grp['count'] / reg_total * 100
    nf_by_reg = (reg_grp[reg_grp['status_group']=='non functional']
                 .set_index('region')['pct'].sort_values(ascending=True))
    fig5 = px.bar(x=nf_by_reg.values, y=[r.title() for r in nf_by_reg.index],
                  orientation='h',
                  color=nf_by_reg.values,
                  color_continuous_scale=[[0,'#28a745'],[0.5,'#ffc107'],[1,'#dc3545']],
                  labels={'x':'% Non-functional','y':'Region'},
                  text=nf_by_reg.values.round(1), height=600)
    fig5.update_traces(texttemplate='%{x:.0f}%', textposition='outside')
    fig5.update_layout(coloraxis_showscale=False, xaxis_range=[0,80])
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**Takeaway:** Some regions have 50%+ non-functional rates — geo-targeted maintenance would have the highest impact.")

# =============================================================
# PAGE: MODEL PERFORMANCE
# =============================================================
elif page == "Model Performance":
    st.subheader("Model Performance — Validation Set (11,880 pumps)")
    st.caption("80/20 stratified train/validation split. Both models trained with SMOTE to improve minority-class recall.")

    with st.spinner("Computing validation predictions..."):
        cm_xgb, cm_rf, y_val, xgb_p, rf_p = compute_val_metrics()

    # ---- Accuracy / summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost Accuracy", f"{(y_val == xgb_p).mean()*100:.1f}%")
    c2.metric("RF Accuracy",      f"{(y_val == rf_p).mean()*100:.1f}%")
    c3.metric("XGBoost Repair Recall",
              f"{cm_xgb[1,1]/cm_xgb[1,:].sum()*100:.0f}%",
              help="% of 'needs repair' pumps correctly caught")
    c4.metric("RF Repair Recall",
              f"{cm_rf[1,1]/cm_rf[1,:].sum()*100:.0f}%")

    st.divider()

    # ---- Confusion Matrices ----
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    labels = le.classes_
    short  = ['Functional', 'Needs Repair', 'Non-Func.']

    def conf_matrix_fig(cm, title):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        text = [[f"{cm[i,j]:,}<br>({pct[i,j]:.0f}%)" for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(
            z=cm, x=short, y=short,
            colorscale='Blues', showscale=False,
            text=text, texttemplate="%{text}",
            textfont=dict(size=13)
        ))
        fig.update_layout(title=title, height=380,
                          xaxis_title='Predicted', yaxis_title='Actual',
                          yaxis=dict(autorange='reversed'))
        return fig

    with col1:
        st.plotly_chart(conf_matrix_fig(cm_xgb, "XGBoost + SMOTE"), use_container_width=True)
    with col2:
        st.plotly_chart(conf_matrix_fig(cm_rf, "Random Forest + SMOTE"), use_container_width=True)

    st.caption(
        "Diagonal = correct predictions. The 'Needs Repair' row is the hardest — "
        "only 7.3% of data. SMOTE nearly doubles repair recall vs non-balanced models."
    )

    st.divider()

    # ---- Per-class metrics ----
    st.markdown("### Per-Class Metrics (Precision / Recall / F1)")

    from sklearn.metrics import precision_recall_fscore_support
    p_xgb, r_xgb, f_xgb, _ = precision_recall_fscore_support(y_val, xgb_p)
    p_rf,  r_rf,  f_rf,  _ = precision_recall_fscore_support(y_val, rf_p)

    metrics_df = pd.DataFrame({
        'Class': labels,
        'XGBoost Precision': p_xgb.round(2), 'XGBoost Recall': r_xgb.round(2), 'XGBoost F1': f_xgb.round(2),
        'RF Precision':      p_rf.round(2),  'RF Recall':      r_rf.round(2),  'RF F1':      f_rf.round(2),
    })

    col1, col2 = st.columns(2)
    for col, model, p, r, f in [
        (col1, "XGBoost + SMOTE", p_xgb, r_xgb, f_xgb),
        (col2, "RF + SMOTE",      p_rf,  r_rf,  f_rf),
    ]:
        chart_df = pd.DataFrame({
            'Class': list(labels)*3,
            'Metric': ['Precision']*3 + ['Recall']*3 + ['F1']*3,
            'Score': list(p) + list(r) + list(f),
        })
        fig_m = px.bar(chart_df, x='Class', y='Score', color='Metric', barmode='group',
                       range_y=[0, 1.05], height=350,
                       color_discrete_sequence=['#4e79a7','#f28e2b','#59a14f'],
                       title=model, text='Score')
        fig_m.update_traces(texttemplate='%{y:.2f}', textposition='outside', textfont_size=10)
        with col:
            st.plotly_chart(fig_m, use_container_width=True)

    st.divider()
    st.markdown("### Why We Prioritise Repair Recall")
    st.info("""
    **Business logic**: Missing a non-functional pump means a community without water.
    Missing a "needs repair" pump means it breaks completely soon.
    A false positive (sending a technician to a working pump) costs far less than a false negative.

    → SMOTE increases repair recall from ~0.25 (raw XGBoost) to **0.48**, catching nearly twice as many at-risk pumps.
    """)

# =============================================================
# PAGE: SINGLE PUMP CHECK
# =============================================================
elif page == "Single Pump Check":
    st.subheader("Check Individual Pump Status")
    st.caption("Fill in the pump details. Both models predict independently — agreement means higher confidence.")

    row = pump_input_form(key_prefix="single_")

    if st.button("🔍 Predict Pump Status", key="single_predict"):
        try:
            xgb_labels, xgb_probas, rf_labels, rf_probas = predict(row)
            st.divider()
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2)
            with c1:
                show_result(xgb_labels[0], xgb_probas[0], "XGBoost + SMOTE")
            with c2:
                show_result(rf_labels[0],  rf_probas[0],  "Random Forest + SMOTE")
            st.divider()
            if xgb_labels[0] == rf_labels[0]:
                v = xgb_labels[0].lower()
                msg = f"**Both models agree: {xgb_labels[0].upper()}**"
                if v == 'functional needs repair':
                    st.warning(msg + "  \nSchedule maintenance soon.")
                elif v == 'non functional':
                    st.error(msg + "  \nRequires immediate field inspection.")
                else:
                    st.success(msg + "  \nContinue routine checks.")
            else:
                st.warning(f"**Models disagree** — XGBoost: *{xgb_labels[0]}* | RF: *{rf_labels[0]}*  \n"
                           "Recommend field inspection to confirm.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# =============================================================
# PAGE: SHAP EXPLAINER
# =============================================================
elif page == "SHAP Explainer":
    st.subheader("SHAP Feature Explainer")
    st.caption("SHAP shows **which features drove the prediction** and by how much. "
               "Red = pushed toward the class. Blue = pushed away.")

    with st.expander("How to read waterfall plots", expanded=False):
        st.markdown("""
        - **Base value** — model's average prediction across all training pumps
        - Each bar = how much one feature shifted the probability
        - **Red** = pushed prediction **toward** the selected class
        - **Blue** = pushed prediction **away from** the selected class
        - Final value = the model's actual output for this pump
        """)

    row = pump_input_form(key_prefix="shap_")

    if st.button("🔬 Explain This Pump", key="shap_explain"):
        try:
            X = preprocess(row)
            xgb_preds  = xgb_model.predict(X);  xgb_probas = xgb_model.predict_proba(X)
            rf_preds   = rf_model.predict(X);   rf_probas  = rf_model.predict_proba(X)
            xgb_label  = le.inverse_transform(xgb_preds)[0]
            rf_label   = le.inverse_transform(rf_preds)[0]

            st.divider()
            c1, c2 = st.columns(2)
            with c1: show_result(xgb_label, xgb_probas[0], "XGBoost + SMOTE")
            with c2: show_result(rf_label,  rf_probas[0],  "Random Forest + SMOTE")
            st.divider()

            with st.spinner("Computing SHAP values..."):
                sv_xgb = explainer_xgb(X)
                sv_rf  = explainer_rf(X)

            selected_class = st.selectbox("Show SHAP for class:", CLASS_NAMES,
                index=int(xgb_preds[0]),
                help="Default = class XGBoost predicted.")
            sel_idx = CLASS_NAMES.index(selected_class)

            st.markdown(f"#### Waterfall plots — **{selected_class}**")
            c1, c2 = st.columns(2)
            with c1: shap_waterfall(sv_xgb, sel_idx, X[0], "XGBoost + SMOTE", selected_class)
            with c2: shap_waterfall(sv_rf,  sel_idx, X[0], "Random Forest + SMOTE", selected_class)

            st.markdown("#### Top feature contributions")
            c1, c2 = st.columns(2)
            with c1: shap_bar(sv_xgb, sel_idx, "XGBoost + SMOTE")
            with c2: shap_bar(sv_rf,  sel_idx, "Random Forest + SMOTE")

            st.markdown("#### Full SHAP value table")
            names = [FEATURE_LABELS.get(f, f) for f in TIER1_FEATURES]
            tbl = pd.DataFrame({
                'Feature': names,
                'Encoded Value': X[0].round(3),
                'SHAP (XGBoost)': sv_xgb.values[0, :, sel_idx].round(4),
                'SHAP (RF)': sv_rf.values[0, :, sel_idx].round(4),
            })
            tbl['Avg |SHAP|'] = ((tbl['SHAP (XGBoost)'].abs() + tbl['SHAP (RF)'].abs()) / 2).round(4)
            st.dataframe(tbl.sort_values('Avg |SHAP|', ascending=False).reset_index(drop=True),
                         use_container_width=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")

# =============================================================
# PAGE: BATCH ANALYSIS
# =============================================================
elif page == "Batch Analysis":
    st.subheader("Batch Analysis — Upload CSV")
    st.info("Upload a CSV with pump data. Missing columns are auto-filled with sensible defaults.")

    with st.expander("Expected columns"):
        st.markdown("""
        | Column | Example |
        |--------|---------|
        | `quantity` | enough, insufficient, dry, seasonal |
        | `water_quality` | soft, salty, milky, coloured |
        | `extraction_type_group` | gravity, hand pump, motorpump |
        | `waterpoint_type` | communal standpipe, hand pump |
        | `source` | spring, shallow well, river |
        | `payment` | never pay, per bucket, monthly |
        | `region` | Arusha, Mwanza, Dodoma |
        | `lga` | District name |
        | `longitude` / `latitude` | GPS coordinates |
        | `gps_height` | Altitude in metres |
        | `pump_age` | Years since installation |
        | `population` | People served |
        | `management` | vwc, wug, company |
        | `installer` | dwe, government, ngo |
        | `funder` | government, world bank |
        """)

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} pumps")
        with st.expander("Preview"):
            st.dataframe(df.head())
        if st.button("🔍 Analyse All Pumps"):
            with st.spinner("Predicting..."):
                try:
                    xgb_labels, xgb_probas, rf_labels, rf_probas = predict(df)
                    df['xgb_prediction'] = xgb_labels;  df['xgb_confidence'] = xgb_probas.max(axis=1).round(4)
                    df['rf_prediction']  = rf_labels;   df['rf_confidence']  = rf_probas.max(axis=1).round(4)
                    df['models_agree']   = df['xgb_prediction'] == df['rf_prediction']
                    df['majority_vote']  = df[['xgb_prediction','rf_prediction']].apply(
                        lambda r: Counter(r).most_common(1)[0][0], axis=1)

                    st.divider()
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("✅ Functional", (df['majority_vote']=='functional').sum())
                    c2.metric("⚠️ Needs Repair", (df['majority_vote']=='functional needs repair').sum())
                    c3.metric("❌ Non-functional", (df['majority_vote']=='non functional').sum())
                    c4.metric("Both Agree", df['models_agree'].sum())
                    st.info(f"Models agreed on **{df['models_agree'].mean()*100:.1f}%** of pumps.")

                    st.divider()
                    priority = df[df['majority_vote'].isin(['functional needs repair','non functional'])].copy()
                    priority = priority.sort_values('xgb_confidence', ascending=False)
                    display_cols = [c for c in ['id','majority_vote','models_agree',
                        'xgb_prediction','xgb_confidence','rf_prediction','rf_confidence']
                        if c in priority.columns]
                    st.subheader("⚠️ Priority Pumps")
                    st.dataframe(priority[display_cols], use_container_width=True, height=400)

                    st.divider()
                    st.download_button("📥 Download Full Results", df.to_csv(index=False),
                        "pump_predictions.csv", "text/csv")
                    if len(priority):
                        st.download_button("📥 Download Priority Pumps Only",
                            priority.to_csv(index=False), "priority_pumps.csv", "text/csv", key="dl2")
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================
# PAGE: ABOUT
# =============================================================
elif page == "About":
    st.subheader("About Hydro Dominion")
    st.markdown("""
    ### Project
    **Hydro Dominion** is a machine learning project developed for the
    ReDI School Data Circle 2026. The goal is to predict the operational
    status of water pumps across Tanzania so that maintenance resources
    can be prioritised where they're needed most.

    ### Data
    - **Source:** [DrivenData — Pump it Up](https://www.drivendata.org/competitions/7/)
    - **Size:** 59,400 water pumps, 41 raw features → 15 selected
    - **Classes:** Functional (54.3%) | Needs Repair (7.3%) | Non-Functional (38.4%)

    ### Models

    | Model | Accuracy | Repair Recall | Repair F1 |
    |-------|----------|--------------|-----------|
    | XGBoost + SMOTE | 78.8% | 0.48 | 0.43 |
    | Random Forest + SMOTE | 78.6% | 0.43 | 0.42 |

    ### Pipeline
    1. Feature engineering (pump_age, population_bin, payment_binary)
    2. OrdinalEncoder — fitted on training data
    3. Top 15 features by Mutual Information score
    4. SMOTE oversampling (7.3% minority → balanced)
    5. RandomizedSearchCV tuning (f1_macro, 5-fold CV)

    ### Why SMOTE?
    Without SMOTE, models see only 3,454 "needs repair" examples vs 32,259 others.
    SMOTE synthesises new minority samples to reach 25,807 — matching the majority class.
    Repair recall improves from ~0.25 to 0.48 with no meaningful accuracy loss.

    ### SHAP Explainability
    The SHAP Explainer page uses **TreeSHAP** (exact, fast) on both models.
    Waterfall plots show per-feature contributions for any individual pump you enter.

    ### Disclaimer
    Decision support tool only. Always verify with field inspection before action.
    """)

# ===== FOOTER =====
st.divider()
st.caption("Hydro Dominion | ReDI School Data Circle 2026 | XGBoost + RF with SMOTE | SHAP Explainability")
