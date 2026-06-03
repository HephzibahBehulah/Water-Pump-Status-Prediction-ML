"""
Water Pump Maintenance Detection App
Focuses on detecting pumps that need repair (minority class)
"""

from typing import Tuple, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger

logger = setup_logger(__name__)

# Define convert_to_str function (needed for unpickling preprocessor)
def convert_to_str(X: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to string for encoding"""
    return X.astype(str)

# Hardcoded feature lists (from training data)
DEFAULT_NUMERIC_FEATURES = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private',
                            'region_code', 'district_code', 'population', 'month_recorded', 'pump_age']
DEFAULT_CATEGORICAL_FEATURES = ['funder', 'installer', 'basin', 'region', 'lga', 'ward', 'public_meeting',
                                'scheme_management', 'permit', 'extraction_type', 'extraction_type_group',
                                'extraction_type_class', 'management', 'management_group', 'payment',
                                'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group',
                                'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group']

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Pump Maintenance Alert",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
<style>
    .high-risk { color: #ff4444; font-weight: bold; }
    .medium-risk { color: #ff8800; font-weight: bold; }
    .low-risk { color: #00aa00; font-weight: bold; }
    .metric-container { padding: 20px; background: #f0f2f6; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ===== TITLE & DESCRIPTION =====
st.title("🚨 Water Pump Maintenance Detector")
st.markdown("""
**Identify pumps that need urgent repair before they fail.**

This AI model focuses on detecting the critical minority class: **"Functional needs repair"**
- Detects ~50% of problem pumps (50% recall on minority class)
- Prioritizes maintenance decisions
- Reduces unexpected downtime
""")

# ===== LOAD MODEL & PREPROCESSOR =====
@st.cache_resource
def load_model_and_preprocessor():
    """Load trained SMOTE XGBoost model and preprocessor"""
    try:
        # Go up one directory from water_pump_app/ to project root
        base_path = Path(__file__).parent.parent
        model_path = base_path / "models/production/xgboost_smote.pkl"
        preprocessor_path = base_path / "models/production/preprocessor.pkl"
        selector_path = base_path / "models/production/selector.pkl"
        label_encoder_path = base_path / "models/production/label_encoder.pkl"
        numeric_features_path = base_path / "models/production/numeric_features.pkl"
        categorical_features_path = base_path / "models/production/categorical_features.pkl"

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        selector = joblib.load(selector_path)
        label_encoder = joblib.load(label_encoder_path)

        # Load feature names - use hardcoded defaults if not found
        try:
            numeric_features = joblib.load(numeric_features_path)
            categorical_features = joblib.load(categorical_features_path)
        except:
            numeric_features = DEFAULT_NUMERIC_FEATURES
            categorical_features = DEFAULT_CATEGORICAL_FEATURES

        # Validate that features were loaded
        if not numeric_features or not categorical_features:
            numeric_features = DEFAULT_NUMERIC_FEATURES
            categorical_features = DEFAULT_CATEGORICAL_FEATURES

        return model, preprocessor, selector, label_encoder, numeric_features, categorical_features
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, preprocessor, selector, label_encoder, numeric_features, categorical_features = load_model_and_preprocessor()

# ===== HELPER FUNCTIONS =====
def get_risk_level(probability: float) -> Tuple[str, str]:
    """Determine risk level based on repair probability

    Args:
        probability: Repair probability (0-1)

    Returns:
        Tuple of (risk_level, emoji)
    """
    if probability > 0.6:
        return "HIGH", "🚨"
    elif probability > 0.3:
        return "MEDIUM", "⚠️"
    else:
        return "LOW", "✅"

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """Preprocess features using saved preprocessor and selector"""
    try:
        # Transform through preprocessor
        X_proc: np.ndarray = preprocessor.transform(X)

        # Get top 15 features (same as training)
        if hasattr(selector, 'scores_'):
            feature_scores = selector.scores_
            top_k = min(15, len(feature_scores))
            top_indices = np.argsort(feature_scores)[-top_k:]
            X_selected = X_proc[:, top_indices]
        else:
            # If selector doesn't have scores, use first 15 features
            X_selected = X_proc[:, :min(15, X_proc.shape[1])]

        return X_selected
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
        raise

def predict_single_pump(features_dict):
    """Make prediction for a single pump"""
    # Convert to DataFrame with proper column names
    X = pd.DataFrame([features_dict])

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Preprocess
    X_processed = preprocess_features(X)

    # Predict
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]

    # Class 1 = "Functional needs repair"
    repair_probability = probabilities[1]
    status_names = ["Functional", "Needs Repair", "Non-functional"]
    predicted_status = status_names[prediction]

    return repair_probability, predicted_status

def predict_batch(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions for multiple pumps

    Args:
        df: Input dataframe with pump features

    Returns:
        Tuple of (predictions, repair_probabilities)
    """
    # Create a new dataframe with only required columns
    df_input: pd.DataFrame = pd.DataFrame(index=df.index)

    try:
        # Add numeric features
        for col in numeric_features:
            if col in df.columns:
                # Use existing column
                df_input[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # Add default value
                if col == 'pump_age':
                    df_input[col] = 5
                elif col == 'month_recorded':
                    df_input[col] = 6
                elif col == 'gps_height':
                    df_input[col] = 1000
                elif col == 'amount_tsh':
                    df_input[col] = 50000
                elif col == 'population':
                    df_input[col] = 1000
                elif col == 'region_code':
                    df_input[col] = 1
                elif col == 'district_code':
                    df_input[col] = 1
                elif col == 'num_private':
                    df_input[col] = 0
                else:
                    df_input[col] = 0

            # Fill any NaN values with 0
            df_input[col] = df_input[col].fillna(0)

        # Add categorical features
        for col in categorical_features:
            if col in df.columns:
                df_input[col] = df[col].astype(str).fillna('unknown')
            else:
                df_input[col] = 'unknown'

        # Verify all columns exist
        missing = set(list(numeric_features) + list(categorical_features)) - set(df_input.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
            raise ValueError(f"Could not create all required columns: {missing}")

        # Preprocess
        X_processed = preprocess_features(df_input)

        # Predict
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)

        # Extract repair probabilities (class 1)
        repair_probs = probabilities[:, 1]

        return predictions, repair_probs

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        st.error(f"Prediction error: {str(e)}")
        raise

# ===== MAIN APP =====
tab1, tab2, tab3 = st.tabs(["Single Pump Check", "Batch Analysis", "About Model"])

# ===== TAB 1: SINGLE PUMP =====
with tab1:
    st.subheader("Check Individual Pump Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        gps_height = st.number_input("GPS Height (m)", min_value=0, max_value=5000, value=1000, step=50)
        longitude = st.number_input("Longitude", min_value=-15.0, max_value=-10.0, value=-12.5, step=0.1)
        pump_age = st.number_input("Pump Age (years)", min_value=0, max_value=100, value=5)

    with col2:
        latitude = st.number_input("Latitude", min_value=-15.0, max_value=-10.0, value=-12.5, step=0.1)
        amount_tsh = st.number_input("Water Volume (TSH)", min_value=0, max_value=350000, value=50000, step=5000)
        waterpoint_type = st.selectbox("Waterpoint Type", ["hand pump", "mechanized", "solar"])

    with col3:
        basin = st.selectbox("Basin", ["Lake Tanganyika", "Lake Victoria", "Lake Malawi", "Other"])
        management = st.selectbox("Management", ["vwa", "wua", "private", "government"])
        installer = st.selectbox("Installer", ["dwe", "government", "ngo", "private"])

    if st.button("🔍 Analyze This Pump", key="single_pump"):
        try:
            # Create feature dict
            features = {
                'gps_height': gps_height,
                'longitude': longitude,
                'latitude': latitude,
                'amount_tsh': amount_tsh,
                'pump_age': pump_age,
                'waterpoint_type': waterpoint_type,
                'basin': basin,
                'management': management,
                'installer': installer
            }

            repair_prob, status = predict_single_pump(features)

            st.divider()
            st.subheader("Results")

            # Display risk level
            risk_level, emoji = get_risk_level(repair_prob)

            col1, col2, col3 = st.columns(3)

            with col1:
                if risk_level == "HIGH":
                    st.error(f"{emoji} HIGH PRIORITY - NEEDS REPAIR")
                elif risk_level == "MEDIUM":
                    st.warning(f"{emoji} MEDIUM - MONITOR CLOSELY")
                else:
                    st.success(f"{emoji} LOW RISK - OK")

            with col2:
                st.metric("Repair Probability", f"{repair_prob*100:.1f}%")

            with col3:
                st.metric("Status Prediction", status)

            # Interpretation
            st.subheader("Recommendation")
            if risk_level == "HIGH":
                st.error("""
                **This pump likely needs urgent repair!**
                - Schedule maintenance ASAP
                - Risk of complete failure: HIGH
                - Estimated time to failure: Days to weeks
                """)
            elif risk_level == "MEDIUM":
                st.warning("""
                **This pump should be monitored.**
                - Schedule maintenance within 1-2 weeks
                - Monitor performance closely
                - Keep spare parts on hand
                """)
            else:
                st.success("""
                **This pump appears to be functioning well.**
                - Continue routine maintenance
                - No urgent action needed
                - Schedule next check in 3-6 months
                """)

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            logger.error(f"Single pump prediction error: {str(e)}", exc_info=True)

# ===== TAB 2: BATCH ANALYSIS =====
with tab2:
    st.subheader("Analyze Multiple Pumps (Batch Upload)")

    with st.expander("📋 Data Format Help"):
        st.markdown("""
        ### Expected Columns
        The app expects a CSV with columns from the water pump dataset.

        **Don't worry about missing columns!** The app will:
        - ✅ Auto-fill missing numeric columns with smart defaults
        - ✅ Auto-fill missing categorical columns with "unknown"
        - ✅ Only use columns it recognizes

        ### Minimum to work:
        - `id` - Pump identifier
        - At least some feature columns (the more, the better!)

        ### If you have the training data CSV:
        Just upload it directly! The app will handle any missing columns.

        **Tip:** Numeric columns should have numeric values, categorical columns should have text values.
        """)

        if numeric_features and categorical_features:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Numeric Features ({len(numeric_features)}):**")
                for feat in numeric_features[:8]:
                    st.text(f"• {feat}")
                if len(numeric_features) > 8:
                    st.text(f"... and {len(numeric_features)-8} more")
            with col2:
                st.markdown(f"**Categorical Features ({len(categorical_features)}):**")
                for feat in categorical_features[:8]:
                    st.text(f"• {feat}")
                if len(categorical_features) > 8:
                    st.text(f"... and {len(categorical_features)-8} more")

    uploaded_file = st.file_uploader("Upload CSV file with pump data", type="csv", key="batch_upload")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.info(f"✅ Loaded {len(df)} pumps")

            # Show preview
            with st.expander("Preview data"):
                st.dataframe(df.head())

            if st.button("🔍 Analyze All Pumps", key="batch_analyze"):
                with st.spinner("Analyzing pumps..."):
                    st.info(f"""
                    **Processing Pipeline (same as training):**
                    - Original features: {len(numeric_features)} numeric + {len(categorical_features)} categorical = {len(numeric_features) + len(categorical_features)} total
                    - After preprocessing & feature selection: **15 top features** selected
                    - Model: XGBoost trained on these 15 features
                    """)
                    try:
                        # Make predictions
                        predictions, repair_probs = predict_batch(df)

                        # Map predictions to status names
                        status_names = {0: "Functional", 1: "Functional needs repair", 2: "Non-functional"}
                        df['predicted_status'] = [status_names[p] for p in predictions]
                        df['repair_probability'] = repair_probs

                        st.divider()
                        st.subheader("Analysis Results")

                        # Count by status
                        functional_count = len(df[df['predicted_status'] == 'Functional'])
                        repair_count = len(df[df['predicted_status'] == 'Functional needs repair'])
                        nonfunctional_count = len(df[df['predicted_status'] == 'Non-functional'])

                        # Display counts
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("✅ Functional", functional_count)
                        with col2:
                            st.metric("🚨 Needs Repair", repair_count)
                        with col3:
                            st.metric("❌ Non-functional", nonfunctional_count)
                        with col4:
                            st.metric("Total Analyzed", len(df))

                        # Repair percentage
                        st.divider()
                        repair_percentage = (repair_count / len(df)) * 100
                        st.info(f"**{repair_percentage:.1f}% of pumps need repair** ({repair_count} out of {len(df)})")

                        # Show pumps that need repair
                        st.divider()
                        st.subheader(f"🚨 Pumps Needing Repair ({repair_count})")

                        repair_pumps = df[df['predicted_status'] == 'Functional needs repair'].sort_values('repair_probability', ascending=False)

                        # Show only columns that exist in original CSV, plus our predictions
                        cols_to_display = ['id', 'repair_probability', 'predicted_status']
                        cols_to_display = [c for c in cols_to_display if c in repair_pumps.columns]

                        if len(repair_pumps) > 0:
                            display_df = repair_pumps[cols_to_display].copy().round(4)
                            st.dataframe(display_df, use_container_width=True, height=400)
                        else:
                            st.success("✅ No pumps need repair!")

                        # Download results
                        st.divider()
                        st.subheader("Download Results")

                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv,
                            file_name="pump_analysis_results.csv",
                            mime="text/csv"
                        )

                        # Download repair pumps only
                        if repair_count > 0:
                            repair_csv = repair_pumps.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Pumps Needing Repair (CSV)",
                                data=repair_csv,
                                file_name="pumps_needing_repair.csv",
                                mime="text/csv",
                                key="repair_download"
                            )

                    except Exception as e:
                        st.error(f"Error during batch analysis: {str(e)}")
                        logger.error(f"Batch analysis error: {str(e)}", exc_info=True)

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

# ===== TAB 3: ABOUT MODEL =====
with tab3:
    st.subheader("About This Model")

    st.markdown("""
    ### Model Details
    - **Algorithm**: XGBoost (Gradient Boosting)
    - **Training Data**: 59,400 water pumps from Tanzania
    - **Class Balance**: SMOTE applied (synthetic oversampling)
    - **Features**: 15 most important features selected

    ### Performance (on minority class detection)
    - **Recall**: 49.71% - Catches ~50% of pumps needing repair
    - **Precision**: 39.11% - 39% of predicted repairs are correct
    - **F1 Macro**: 0.407 - Fair metric across all classes

    ### What This Means
    ✅ **Strengths:**
    - Better than random guessing (7.3% baseline)
    - Prioritizes finding maintenance cases
    - Reduces unexpected pump failures

    ⚠️ **Limitations:**
    - Misses ~50% of problem pumps (false negatives)
    - Some false alarms (false positives)
    - Always combine with expert judgment

    ### How to Use
    1. **Single Pump**: Enter details and get instant risk assessment
    2. **Batch**: Upload CSV with pump data, get priority ranking
    3. **Action**: HIGH PRIORITY = maintenance within days

    ### Data Requirements (for Batch Upload)
    Required columns for batch analysis:
    - `id` - Pump identifier
    - `gps_height`, `longitude`, `latitude` - Location
    - `amount_tsh` - Water volume
    - `pump_age` - Age in years
    - Other categorical features (type, basin, management, etc.)

    ### Disclaimer
    ⚠️ This model is a **decision support tool**, not a replacement for expert judgment.
    Always verify predictions with field inspection before making maintenance decisions.
    """)

    st.divider()
    st.info("""
    📊 **Model Training Details**
    - Training samples: 47,520 pumps
    - Validation samples: 11,880 pumps
    - SMOTE balanced the minority class from 7.3% to 33.3% in training data
    - Evaluated on original (unbalanced) validation set
    - Best model: XGBoost with SMOTE balancing
    """)

# ===== FOOTER =====
st.divider()
st.markdown("""
---
**Built for**: Water Pump Maintenance Detection
**Model**: SMOTE + XGBoost
**Focus**: Detecting critical "Functional needs repair" cases
""")
st.sidebar.markdown("""
---
### Quick Tips
- 🚨 **HIGH**: Schedule repair within days
- ⚠️ **MEDIUM**: Monitor & plan maintenance
- ✅ **LOW**: Continue routine checks

### Model Performance
- Detects ~50% of problem pumps
- 39% precision on repairs
- Best for prioritizing maintenance
""")
