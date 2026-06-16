# Water Pump Maintenance Detection App

**🚨 AI-powered detection of water pumps that need urgent repair**

This Streamlit app uses a trained XGBoost model with SMOTE balancing to identify pumps that need maintenance, focusing on the critical "Functional needs repair" class.

## Features

### 1️⃣ Single Pump Check
- Enter details for one pump
- Get instant risk assessment (HIGH/MEDIUM/LOW)
- Get repair probability and status prediction
- Actionable maintenance recommendations

### 2️⃣ Batch Analysis
- Upload CSV file with multiple pumps
- Get priority ranking of all pumps
- Export HIGH PRIORITY pumps for immediate action
- Download full analysis results

### 3️⃣ Model Information
- Learn about model performance
- Understand limitations
- See how to use the tool effectively

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure model files exist:**
```
models/production/
├── xgboost_smote.pkl          # Trained model
├── preprocessor.pkl           # Feature preprocessor
├── selector.pkl               # Feature selector
└── label_encoder.pkl          # Class label encoder
```

## Running the App

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

## Usage

### Single Pump Mode
1. Enter pump details (location, age, water volume, type)
2. Click "Analyze This Pump"
3. Get risk level (🚨 HIGH / ⚠️ MEDIUM / ✅ LOW)
4. Follow recommendation for maintenance action

### Batch Mode
1. Prepare CSV with pump data
2. Upload the file
3. Click "Analyze All Pumps"
4. Review HIGH PRIORITY list
5. Download results for your team

## CSV Format for Batch Analysis

Required columns:
```
id,gps_height,longitude,latitude,amount_tsh,pump_age,waterpoint_type,basin,management,installer
1,1000,-12.5,-12.5,50000,5,hand pump,Lake Victoria,vwa,dwe
2,1500,-12.6,-12.6,75000,3,mechanized,Lake Tanganyika,wua,government
...
```

## Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **Recall** | 49.71% | Catches ~50% of problem pumps |
| **Precision** | 39.11% | 39% of alerts are correct |
| **F1 Macro** | 0.407 | Fair across all classes |
| **Accuracy** | 77.79% | Overall correctness |

## Important Notes

⚠️ **This is a decision-support tool, not ground truth**
- Always verify HIGH PRIORITY predictions with field inspection
- Combine AI results with expert judgment
- Use for maintenance planning, not absolute diagnosis

## Risk Levels

- 🚨 **HIGH (>60% repair probability)**: Schedule repair within DAYS
- ⚠️ **MEDIUM (30-60%)**: Schedule repair within WEEKS, monitor closely
- ✅ **LOW (<30%)**: Continue routine maintenance, no urgent action

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Troubleshooting

**"Model not found" error:**
- Ensure model files are in `models/production/`
- Run training scripts first if models don't exist

**"Column not found" error in batch mode:**
- Check CSV column names match expected format
- See CSV Format section above

**App runs slowly:**
- First load caches the model (normal)
- Subsequent predictions are instant
- Batch processing may take longer for large files (>1000 pumps)

## Next Steps

1. Start the app: `streamlit run app.py`
2. Test with single pump data
3. Upload test CSV with batch data
4. Review HIGH PRIORITY results
5. Integrate into maintenance workflow

---

**Built with**: Streamlit, XGBoost, SMOTE, scikit-learn
