# 🚀 Quick Start Guide

Get up and running with Water Pump Prediction ML project in 5 minutes!

## Prerequisites
- Python 3.8+
- Git
- (Optional) Make installed

## 1️⃣ Clone & Setup

```bash
# Clone repository
git clone <your-repo-url>
cd water-pump-prediction

# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Prepare Data

Download data from [DrivenData Competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/):
- `Training Set Values.csv`
- `Training Set Labels.csv`
- `Test Set Values.csv`

Place in `data/raw/`:
```
data/raw/
├── training_values.csv
├── training_labels.csv
└── test_values.csv
```

Track with DVC:
```bash
dvc add data/raw/training_values.csv
dvc add data/raw/training_labels.csv
```

## 3️⃣ Start MLflow UI (Optional - for tracking)

**Terminal 1:**
```bash
mlflow ui
# Visit http://localhost:5000
```

## 4️⃣ Train Model

**Terminal 2:**
```bash
# Option 1: Quick training
python scripts/train_pipeline.py

# Option 2: Using Make
make train
```

**Output:**
```
✅ Loaded 59400 training records
✅ Training Random Forest...
   Train Accuracy: 0.8234
   Validation Accuracy: 0.7956
💾 Model saved to models/production/random_forest_latest.pkl
📦 Model logged to MLflow
```

## 5️⃣ View Results

- **MLflow UI:** http://localhost:5000
  - See all training runs
  - Compare metrics
  - Register models

- **Metrics file:** `metrics.json`
  ```bash
  cat metrics.json
  ```

---

## Common Commands (with Make)

```bash
# View all available commands
make help

# Training
make train          # Train model
make pipeline       # Run DVC pipeline
make mlflow-ui      # Start MLflow

# Code quality
make lint           # Check code style
make format         # Auto-format code
make test           # Run tests

# DVC
make dvc-status     # Check pipeline status
make dvc-dag        # View pipeline
make dvc-pull       # Get data from storage
make dvc-push       # Save data to storage

# Cleanup
make clean          # Remove cache/logs
```

---

## Common Commands (without Make)

```bash
# Training
python scripts/train_pipeline.py

# MLflow
mlflow ui

# DVC
dvc repro           # Run pipeline
dvc status          # Check status
dvc dag             # View DAG
dvc pull            # Pull data
dvc push            # Push data

# Testing
pytest tests/ -v

# Code quality
flake8 src/
black src/ --check
isort src/ --check
```

---

## Project Structure

```
water-pump-prediction/
├── src/                      # Source code
│   ├── models/              # Model training code
│   ├── app/                 # Streamlit app
│   └── utils/               # Helper functions
│
├── scripts/                 # Training scripts
│   └── train_pipeline.py   # Main training script
│
├── data/
│   ├── raw/                # Original data (tracked by DVC)
│   └── processed/          # Processed data
│
├── models/
│   ├── production/         # Production models (tracked by DVC)
│   └── staging/            # Models in testing
│
├── config/
│   ├── development.yaml    # Dev configuration
│   └── production.yaml     # Production configuration
│
├── tests/                  # Unit tests
├── docs/                   # Documentation
│
├── dvc.yaml               # DVC pipeline definition
├── requirements.txt       # Python dependencies
├── Makefile              # Common commands
└── README.md             # Project documentation
```

---

## Next Steps

1. **Read the full guides:**
   - [DVC & MLflow Guide](./DVC_AND_MLFLOW_GUIDE.md) - Detailed explanation
   - [Model Card](./MODEL_CARD.md) - Model details
   - [Architecture](./ARCHITECTURE.md) - System design

2. **Modify and experiment:**
   - Edit `config/development.yaml` to change hyperparameters
   - Modify `scripts/train_pipeline.py` for custom training
   - Edit `src/models/pipeline.py` to change model logic

3. **Collaborate:**
   - Push code to Git: `git push`
   - Push data to DVC: `dvc push`
   - Share MLflow runs with team

4. **Deploy:**
   - Run Streamlit app: `streamlit run src/app/main.py`
   - Deploy with MLflow: `mlflow models serve`

---

## Troubleshooting

**❌ "ModuleNotFoundError: No module named 'src'"**
```bash
# Solution: Install package in development mode
pip install -e .
```

**❌ "MLflow UI not accessible"**
```bash
# Solution: Make sure MLflow is running
# Terminal: mlflow ui
# Browser: http://localhost:5000
```

**❌ "DVC file not found"**
```bash
# Solution: Initialize DVC
dvc init
dvc add data/raw/training_values.csv
```

**❌ "Data files missing"**
```bash
# Solution: Download from competition and place in data/raw/
# Or pull from DVC remote: dvc pull
```

---

## Getting Help

- 📖 [Full DVC & MLflow Guide](./DVC_AND_MLFLOW_GUIDE.md)
- 🔗 [DVC Documentation](https://dvc.org/doc)
- 🔗 [MLflow Documentation](https://www.mlflow.org/docs/)
- 💬 [GitHub Issues](https://github.com/your-org/water-pump-prediction/issues)

---

**Ready to train?** Run:
```bash
python scripts/train_pipeline.py
```

Happy coding! 🎉
