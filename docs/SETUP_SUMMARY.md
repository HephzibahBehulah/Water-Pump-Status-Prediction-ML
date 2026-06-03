# ✅ DVC & MLflow Setup Complete!

This document summarizes what has been set up for your Water Pump Prediction project.

---

## 📦 What Was Installed

✅ **DVC (Data Version Control)**
- For versioning large data files and models
- Pipeline definition (`dvc.yaml`)

✅ **MLflow**
- For experiment tracking
- Model registry and management

✅ **Project Structure**
- Production-ready directory layout
- Separate source code, scripts, tests, docs

✅ **Configuration Files**
- `config/production.yaml` - Production settings
- `config/development.yaml` - Development settings
- `.env.example` - Environment template
- `dvc.yaml` - DVC pipeline definition

✅ **Utility Modules**
- Logger setup (`src/utils/logger.py`)
- Config loader (`src/utils/config.py`)
- Model pipeline (`src/models/pipeline.py`)

✅ **Training Script**
- `scripts/train_pipeline.py` - Complete training with MLflow integration

✅ **Documentation**
- DVC & MLflow Guide (`docs/DVC_AND_MLFLOW_GUIDE.md`)
- Quick Start Guide (`docs/QUICKSTART.md`)
- Setup Summary (this file)

✅ **Helper Tools**
- Makefile for common commands
- `.gitignore` configured for DVC/MLflow

---

## 🎯 Next Steps: What to Do Now

### Step 1: Create `.env` file (Optional but Recommended)

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
ENVIRONMENT=development
DEBUG=True
MLFLOW_EXPERIMENT_NAME=water-pump-development
```

### Step 2: Download Data

1. Go to [DrivenData Competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
2. Download these files:
   - Training Set Values.csv
   - Training Set Labels.csv
   - Test Set Values.csv
3. Place in `data/raw/` folder

### Step 3: Track Data with DVC

```bash
cd data/raw/
dvc add training_values.csv
dvc add training_labels.csv
dvc add test_values.csv
```

This creates `.dvc` files (tiny, go to Git) and adds CSVs to `.gitignore`.

### Step 4: Start MLflow UI (in separate terminal)

```bash
mlflow ui
```

Then visit: http://localhost:5000

### Step 5: Run Training

```bash
python scripts/train_pipeline.py
```

Or using Make:
```bash
make train
```

### Step 6: Check Results

- **MLflow UI:** http://localhost:5000
- **Metrics file:** `metrics.json`
- **Model saved:** `models/production/random_forest_latest.pkl`

---

## 📚 File Structure Created

```
water-pump-prediction/
│
├── src/                              # ← Production source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pipeline.py              # ← Model training logic
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # ← Streamlit app (to be migrated)
│   │   ├── pages/                   # ← Multi-page app
│   │   └── components/              # ← Reusable UI components
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                # ← Logging setup
│       └── config.py                # ← Config loader
│
├── scripts/
│   └── train_pipeline.py            # ← Main training script with MLflow
│
├── config/
│   ├── development.yaml             # ← Dev config (lighter models)
│   └── production.yaml              # ← Prod config (tuned models)
│
├── data/
│   ├── raw/                         # ← Tracked by DVC (CSV files)
│   │   └── .gitkeep
│   └── processed/                   # ← Processed data (tracked by DVC)
│       └── .gitkeep
│
├── models/
│   ├── production/                  # ← Production models (tracked by DVC)
│   │   └── .gitkeep
│   └── staging/                     # ← Models being tested
│       └── .gitkeep
│
├── notebooks/                       # ← Jupyter experiments (optional)
│   └── .gitkeep
│
├── tests/                           # ← Unit tests
│   ├── __init__.py
│   └── .gitkeep
│
├── docs/
│   ├── DVC_AND_MLFLOW_GUIDE.md     # ← Detailed guide (READ THIS!)
│   ├── QUICKSTART.md                # ← Get started quickly
│   └── SETUP_SUMMARY.md            # ← This file
│
├── .dvc/                            # ← DVC configuration (auto-created)
│   ├── config
│   └── .gitignore
│
├── dvc.yaml                         # ← DVC pipeline definition
├── .gitignore                       # ← Updated for DVC/MLflow
├── requirements.txt                 # ← Python dependencies
├── Makefile                         # ← Common commands
├── .env.example                     # ← Environment template
└── README.md                        # ← Project overview
```

---

## 🔄 Typical Workflow

### Day 1: Initial Setup
```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data and track with DVC
dvc add data/raw/training_values.csv

# 3. Start MLflow
mlflow ui  # In separate terminal

# 4. Train
python scripts/train_pipeline.py

# 5. Commit
git add src/ config/ scripts/ dvc.yaml
git commit -m "Initial setup with DVC+MLflow"
dvc push  # If configured
git push
```

### Day 2+: Experimentation
```bash
# 1. Make changes
# Edit: config/development.yaml
# Edit: scripts/train_pipeline.py

# 2. Train (automatically logs to MLflow)
python scripts/train_pipeline.py

# 3. View results in MLflow UI
# http://localhost:5000
# Compare runs, select best model

# 4. Commit changes
git add config/ scripts/
git commit -m "Improve features: 79.5% → 81.2% accuracy"

# 5. Sync data/models
dvc push
git push
```

---

## 📊 How DVC Works (Simple Explanation)

**Before DVC (Problem):**
```
❌ Try to commit model file to Git
❌ Git repository becomes huge (100MB+)
❌ Slow cloning, messy history
```

**With DVC (Solution):**
```
✅ Create training_values.csv
✅ Run: dvc add training_values.csv
✅ Creates: training_values.csv.dvc (1KB text file)
✅ Git commits the .dvc file only
✅ Actual CSV stored separately (local or cloud)
✅ Others run: dvc pull to get the actual file
```

**Visual:**
```
Git Repository:
├── training_values.csv.dvc    ← Git tracks this
├── model.pkl.dvc              ← Git tracks this
└── code, configs              ← Git tracks these

DVC Storage (Local/Cloud):
├── training_values.csv        ← DVC stores this
├── model.pkl                  ← DVC stores this
└── Other large files
```

---

## 📊 How MLflow Works (Simple Explanation)

**Without MLflow (Problem):**
```
❌ "What hyperparameters gave 79.7% accuracy?"
❌ Spreadsheet of results (error-prone)
❌ Manual model management
❌ Hard to reproduce
```

**With MLflow (Solution):**
```
✅ Every training run automatically tracked
✅ Parameters logged: n_estimators=200, max_depth=15
✅ Metrics logged: accuracy=0.797, f1=0.794
✅ Code snapshot saved with each run
✅ All artifacts saved (plots, confusion matrix)
✅ Model registry: Development → Staging → Production
```

**Visual:**
```
MLflow Experiment:
├── Run 1 (Nov 1)
│   ├── params: n_estimators=100
│   ├── metrics: accuracy=0.785
│   └── artifacts: model.pkl
│
├── Run 2 (Nov 2)  ← Best!
│   ├── params: n_estimators=200
│   ├── metrics: accuracy=0.797
│   └── artifacts: model.pkl
│
└── Run 3 (Nov 3)
    ├── params: n_estimators=300
    ├── metrics: accuracy=0.791
    └── artifacts: model.pkl
```

---

## 🚀 Key Commands to Remember

```bash
# Training
python scripts/train_pipeline.py          # Train with MLflow
dvc repro                                 # Run full pipeline

# MLflow
mlflow ui                                 # Start UI
mlflow models register -m <uri> -n <name> # Register model
mlflow models serve -m <uri> --port 8000  # Deploy

# DVC
dvc add <file>                           # Track file
dvc push                                 # Upload to storage
dvc pull                                 # Download from storage
dvc repro                                # Run pipeline
dvc dag                                  # View pipeline

# Git + DVC
git add .                                # Stage changes
git commit -m "message"                  # Commit
dvc push                                 # Push large files
git push                                 # Push code
```

---

## ⚠️ Common Mistakes to Avoid

❌ **Don't:** Commit large files to Git
```bash
# Wrong
git add models/production/*.pkl
```

✅ **Do:** Use DVC for large files
```bash
# Right
dvc add models/production/random_forest.pkl
git add models/production/*.pkl.dvc
```

---

❌ **Don't:** Forget to start MLflow
```bash
# Won't see any training runs
python scripts/train_pipeline.py
```

✅ **Do:** Start MLflow first
```bash
# Terminal 1
mlflow ui

# Terminal 2
python scripts/train_pipeline.py
```

---

❌ **Don't:** Hard-code paths
```python
# Bad
df = pd.read_csv("../../../data/training_values.csv")
```

✅ **Do:** Use config
```python
from src.utils import get_config
cfg = get_config()
df = pd.read_csv(cfg.get('data.raw_dir') + "/training_values.csv")
```

---

## 📖 Documentation to Read

1. **[DVC_AND_MLFLOW_GUIDE.md](./DVC_AND_MLFLOW_GUIDE.md)** ← Start here!
   - Detailed explanations
   - Complete workflows
   - Troubleshooting

2. **[QUICKSTART.md](./QUICKSTART.md)**
   - Get running in 5 minutes
   - Common commands

3. **[../README.md](../README.md)**
   - Project overview
   - Team info

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
✅ DVC initialized
[ ] ls .dvc/config

✅ Requirements installed
[ ] pip list | grep -E "dvc|mlflow"

✅ Config files exist
[ ] cat config/production.yaml

✅ Source code structure
[ ] ls -la src/

✅ Training script works
[ ] python scripts/train_pipeline.py --help

✅ Data files present
[ ] ls data/raw/ (should have CSV files)

✅ Models can be saved
[ ] mkdir -p models/production

✅ MLflow can be started
[ ] mlflow ui
[ ] Visit http://localhost:5000 in browser
```

---

## 🤝 Team Collaboration

Now that DVC + MLflow is set up:

**For New Team Members:**
```bash
git clone <repo>
dvc pull           # Get latest data/models
python scripts/train_pipeline.py  # Train locally
# See all experiments in MLflow UI
```

**For Sharing Results:**
```bash
# After training, your run is automatically in MLflow
# Share link: http://localhost:5000/experiments/1/runs/abc123

# To let others get your data
dvc push           # Push to shared storage
git push           # Push code
```

---

## 🎓 Learning Path

1. **Understand DVC:**
   - [DVC_AND_MLFLOW_GUIDE.md](./DVC_AND_MLFLOW_GUIDE.md) - DVC Section
   - [dvc.org/doc](https://dvc.org/doc)

2. **Understand MLflow:**
   - [DVC_AND_MLFLOW_GUIDE.md](./DVC_AND_MLFLOW_GUIDE.md) - MLflow Section
   - [mlflow.org/docs](https://www.mlflow.org/docs/)

3. **Experiment:**
   - Modify `config/development.yaml`
   - Run `python scripts/train_pipeline.py`
   - Observe results in MLflow UI

4. **Collaborate:**
   - `dvc push` to share data
   - `git push` to share code
   - Share MLflow run links

---

## ❓ Questions?

Check these files in order:
1. This file (SETUP_SUMMARY.md)
2. [QUICKSTART.md](./QUICKSTART.md)
3. [DVC_AND_MLFLOW_GUIDE.md](./DVC_AND_MLFLOW_GUIDE.md)
4. [../README.md](../README.md)

---

**You're all set! 🎉**

Start with:
```bash
python scripts/train_pipeline.py
```

Then visit MLflow UI at: http://localhost:5000
