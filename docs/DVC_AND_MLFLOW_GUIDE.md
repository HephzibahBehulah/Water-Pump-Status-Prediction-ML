# DVC & MLflow Guide for Water Pump Prediction Project

## Table of Contents
1. [Overview](#overview)
2. [DVC Setup](#dvc-setup)
3. [MLflow Setup](#mlflow-setup)
4. [Complete Workflow](#complete-workflow)
5. [Common Commands](#common-commands)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### What is DVC?
**Data Version Control** (DVC) tracks large files and pipelines without storing them in Git.

**Why we use it:**
- Version control for datasets and models
- Reproducible ML pipelines
- Automatic dependency management
- Integration with cloud storage (S3, Azure, GCS)

### What is MLflow?
**MLflow** tracks experiments, parameters, metrics, and manages model lifecycle.

**Why we use it:**
- Experiment tracking (see all training runs)
- Model registry (staging → production)
- REST API deployment
- Full reproducibility (code snapshot, dependencies)

---

## DVC Setup

### 1. Initialize DVC (Already Done)
```bash
# Navigate to project directory
cd water-pump-prediction

# DVC is already initialized (check .dvc folder)
# If not, run: dvc init
```

### 2. Track Data Files

Your raw data files should be tracked by DVC, not Git:

```bash
# Add training data to DVC
dvc add data/raw/training_values.csv
dvc add data/raw/training_labels.csv
dvc add data/raw/test_values.csv

# This creates .dvc files (commit to Git)
# Actual CSVs are ignored by Git
```

**What this does:**
- Creates `training_values.csv.dvc` (tiny file, goes to Git)
- `.gitignore` automatically prevents CSV from being committed
- DVC tracks the actual file separately

### 3. Fetch Data (Team Members)

When others clone the repo:
```bash
git clone <repo>
cd water-pump-prediction

# Get all tracked files (data, models)
dvc pull

# Now you have the actual data files locally
```

### 4. Configure Remote Storage (Optional - for Team Sharing)

```bash
# For local NAS/network drive:
dvc remote add -d myremote /mnt/nas/ml-projects/dvc-storage

# For AWS S3:
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote profile default

# For Google Cloud:
dvc remote add -d myremote gs://my-bucket/dvc-storage

# View remote config:
dvc remote list
dvc remote list -v
```

### 5. Push Data to Remote

```bash
# After training, save models and data
dvc add models/production/random_forest_latest.pkl
dvc push  # Upload to remote storage

# Or push specific stage:
dvc push data/
```

---

## MLflow Setup

### 1. Initialize MLflow (Local)

MLflow uses local SQLite database by default (no setup needed):

```bash
# Database is automatically created in:
# mlflow.db (or .mlruns/ for local file storage)
```

### 2. Start MLflow UI

**In Terminal/PowerShell:**
```bash
mlflow ui
```

**Then visit:**
```
http://localhost:5000
```

You'll see:
- ✅ All experiments
- ✅ All training runs
- ✅ Parameters, metrics, artifacts
- ✅ Model versions

### 3. Configure for Your Project

MLflow settings are in `config/production.yaml`:

```yaml
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: water-pump-production
  registry_uri: sqlite:///mlflow_registry.db
```

---

## Complete Workflow

### 🔄 Typical Day-to-Day Workflow

**Step 1: Update Code**
```bash
# You improve your model training script
# Edit: scripts/train_pipeline.py
```

**Step 2: Run Training Pipeline**
```bash
# Option A: Run single script (logs to MLflow)
python scripts/train_pipeline.py

# Option B: Run full DVC pipeline (recommended for production)
dvc repro
```

**Step 3: View Results in MLflow**
```bash
# Open MLflow UI
mlflow ui

# Visit http://localhost:5000
# See your new run in the experiment
```

**Step 4: Compare Experiments**
```
MLflow UI → Click on experiment
→ Select multiple runs
→ Click "Compare" button
→ See side-by-side metrics comparison
```

**Step 5: Register Best Model**

In MLflow UI:
1. Click on best run
2. Click "Register Model"
3. Enter model name: `water-pump-rf`
4. Create new model version

Or via CLI:
```bash
mlflow models register \
  -m "runs:/abc123/model" \
  -n "water-pump-rf"
```

**Step 6: Transition Model to Production**

In MLflow UI:
1. Go to "Models" tab
2. Click model name
3. Find best version
4. Change stage: Development → Production

Or via CLI:
```bash
mlflow models transition-stage \
  -m "water-pump-rf" \
  -v 1 \
  -s Production
```

**Step 7: Commit & Push**
```bash
# Commit code changes
git add src/ scripts/ config/
git commit -m "Improve RF model: 79.7% → 81.2% accuracy"

# Push data/models to DVC remote
dvc push

# Push to Git
git push
```

---

## Common Commands

### DVC Commands

```bash
# Initialize (already done)
dvc init

# Track a file/folder
dvc add data/raw/training_values.csv

# Run pipeline
dvc repro

# Check pipeline status
dvc dag  # Show pipeline graph
dvc status  # What changed?

# Push to remote
dvc push

# Pull from remote
dvc pull

# Remove tracking (but keep file)
dvc remove data/raw/training_values.csv.dvc

# View remote storage
dvc remote list -v
```

### MLflow Commands

```bash
# Start UI (local)
mlflow ui --port 5000

# View experiments
mlflow experiments list

# View runs in experiment
mlflow runs list --experiment-name "water-pump-production"

# Register model
mlflow models register -m "runs:/RUN_ID/model" -n "water-pump-rf"

# Serve model locally
mlflow models serve -m "models:/water-pump-rf/Production" --port 8000

# View model info
mlflow models describe water-pump-rf

# Get model URI
mlflow models get-latest-versions water-pump-rf
```

### Git + DVC Combined

```bash
# Normal Git workflow
git add src/ config/ scripts/
git commit -m "Update training script"

# Commit DVC changes
git add *.dvc .gitignore
git commit -m "Update DVC tracked files"

# Push DVC files
dvc push

# Push to GitHub
git push
```

---

## Example: Complete Training Run

### Scenario: You want to train a new model and track everything

**Step 1: Prepare environment**
```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data (if needed)
# https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/
# Place in data/raw/

# Track data with DVC
dvc add data/raw/training_values.csv
dvc add data/raw/training_labels.csv
```

**Step 2: Update code (optional)**
```python
# Edit config/production.yaml
# Change model hyperparameters if desired

# Or edit scripts/train_pipeline.py
# Add new feature engineering logic
```

**Step 3: Start MLflow UI (separate terminal)**
```bash
mlflow ui
# Visit http://localhost:5000
```

**Step 4: Run training**

**Option A - Single script:**
```bash
python scripts/train_pipeline.py
```

Output:
```
✅ Loaded 59400 training records
✅ Train/Val split: 47520 / 11880
✅ Training Random Forest...
   Train Accuracy: 0.8234
   Validation Accuracy: 0.7956
   Validation F1: 0.7945
💾 Model saved to models/production/random_forest_latest.pkl
📦 Model logged to MLflow
```

**Option B - Full pipeline with DVC:**
```bash
dvc repro
```

Output:
```
Reproducing pipeline...
Running stage 'train_model'...
✅ Training completed
✅ Metrics logged to metrics.json
```

**Step 5: Check results in MLflow**
```
Visit http://localhost:5000
→ Go to Experiments
→ Click "water-pump-production"
→ See your new run
→ View metrics, parameters, artifacts
```

**Step 6: Save metrics and commit**
```bash
# Push models to DVC remote (if configured)
dvc push

# Commit changes
git add config/ scripts/ dvc.yaml dvc.lock
git commit -m "Train RF model v2: 79.56% accuracy"

git add *.dvc
git commit -m "Update DVC tracked artifacts"

git push
git push dvc  # Push DVC files
```

---

## Troubleshooting

### Problem: "MLflow tracking URI not found"
**Solution:**
```bash
# Make sure MLflow is running
mlflow ui

# Check in a separate terminal
# Should show: WARNING: This is a development server...
```

### Problem: "dvc.yaml not found"
**Solution:**
```bash
# Check if in project root
ls dvc.yaml

# Reinitialize if needed
dvc init --no-scm
```

### Problem: "Model file too large for Git"
**Solution:**
```bash
# Remove from Git history
git rm --cached models/production/*.pkl

# Track with DVC instead
dvc add models/production/random_forest_latest.pkl

# Commit DVC file
git add models/production/*.dvc
git commit -m "Track models with DVC"
```

### Problem: "DVC pull not getting latest data"
**Solution:**
```bash
# Check remote config
dvc remote list -v

# Force refresh
dvc fetch --force
dvc checkout
```

### Problem: "Can't connect to MLflow database"
**Solution:**
```bash
# Reset MLflow (careful - loses history)
rm -rf mlruns/
rm mlflow.db

# Restart UI
mlflow ui
```

---

## Best Practices

✅ **Do:**
- Track all large files (>10MB) with DVC
- Run `dvc repro` for reproducibility
- Log parameters & metrics in every training run
- Commit `dvc.lock` to show exact pipeline state
- Use descriptive commit messages

❌ **Don't:**
- Commit large model files to Git
- Manually copy model files around
- Skip MLflow logging
- Delete `.dvc` files without understanding
- Push credentials to Git (use .env)

---

## Next Steps

1. ✅ Set up remote storage (NAS/S3) for team sharing
2. ✅ Create CI/CD pipeline (GitHub Actions) to auto-train models
3. ✅ Set up model deployment with MLflow serving
4. ✅ Create monitoring dashboard for production models

---

**Questions?** Check:
- [DVC Docs](https://dvc.org/doc)
- [MLflow Docs](https://www.mlflow.org/docs/)
- Project README
