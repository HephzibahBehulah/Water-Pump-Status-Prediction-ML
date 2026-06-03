# Virtual Environment Setup Guide

**Current Status:** ❌ NO VIRTUAL ENVIRONMENT DETECTED
**Python Location:** C:\Users\vaibh\AppData\Local\Programs\Python\Python312\python.exe
**Status:** Using global Python installation

---

## ❌ Why You Need a Virtual Environment

Without a virtual environment:
- ❌ Dependencies pollute global Python
- ❌ Version conflicts between projects
- ❌ Other projects break if you upgrade a package
- ❌ Not reproducible on other machines
- ❌ Not enterprise standard
- ❌ Not safe for production

**Enterprise Standard:** Every project has isolated dependencies ✅

---

## ✅ Setup Virtual Environment

### Option 1: Using Python's built-in `venv` (RECOMMENDED)

#### Step 1: Create Virtual Environment
```bash
# Navigate to project root
cd D:\ReDi\Project\Water-Pump-Status-Prediction-ML

# Create virtual environment
python -m venv venv
```

#### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

#### Step 3: Verify Activation
You should see `(venv)` prefix in your terminal:
```
(venv) D:\ReDi\Project\Water-Pump-Status-Prediction-ML>
```

#### Step 4: Install Dependencies
```bash
# Main dependencies
pip install -r requirements.txt

# App-specific dependencies
pip install -r water_pump_app/requirements.txt
```

#### Step 5: Verify Installation
```bash
pip list
```

You should see:
- pandas==2.0.3
- scikit-learn==1.3.0
- xgboost==1.7.5
- streamlit==1.28.0
- etc.

---

## 🔄 Daily Usage

### Start Working
```bash
# Activate venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate     # Mac/Linux

# You should see (venv) in terminal
(venv) D:\...>
```

### Run Application
```bash
# Make sure venv is activated
streamlit run water_pump_app/app.py
```

### Run Tests
```bash
# Install pytest if not already
pip install pytest

# Run tests
pytest tests/ -v
```

### Stop Working
```bash
deactivate
```

---

## 📋 Project Structure with venv

```
D:\ReDi\Project\Water-Pump-Status-Prediction-ML/
├── venv/                          # Virtual environment (NEW)
│   ├── Scripts/
│   ├── Lib/
│   ├── pyvenv.cfg
│   └── ...
├── src/
├── scripts/
├── water_pump_app/
├── tests/
├── requirements.txt
├── water_pump_app/requirements.txt
└── .gitignore                      # Should exclude venv/
```

---

## 🚫 Add to .gitignore

Make sure `.gitignore` includes venv:

```bash
# Virtual environments
venv/
env/
ENV/
.venv

# Compiled files
__pycache__/
*.pyc
*.pyo
*.egg-info/
```

---

## 🎯 Makefile Command (Optional)

Add to your `Makefile` for convenience:

```makefile
.PHONY: venv
venv:
	python -m venv venv
	.\venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	pip install -r water_pump_app/requirements.txt
	@echo "Virtual environment created and activated!"

.PHONY: activate
activate:
	@echo "Run: .\venv\Scripts\Activate.ps1"

.PHONY: deactivate
deactivate:
	deactivate
```

Then use:
```bash
make venv      # Create and setup
make activate  # Show activation command
```

---

## ✅ Verification Checklist

After setup:

- [ ] `venv/` folder created
- [ ] `(venv)` shows in terminal
- [ ] `pip list` shows correct versions
- [ ] `streamlit --version` works
- [ ] `pytest --version` works
- [ ] `venv/` is in `.gitignore`
- [ ] App runs: `streamlit run water_pump_app/app.py`

---

## 🆘 Troubleshooting

### Problem: `venv\Scripts\Activate.ps1` not found
**Solution:** Check path is correct
```bash
ls venv/Scripts/  # Should exist
```

### Problem: "Permission denied" error (PowerShell)
**Solution:** Enable script execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: `pip install` fails
**Solution:** Ensure venv is activated
```bash
which python  # Should show venv path
```

### Problem: Wrong Python version being used
**Solution:** Verify activation
```bash
python --version  # Should show 3.12
which python      # Should point to venv
```

---

## 📊 Benefits of Virtual Environment

| Aspect | Without venv | With venv |
|--------|--------------|-----------|
| **Isolation** | ❌ Global | ✅ Local |
| **Reproducibility** | ❌ Mixed versions | ✅ Exact versions |
| **Safety** | ❌ Conflicts | ✅ No conflicts |
| **Portability** | ❌ Machine-specific | ✅ Works everywhere |
| **Enterprise Standard** | ❌ Not recommended | ✅ Required |

---

## 🚀 Next Steps

1. **Create venv:**
   ```bash
   python -m venv venv
   ```

2. **Activate venv:**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r water_pump_app/requirements.txt
   ```

4. **Verify setup:**
   ```bash
   pip list
   ```

5. **Update .gitignore:**
   ```
   # Add to .gitignore
   venv/
   ```

6. **Start working:**
   ```bash
   streamlit run water_pump_app/app.py
   ```

---

## 💡 Pro Tips

### Always activate venv when starting work
Create a batch file for quick activation:

**activate.bat** (Windows):
```batch
@echo off
.\venv\Scripts\activate.bat
cmd /k
```

Run: `activate.bat`

### Use `requirements-dev.txt` for development
```
# requirements-dev.txt
-r requirements.txt
pytest==7.4.0
pytest-cov==4.1.0
black==23.9.1
flake8==6.1.0
```

Install for development:
```bash
pip install -r requirements-dev.txt
```

---

**Status:** Ready for setup! ✅
