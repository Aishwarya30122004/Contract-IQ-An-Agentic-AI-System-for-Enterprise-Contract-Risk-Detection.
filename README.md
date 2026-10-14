# ContractIQ — VS Code Starter Project

AI-assisted contract clause classification and risk review using TF-IDF and Logistic Regression.

## Features
- Upload PDF, DOCX, or TXT contracts
- Extract text from text-based PDFs and DOCX
- Split text into candidate clauses
- Predict clause type and risk level
- Display model confidence
- Export results to CSV
- Test a clause directly

## Dataset
The project expects `data/ContractIQ_clause_dataset.csv` with columns `clause_text`, `clause_type`, and `risk_level`. The supplied dataset is included if available.

## Run in VS Code (Windows)
1. Extract the ZIP and open its folder in VS Code.
2. In Terminal, run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal, usually `http://localhost:8501`.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Or use Command Prompt with `.venv\\Scripts\\activate.bat`.

## Push to GitHub
Create an empty repository on GitHub, then in the project folder run:
```bash
git init
git add .
git commit -m "Initial ContractIQ project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the remote URL with your actual repository URL.

## Important
Before pushing, ensure you have permission to publish the dataset. Do not commit real contracts, API keys, passwords, or confidential information. If the dataset is restricted, remove it from the repo and load it through an approved private method.

This is an educational prototype, not a production legal-review system. Validate predictions with qualified reviewers. Scanned PDFs require OCR, which is not included.
