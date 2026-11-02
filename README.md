# BurnoutSense AI – Employee Burnout Risk Prediction System

An end-to-end Machine Learning project that predicts employee burnout risk, explains individual predictions using SHAP, recommends HR actions, and generates human-readable explanations using an LLM. The system provides an interactive HR dashboard built with Streamlit and a REST API powered by FastAPI.

## Overview

Employee burnout can affect productivity, employee well-being, and organizational performance. This project helps HR teams identify employees who may be at risk using machine learning and explainable AI.

The system:

1. Analyzes historical employee data.
2. Predicts an employee's burnout risk score.
3. Classifies employees into Low, Medium, and High risk categories.
4. Identifies the top two factors influencing each prediction using SHAP.
5. Recommends suitable HR actions.
6. Generates human-readable explanations using an LLM.
7. Displays predictions and insights through an interactive HR dashboard.
8. Supports real-time predictions for newly added employees.

## Dataset

- **Source:** Kaggle – Employee Burnout Rate dataset.
- **Original size:** 22,750 records and 9 columns.
- **After cleaning:** 21,626 records.
- **Features:** Gender, Company Type, WFH Setup Available, Designation, Resource Allocation, Mental Fatigue Score, and engineered Days of Service.
- **Target:** Burn Rate (0–1 scale).

The original dataset is not included in this GitHub repository. Users must obtain it from the original dataset source and place it in the `data` folder.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
-- **Machine Learning:** Scikit-learn (Linear Regression, Random Forest Regressor, XGBoost Regressor)
- **Explainable AI:** SHAP
- **Dashboard:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Backend API:** FastAPI, Uvicorn
- **LLM Integration:** Groq API
- **Environment Management:** python-dotenv

## System Architecture and Methodology

### 1. Data Cleaning

File: `src/data_cleaning.py`

- Checks missing values and missingness patterns.
- Examines target-variable distribution and skewness.
- Performs correlation analysis.
- Uses median imputation for missing numerical feature values.
- Removes records with missing target values.

### 2. Feature Engineering

File: `src/feature_engineering.py`

- Encodes categorical variables.
- Creates the `Days_of_Service` feature from the joining date.
- Prepares feature sets for machine learning models.
- Applies scaling where required.

### 3. Model Training

File: `src/train_model.py`

Three machine learning algorithms are trained and compared:

- Linear Regression
- Random Forest
- XGBoost

The dataset is divided into training and testing sets using an 80:20 split.

### 4. Model Evaluation

File: `src/evaluate.py`

Models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- 5-Fold Cross-Validation

**XGBoost is selected as the final model based on its evaluation performance.**

| Metric | XGBoost |
|---|---:|
| Test R² Score | 0.9045 |
| Training R² Score | 0.9196 |
| MAE | 0.0473 |
| RMSE | 0.0603 |
| 5-Fold CV Mean R² | 0.9067 |
| CV Standard Deviation | 0.0036 |

### 5. Explainable AI – SHAP

File: `src/shap_explain.py`

- Uses SHAP TreeExplainer with the trained XGBoost model.
- Calculates feature contributions for individual predictions.
- Identifies the top two factors influencing each employee's predicted burnout risk.
- Helps HR users understand why a prediction was made.

### 6. HR Action Mapping

File: `src/action_mapping.py`

- Maps important risk-driving features to suggested HR actions.
- Combines the top two influencing features to generate relevant recommendations.
- Uses rule-based logic for consistent and understandable recommendations.

### 7. LLM Explanation Layer

File: `src/llm_reason.py`

- Uses the Groq API to generate natural-language explanations.
- Uses cached explanations to reduce repeated API calls.
- Combines explanations with employee-specific prediction information.
- Provides a deterministic fallback explanation if the API request fails.

### 8. Pipeline Orchestration

File: `src/pipeline.py`

Runs the complete machine learning workflow:

- Data cleaning
- Feature engineering
- Model training
- Model evaluation
- SHAP explanations
- HR action recommendations
- LLM-generated explanations

The pipeline generates:

- `data/burnout_predictions.csv`
- `models/burnout_model.pkl`

The trained model file stores the model and required preprocessing artifacts for future predictions.

### 9. Real-Time Employee Prediction

File: `src/inference.py`

- Loads the trained model.
- Accepts new employee information.
- Applies the required preprocessing.
- Predicts burnout risk.
- Generates SHAP explanations and HR recommendations.

New employees can be analyzed without retraining the model.

### 10. HR Dashboard

File: `src/dashboard.py`

The Streamlit dashboard includes:

- HR login and logout.
- Total employee count.
- High, Medium, and Low burnout risk summaries.
- Burnout risk distribution donut chart.
- Top risk-driving feature chart.
- Filterable and sortable employee records.
- Add New Employee form.
- Real-time burnout prediction.
- SHAP-based explanations and HR recommendations.

### 11. REST API

File: `src/api.py`

The FastAPI backend provides an API interface for accessing prediction functionality. It can be started separately from the Streamlit dashboard.

## Authentication

The application provides multi-user HR login.

- HR accounts are created using `src/manage_users.py`.
- Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes.
- Legacy SHA-256 password records are upgraded after successful login.
- Passwords are not stored as plain text.
- HR account records are stored locally in `data/hr_users.json`.

**Note:** This is a student portfolio project. The local JSON-based authentication system is not intended to replace enterprise identity management or production-grade access control.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/beenajayshrij-arch/BurnoutSense-AI.git
cd BurnoutSense-AI
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the original Employee Burnout Rate dataset from Kaggle.

Place the original CSV file in the project's `data` folder using the filename expected by the pipeline:

```text
data/employee_burnout.csv
```

Check the dataset source and licence before redistributing the data.

### 5. Configure the LLM API

If using LLM-generated explanations, configure your Groq API credentials in a local `.env` file according to the environment variable expected by `src/llm_reason.py`.

Never upload `.env` or API credentials to GitHub.

The application includes a fallback explanation mechanism if the LLM request fails.

### 6. Run the machine learning pipeline

```powershell
python src/pipeline.py
```

This generates the required prediction data and model artifacts.

### 7. Create an HR account

```powershell
python src/manage_users.py
```

Follow the prompts to create an HR username and password.

### 8. Launch the Streamlit dashboard

```powershell
python -m streamlit run src/dashboard.py
```

Open the local dashboard at:

`http://localhost:8501`

Log in using the HR credentials created earlier.

### 9. Launch the FastAPI backend

Open another PowerShell terminal in the project folder and activate the virtual environment.

Run:

```powershell
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

The API documentation is available at:

`http://127.0.0.1:8000/docs`

## Project Structure

```text
employee_burnout/
│
├── data/
│   └── (Dataset and generated files)
│
├── models/
│   └── burnout_model.pkl
│
├── src/
│   ├── action_mapping.py
│   ├── api.py
│   ├── check_models.py
│   ├── dashboard.py
│   ├── data_cleaning.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── generate_data.py
│   ├── inference.py
│   ├── llm_reason.py
│   ├── manage_users.py
│   ├── pipeline.py
│   ├── run_employee_report.py
│   ├── shap_explain.py
│   └── train_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Future Enhancements

- Role-based access control for different HR users.
- Enhanced employee trend analysis.
- More advanced burnout prediction models.
- Improved dashboard reporting and visualization.
- Production-ready authentication and deployment.
- Integration with additional HR analytics systems.

## Project Goal

The goal of BurnoutSense AI is to combine machine learning, explainable AI, and natural-language generation to make employee burnout risk analysis more understandable and actionable for HR teams.

**Disclaimer:** This project is an academic and portfolio demonstration. Predictions are estimates and should not be treated as medical diagnoses or used as the sole basis for employment decisions.
