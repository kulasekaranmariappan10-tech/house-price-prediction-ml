# House Price Prediction – End-to-End Machine Learning Project

A beginner-friendly regression project using the **California Housing** dataset, a real public dataset derived from 1990 U.S. census data.

## What this project does
- Loads and inspects a real public housing dataset.
- Separates input features (`X`) and target (`y`).
- Splits data into training and test sets.
- Trains Linear Regression and Random Forest Regression models.
- Uses preprocessing pipelines and 5-fold cross-validation.
- Compares models using MAE, RMSE and R².
- Saves the best model with Joblib.
- Provides a Streamlit web app for interactive predictions.

## Important dataset limitation
This dataset is based on California census information from 1990. Its target (`MedHouseVal`) is expressed in units of **$100,000** and was capped in the original dataset. It is not current market data and is not suitable for valuing homes in India, Chennai, or any real property transaction. Predictions are for learning only.

## 1. Requirements
- Python 3.10 or newer recommended
- Internet connection for the first dataset download
- VS Code or another Python editor

## 2. Setup (Windows PowerShell)
Open the project folder in VS Code Terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run commands with `python -m pip ...` without activating the environment.

## 3. Train and evaluate
```powershell
python train_model.py
```

The first run downloads the dataset through scikit-learn and may take a little time. It then trains both models and saves:
- `models/house_price_model.joblib`
- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/actual_vs_predicted.png`
- `outputs/california_housing_sample.csv`

## 4. Launch the web app
After training:
```powershell
streamlit run app.py
```
Open the local address printed in the terminal (usually `http://localhost:8501`).

## Dataset columns
- `MedInc`: median income in the block group (10,000 USD units)
- `HouseAge`: median house age
- `AveRooms`: average rooms per household
- `AveBedrms`: average bedrooms per household
- `Population`: block-group population
- `AveOccup`: average household occupancy
- `Latitude`, `Longitude`: geographic coordinates
- `MedHouseVal`: target median house value (100,000 USD units)

## ML concepts demonstrated
- Regression
- Feature / target separation
- Train/test split
- Missing-value imputation
- Standardization (Linear Regression pipeline)
- Pipeline-based leakage prevention
- Cross-validation
- Model comparison
- MAE, RMSE, R²
- Model persistence and inference

## Project structure
```text
house_price_prediction_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── models/
└── outputs/
```
