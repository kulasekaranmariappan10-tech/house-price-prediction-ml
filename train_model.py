"""
House Price Prediction - End-to-end ML training
Dataset: California Housing (real census-derived public dataset).
Note: Target is in units of $100,000, not today's market price.
"""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def main():
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.to_csv(OUTPUT_DIR / "california_housing_sample.csv", index=False)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", df.isna().sum())

    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]  # measured in units of $100,000

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # All preprocessing is inside the pipeline to avoid data leakage.
    linear_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    forest_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    candidates = {
        "Linear Regression": linear_pipeline,
        "Random Forest": forest_pipeline
    }
    results = {}
    best_name, best_model, best_rmse = None, None, float("inf")

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        results[name] = {
            "cv_rmse_mean_100k": float(-cv_scores.mean()),
            "test_mae_100k": float(mae),
            "test_rmse_100k": float(rmse),
            "test_r2": float(r2)
        }
        print(f"CV RMSE: ${-cv_scores.mean() * 100000:,.0f}")
        print(f"Test MAE: ${mae * 100000:,.0f}")
        print(f"Test RMSE: ${rmse * 100000:,.0f}")
        print(f"Test R²: {r2:.4f}")

        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, rmse
            best_predictions = predictions

    joblib.dump({
        "model": best_model,
        "features": list(X.columns),
        "target_units": "USD x 100,000",
        "best_model_name": best_name
    }, MODEL_DIR / "house_price_model.joblib")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "models": results}, f, indent=2)

    comparison = pd.DataFrame([
        {"Model": name, **metrics} for name, metrics in results.items()
    ])
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Prediction vs actual plot for the selected model
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=best_predictions, alpha=0.35)
    low = min(y_test.min(), best_predictions.min())
    high = max(y_test.max(), best_predictions.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual value ($100,000 units)")
    plt.ylabel("Predicted value ($100,000 units)")
    plt.title(f"Actual vs Predicted - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name}")
    print("Saved model: models/house_price_model.joblib")
    print("Saved metrics and chart in outputs/")
    print("\nReminder: educational model; not a current property valuation tool.")

if __name__ == "__main__":
    main()
