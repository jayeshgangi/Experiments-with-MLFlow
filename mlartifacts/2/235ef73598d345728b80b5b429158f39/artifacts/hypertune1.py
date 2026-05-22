from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import pandas as pd

# Load dataset
data = load_breast_cancer()

x = data.data
y = data.target

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=35,
    stratify=y
)

# Model
rf = RandomForestClassifier(random_state=35)

# Hyperparameters
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 2, 4, 6, 10, 20, 30]
}

# Grid Search
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Hyperparameter Tuning of Breast Cancer Dataset")

with mlflow.start_run():

    # Train
    grid.fit(x_train, y_train)

    best_params = grid.best_params_
    best_score = grid.best_score_

    # Log params & metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("Accuracy", best_score)

    # Convert numpy arrays to DataFrames
    train_df = pd.DataFrame(x_train, columns=data.feature_names)
    train_df['target'] = y_train

    test_df = pd.DataFrame(x_test, columns=data.feature_names)
    test_df['target'] = y_test

    # Save CSVs temporarily
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # Log datasets as artifacts
    mlflow.log_artifact("train.csv")
    mlflow.log_artifact("test.csv")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=grid.best_estimator_,
        artifact_path="rf_model"
    )

    # Tags
    mlflow.set_tags({
        'Author': 'Jayesh',
        'model': 'RandomForest',
        'dataset': 'Breast Cancer'
    })

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")