import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Read dataset
df = pd.read_csv("data/demo.csv")

# Features and target
X = df.drop("pass_fail", axis=1)
y = df["pass_fail"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow experiment
mlflow.set_experiment("student_pass_fail_experiment")

with mlflow.start_run(run_name="Random Forest - Dataset V3"):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    mlflow.log_param("model_name", "Random Forest")
    mlflow.log_param("dataset_version", "V3")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")