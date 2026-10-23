import joblib
from clearml import Task
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # Task.init must come first. Credentials come from ~/clearml.conf (Mac / agent)
    # or from Task.set_credentials(...) called earlier in the same process (Kaggle).
    task = Task.init(
        project_name="ClearML-Test",
        task_name="iris-rf",
        output_uri=True,  # upload to the ClearML files server configured for your account
    )

    # Hyperparameters: editable in the UI when the task is cloned and enqueued
    params = {"n_estimators": 100, "max_depth": 5}
    params = task.connect(params)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    task.get_logger().report_scalar(
        title="accuracy", series="test", value=acc, iteration=0
    )

    # ClearML auto-captures joblib.dump and registers it as an output model
    joblib.dump(model, "model.pkl")

    task.close()


if __name__ == "__main__":
    main()
