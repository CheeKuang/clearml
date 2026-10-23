import joblib
from clearml import Task, OutputModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

S3_BUCKET = "s3://localhost:9000/clearml-bucket"

task = Task.init(
    project_name="s3-setup",
    task_name="iris-train-register",
    output_uri=S3_BUCKET,
    auto_connect_frameworks={"scikit": False, "joblib": False},
)

params = {"n_estimators": 50, "max_depth": 4, "random_state": 42}
task.connect(params)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(**params).fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print("accuracy:", acc)
task.get_logger().report_scalar("metrics", "accuracy", value=acc, iteration=0)

joblib.dump(model, "model.joblib")

# Register the weights as a ClearML model and upload them to S3
output_model = OutputModel(
    task=task,
    name="iris-rf",
    framework="ScikitLearn",
    tags=["iris", "demo"],
)
output_model.update_design(config_dict=params)
output_model.update_weights(
    weights_filename="model.joblib",
    auto_delete_file=False,   # keep the local copy
)
output_model.publish()

print("Model ID:", output_model.id)
print("Model URL:", output_model.url)
print("Task ID:", task.id)
task.close()
