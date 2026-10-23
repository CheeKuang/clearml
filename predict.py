import sys
import joblib
from clearml import Task, Model
from sklearn.datasets import load_iris

# boto3 is only used implicitly (S3 download), so ClearML can't detect it
Task.add_requirements("boto3")

task = Task.init(project_name="s3-setup", task_name="iris-predict-docker")

params = {"model_id": "2eee7337c1ae43309e38b1da2c3d2cf8"}
task.connect(params)

# Everything above runs locally; the task is then queued and this process exits.
task.execute_remotely(queue_name="production", exit_process=True)

# Everything below runs on the agent.
print("running with:", sys.executable)

model_meta = Model(model_id=params["model_id"])
print("model:", model_meta.name, "| published:", model_meta.published)
print("url:", model_meta.url)

local_path = model_meta.get_local_copy()
print("downloaded to:", local_path)

clf = joblib.load(local_path)
X, y = load_iris(return_X_y=True)
print("predictions:", clf.predict(X[:5]).tolist())
print("expected:   ", y[:5].tolist())
task.close()
