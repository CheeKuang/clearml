from clearml import Task

S3_BUCKET = "s3://localhost:9000/clearml-bucket"

task = Task.init(
    project_name="s3-setup",
    task_name="s3-artifact-test",
    output_uri=S3_BUCKET,
)

task.upload_artifact("hello_artifact", artifact_object={"msg": "hello from clearml", "n": 42})
task.upload_artifact("notes_file", artifact_object="s3_test.py")

print("Task ID:", task.id)
task.close()
