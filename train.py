from clearml import Task

task = Task.init(project_name="demo", task_name="remote run")
task.execute_remotely(queue_name="production")

print("This part runs on the agent")
