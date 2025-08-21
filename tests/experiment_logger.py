experiment_log = []

def log_experiment_step(step_data):
    experiment_log.append(step_data)

def get_experiment_log():
    return experiment_log
