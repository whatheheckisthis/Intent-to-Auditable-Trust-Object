import subprocess
import os
import csv
from datetime import datetime

def run_isabelle():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(project_dir, "isabelle_proof_log.txt")
    csv_file = os.path.join(project_dir, "isabelle_proof_log.csv")
    
    cmd = [
        "isabelle", "build",
        "-D", project_dir,
        "-v"
    ]
    
    # Run Isabelle build and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save to text log
    with open(log_file, "w") as f:
        f.write(result.stdout)
        f.write("\n--- ERRORS / STDERR ---\n")
        f.write(result.stderr)
    
    # Save to CSV with timestamp and status
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "status", "stdout", "stderr"])
        writer.writerow([datetime.now().isoformat(), result.returncode, result.stdout, result.stderr])
    
    print(f"Proof log saved to: {log_file}")
    print(f"CSV log saved to: {csv_file}")

if __name__ == "__main__":
    run_isabelle()
