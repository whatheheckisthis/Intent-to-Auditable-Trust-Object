import os
import subprocess
from datetime import datetime

def run(cmd):
    return subprocess.getoutput(cmd).strip()

def header(title):
    print(f"\n{'='*40}\n{title}\n{'='*40}")

def main():
    header("PROJECT REPORT")
    print(f"Generated: {datetime.utcnow().isoformat()} UTC")

    # Git summary
    header("Git Commits (Last 5)")
    print(run("git log --oneline -n 5"))

    header("Git Tags")
    print(run("git tag --sort=-creatordate | head -n 5"))

    # Log file summary
    header("Log Files Summary")
    log_dir = "logs"
    if os.path.isdir(log_dir):
        logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        for log in logs:
            path = os.path.join(log_dir, log)
            size_kb = os.path.getsize(path) // 1024
            print(f"{log}: {size_kb} KB")
    else:
        print("No logs directory found.")

    # File size summary (optional)
    header("Top-Level Files")
    for f in os.listdir("."):
        if os.path.isfile(f):
            size_kb = os.path.getsize(f) // 1024
            print(f"{f}: {size_kb} KB")

    print("\nReport generation complete.")

if __name__ == "__main__":
    main()

