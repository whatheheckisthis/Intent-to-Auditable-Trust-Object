#!/usr/bin/env python3
"""
run_18x_kernel.py

This script orchestrates the setup of a 6-week learning environment for students.
It auto-generates six weekly .sh scripts (week1.sh ... week6.sh) containing
progressive foundational tasks. Students only need to run this Python script
once, and then follow the generated weekly scripts.

Usage:
    python3 run_18x_kernel.py
"""

import os
import stat

# Directory to store weekly scripts
WEEK_DIR = "weekly_scripts"

# Weekly notes & commands
WEEKS = {
    1: [
        "# Week 1: Environment Basics",
        "echo 'Setting up Week 1: Linux shell, Python basics, and Git.'",
        "sudo apt update -y",
        "sudo apt install -y python3 python3-pip git",
        "python3 --version",
        "git --version",
    ],
    2: [
        "# Week 2: Virtual Environments & Editors",
        "echo 'Setting up Week 2: Virtual environments and editors.'",
        "pip3 install virtualenv",
        "virtualenv venv_week2",
        "source venv_week2/bin/activate",
        "pip install jupyter",
        "echo 'alias week2_notebook=\"jupyter notebook\"' >> ~/.bashrc",
    ],
    3: [
        "# Week 3: Git & Version Control",
        "echo 'Setting up Week 3: Git usage and GitHub connection.'",
        "git config --global user.name 'student'",
        "git config --global user.email 'student@example.com'",
        "mkdir week3_repo && cd week3_repo && git init",
        "echo '# Week 3 repo' > README.md",
        "git add README.md && git commit -m 'Initial commit for Week 3'",
    ],
    4: [
        "# Week 4: Frontend Environment",
        "echo 'Setting up Week 4: Node.js and frontend basics.'",
        "sudo apt install -y nodejs npm",
        "mkdir week4_frontend && cd week4_frontend",
        "npm init -y",
        "npm install react react-dom",
        "echo 'Frontend starter created in week4_frontend'",
    ],
    5: [
        "# Week 5: Python + Frontend Bridge",
        "echo 'Setting up Week 5: Flask backend + React bridge.'",
        "pip install flask",
        "mkdir week5_backend && cd week5_backend",
        "echo 'from flask import Flask\napp = Flask(__name__)\n@app.route(\"/\")\ndef home():\n    return \"Hello from Flask!\"\n\nif __name__==\"__main__\":\n    app.run(debug=True)' > app.py",
    ],
    6: [
        "# Week 6: Full Stack Integration",
        "echo 'Setting up Week 6: Connecting backend + frontend.'",
        "mkdir week6_fullstack && cd week6_fullstack",
        "echo 'Placeholder for full stack project setup'",
        "echo 'Students combine Flask + React here with API calls'",
    ],
}

def make_executable(filepath):
    """Make a file executable"""
    st = os.stat(filepath)
    os.chmod(filepath, st.st_mode | stat.S_IEXEC)

def main():
    os.makedirs(WEEK_DIR, exist_ok=True)
    print("Generating weekly scripts in:", WEEK_DIR)

    for week, lines in WEEKS.items():
        filename = os.path.join(WEEK_DIR, f"week{week}.sh")
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("\n".join(lines))
            f.write("\n")
        make_executable(filename)
        print(f"✅ Created {filename}")

    print("\n Setup complete!")
    print("To start Week 1, run:")
    print("   bash weekly_scripts/week1.sh")

if __name__ == "__main__":
    main()

