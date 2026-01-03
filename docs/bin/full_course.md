
# Beginner Foundations – 6 Week Kernel Environment

This repo teaches you how to set up a **Python-driven invisible frontend environment** (mimicking a JS/React setup) with a structured 6-week program.
All tasks are designed as `bash` scripts, and you can run them directly without worrying about JavaScript.

---

## Pre-Setup Checklist


# Environment Setup for 18x Kernel Frontend

This checklist will help you set up the environment step-by-step.  
Each box represents a task. Run the command in your terminal, then tick it off once completed .



* [ ] **Install Python 3 and pip**

  ```bash
  sudo apt install -y python3 python3-pip
  ```

* [ ] **Install Node.js and npm**

  ```bash
  sudo apt install -y nodejs npm
  ```

* [ ] **Check versions**

  ```bash
  python3 --version
  pip3 --version
  node -v
  npm -v
  ```

* [ ] **Install virtualenv**

  ```bash
  pip3 install virtualenv
  ```

* [ ] **Create a project directory**

  ```bash
  mkdir 18x_project && cd 18x_project
  ```

* [ ] **Setup virtual environment**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* [ ] **Install Jupyter**

  ```bash
  pip install jupyterlab
  ```

* [ ] **Install project dependencies**

  ```bash
  pip install numpy pandas matplotlib
  ```

* [ ] **Initialize Node.js frontend**

  ```bash
  npm init -y
  ```

* [ ] **Install React (optional, for UI later)**

  ```bash
  npx create-react-app frontend
  ```

* [ ] **Run first hello world test**

  ```bash
  echo 'print("Hello World from 18x Kernel!")' > run_18x_kernel.py
  python run_18x_kernel.py
  ```

---

✅ Once all boxes are checked, your environment is ready!



---

## Week 0: Environment Setup

```bash
#!/bin/bash
# setup_env.sh

echo "Setting up Python venv..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install flask jinja2 rich

echo "Environment ready!"
```

Checklist:

```bash
# [ ] Run `bash setup_env.sh`
# [ ] Verify venv is active
# [ ] Confirm Flask runs: python -m flask --version
```

---

## Week 1: Hello World (Full Doc)

```bash
#!/bin/bash
# week1_hello.sh

echo "Running Week 1: Hello World (Python-as-Frontend)"

cat <<'PYCODE' > hello.py
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello World </h1><p>This is your invisible frontend powered by Python!</p>"

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python hello.py
```

Checklist:

```bash
# [ ] Run `bash week1_hello.sh`
# [ ] Open browser: http://127.0.0.1:5000
# [ ] See Hello World page
# [ ] Modify message & re-run
```

---

## Week 2: Templates & Static Files

```bash
#!/bin/bash
# week2_templates.sh

mkdir -p templates static/css

cat <<'HTML' > templates/index.html
<!DOCTYPE html>
<html>
<head>
  <title>Week 2 - Templates</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <h1>Welcome to Week 2</h1>
  <p>This uses HTML templates, but powered by Python!</p>
</body>
</html>
HTML

cat <<'CSS' > static/css/style.css
body {
  font-family: Arial;
  background: #fafafa;
  text-align: center;
  margin: 50px;
}
h1 { color: #007BFF; }
CSS

cat <<'PYCODE' > app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python app.py
```

Checklist:

```bash
# [ ] Run `bash week2_templates.sh`
# [ ] Page now loads with styles
# [ ] Edit `style.css` to change colors
```

---

## Week 3: Inputs & Forms

```bash
#!/bin/bash
# week3_forms.sh

cat <<'HTML' > templates/form.html
<!DOCTYPE html>
<html>
<head><title>Week 3 Forms</title></head>
<body>
  <h1>Enter Your Name</h1>
  <form method="POST">
    <input type="text" name="username" placeholder="Your Name">
    <button type="submit">Submit</button>
  </form>
  {% if name %}
  <p>Hello, {{ name }}!</p>
  {% endif %}
</body>
</html>
HTML

cat <<'PYCODE' > forms_app.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def form():
    name = None
    if request.method == "POST":
        name = request.form.get("username")
    return render_template("form.html", name=name)

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python forms_app.py
```

Checklist:

```bash
# [ ] Run `bash week3_forms.sh`
# [ ] Submit your name in the form
# [ ] See greeting appear
```

---

## Week 4: Persistent Data (Notes)

```bash
#!/bin/bash
# week4_notes.sh

cat <<'HTML' > templates/notes.html
<!DOCTYPE html>
<html>
<head><title>Notes</title></head>
<body>
  <h1>Week 4 - Notes</h1>
  <form method="POST">
    <input type="text" name="note" placeholder="Write a note">
    <button type="submit">Save</button>
  </form>
  <ul>
  {% for n in notes %}
    <li>{{n}}</li>
  {% endfor %}
  </ul>
</body>
</html>
HTML

cat <<'PYCODE' > notes_app.py
from flask import Flask, render_template, request

app = Flask(__name__)
notes = []

@app.route("/", methods=["GET","POST"])
def notes_view():
    if request.method == "POST":
        note = request.form.get("note")
        if note:
            notes.append(note)
    return render_template("notes.html", notes=notes)

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python notes_app.py
```

Checklist:

```bash
# [ ] Run `bash week4_notes.sh`
# [ ] Add a note
# [ ] Refresh to see list of notes
```

---

## Week 5: JSON API

```bash
#!/bin/bash
# week5_api.sh

cat <<'PYCODE' > api_app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/hello")
def hello_api():
    return jsonify({"message": "Hello API World", "status": "ok"})

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python api_app.py
```

Checklist:

```bash
# [ ] Run `bash week5_api.sh`
# [ ] Visit http://127.0.0.1:5000/api/hello
# [ ] See JSON response
```

---

## Week 6: Combined Mini-App

```bash
#!/bin/bash
# week6_full.sh

cat <<'PYCODE' > mini_app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
notes = []

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        note = request.form.get("note")
        if note:
            notes.append(note)
    return render_template("notes.html", notes=notes)

@app.route("/api/notes")
def notes_api():
    return jsonify(notes)

if __name__ == "__main__":
    app.run(port=5000)
PYCODE

python mini_app.py
```

Checklist:

```bash
# [ ] Run `bash week6_full.sh`
# [ ] Add notes via UI
# [ ] Fetch notes via API
```

---

By the end of 6 weeks, students will have built a **Python-only invisible frontend** that mimics a JS/React workflow but stays entirely inside `bash` + `Flask`.





