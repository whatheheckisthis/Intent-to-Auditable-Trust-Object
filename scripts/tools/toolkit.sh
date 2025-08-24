<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>UtilitiesGrid</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --primary: #2563eb;
      --primary-dark: #1e40af;
      --accent: #22d3ee;
      --bg: #111827;
      --bg-modal: #1e293b;
      --border: #334155;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #22c55e;
      --gray: #64748b;
      --yellow: #fde047;
      --purple: #a78bfa;
      --shadow: 0 8px 32px rgba(30,41,59,0.14);
      --copy-bg: #181b24;
      --copy-fg: #dbeafe;
      --card-hover: #172136;
    }
    html, body {
      height: 100%;
    }
    body {
      background: var(--bg);
      font-family: 'JetBrains Mono', 'Fira Mono', 'Menlo', 'Consolas', 'Segoe UI', monospace, sans-serif;
      color: var(--text-main);
      margin: 0;
      padding: 0;
      min-height: 100vh;
    }
    header {
      background: linear-gradient(90deg, var(--primary), var(--accent) 80%);
      color: #fff;
      padding: 2.2rem 0 1.2rem 0;
      text-align: center;
      border-bottom-left-radius: 32px;
      border-bottom-right-radius: 32px;
      box-shadow: var(--shadow);
      margin-bottom: 36px;
    }
    header h1 {
      font-size: 2.2em;
      font-weight: 900;
      margin: 0 0 0.45em 0;
      letter-spacing: -2px;
      text-shadow: 0 2px 8px #2563eb55;
    }
    header p {
      font-size: 1.13em;
      color: #e0e7ef;
      margin: 0 auto 1.3em;
      max-width: 700px;
      font-weight: 500;
      text-shadow: 0 1px 8px #2563eb44;
    }
    .utility-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      grid-auto-rows: 1fr;
      gap: 36px;
      margin: 3em auto 0;
      max-width: 1600px;
      min-height: 80vh;
      padding-bottom: 48px;
      box-sizing: border-box;
    }
    .utility-card {
      background: var(--bg-modal);
      border: 2px solid var(--primary);
      border-radius: 20px;
      box-shadow: 0 6px 30px 0 #2563eb28;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      min-width: 0;
      min-height: 340px;
      height: 100%;
      position: relative;
      transition: background 0.22s, transform 0.17s, box-shadow 0.19s;
      overflow: hidden;
      padding: 24px 18px 18px 18px;
      box-sizing: border-box;
    }
    .utility-card:hover, .utility-card:focus-within {
      background: var(--card-hover);
      transform: scale(1.01);
      box-shadow: 0 10px 36px #22d3ee33, 0 0px 0px #0000;
      border-color: var(--accent);
      z-index: 1;
    }
    .utility-card h3 {
      margin-top: 0;
      margin-bottom: 0.35em;
      color: var(--primary-dark);
      font-size: 1.08em;
      font-weight: 800;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .utility-card .badge {
      font-size: 0.74em;
      font-weight: 700;
      background: var(--primary);
      color: #fff;
      border-radius: 7px;
      padding: 2.5px 10px;
      margin-left: 6px;
      box-shadow: 0 1px 5px #2563eb33;
      text-shadow: 0 1px 1px #0008;
    }
    .utility-card .badge-bash { background: var(--gray); color: #fff; }
    .utility-card .badge-python { background: var(--yellow); color: #1e293b;}
    .utility-card .badge-fastapi { background: var(--accent); color: #1e293b; }
    .utility-card .badge-git { background: var(--purple); color: #1e293b; }
    .utility-card .badge-ops { background: var(--success); color: #fff; }
    .utility-card p {
      color: var(--text-muted);
      font-size: 0.98em;
      margin-bottom: 0.9em;
      margin-top: 0.5em;
      min-height: 36px;
      line-height: 1.6;
      letter-spacing: 0.01em;
    }
    .utility-card .download-btn {
      display: inline-block;
      background: var(--primary-dark);
      color: #fff;
      font-weight: bold;
      border: none;
      border-radius: 7px;
      padding: 7px 18px;
      font-size: 1.02em;
      margin-bottom: 12px;
      margin-left: 0;
      margin-top: 2px;
      margin-right: auto;
      box-shadow: 0 1px 5px #161b2220;
      cursor: pointer;
      transition: background 0.13s, color 0.11s;
    }
    .utility-card .download-btn:active, .utility-card .download-btn:focus {
      background: var(--success);
      color: #fff;
      outline: none;
    }
    .utility-card .expand-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: none;
      color: var(--primary-dark);
      border: none;
      font-size: 1.3em;
      font-weight: bold;
      cursor: pointer;
      margin: 0 0 7px 0;
      align-self: flex-end;
      transition: color 0.13s;
      padding: 1px 2px 1px 2px;
      outline: none;
      height: 32px;
      width: 32px;
      border-radius: 4px;
      line-height: 1;
    }
    .utility-card .expand-btn:hover,
    .utility-card .expand-btn:focus {
      background: var(--primary);
      color: #fff;
    }
    .utility-card pre {
      background: var(--copy-bg);
      color: var(--copy-fg);
      border-radius: 10px;
      padding: 13px 11px 13px 17px;
      margin: 0 0 0.85em 0;
      font-size: 0.97em;
      overflow-x: auto;
      overflow-y: auto;
      position: relative;
      font-family: 'JetBrains Mono', 'Fira Mono', 'Menlo', 'Consolas', monospace;
      box-shadow: 0 2px 14px #22d3ee11;
      line-height: 1.6;
      letter-spacing: 0.01em;
      border-left: 4px solid var(--accent);
      word-break: break-word;
      white-space: pre-wrap;
      max-height: 180px;
      min-height: 0;
      transition: max-height 0.25s;
      flex: 1 1 auto;
    }
    .utility-card.expanded pre {
      max-height: 600px;
      min-height: 180px;
      transition: max-height 0.25s;
    }
    .utility-card .fadeout {
      content: "";
      display: block;
      position: absolute;
      left: 0; right: 0; bottom: 0;
      height: 30px;
      background: linear-gradient(180deg, rgba(24,27,36,0.02) 0%, var(--bg-modal) 96%);
      pointer-events: none;
    }
    .utility-card {
      height: 100%;
      min-height: 340px;
      box-sizing: border-box;
    }
    @media (max-width: 1400px) {
      .utility-grid {
        max-width: 98vw;
        gap: 18px 18px;
      }
    }
    @media (max-width: 1150px) {
      .utility-grid {
        grid-template-columns: repeat(3, 1fr);
        grid-auto-rows: 1fr;
        gap: 22px 22px;
      }
    }
    @media (max-width: 900px) {
      .utility-grid {
        grid-template-columns: 1fr 1fr;
        grid-auto-rows: 1fr;
        gap: 18px 18px;
      }
      .utility-card { padding: 18px 8px 14px 8px }
    }
    @media (max-width: 650px) {
      .utility-grid {
        grid-template-columns: 1fr;
        gap: 14px 0;
      }
      .utility-card { padding: 10px 0.5em 10px 0.5em }
    }
    .external-links {
      margin: 42px auto 0 auto;
      display: flex;
      gap: 22px;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
      text-align: center;
      font-size: 1.04em;
    }
    .external-links a {
      color: var(--primary);
      text-decoration: underline;
      font-weight: 500;
      border-radius: 4px;
      transition: background 0.13s;
      padding: 2px 8px;
      display: inline-block;
    }
    .external-links a:hover, .external-links a:focus {
      background: var(--accent);
      color: #155e75;
      outline: 1px solid var(--primary-dark);
    }
    .expand-btn svg {
      display: block;
      width: 22px;
      height: 22px;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <header>
    <h1>Secure Bash Utilities Grid</h1>
    <p>
      256-color inspired utility suite for file encryption, decryption, secure file handling, and daily DevOps/support operations.<br>
      <span style="color:var(--accent);font-weight:bold;">OpenSSL</span>, <span style="color:var(--primary-dark);font-weight:bold;">Bash</span>, <span style="color:var(--yellow);font-weight:bold;">Python</span>, <span style="color:var(--success);font-weight:bold;">FastAPI</span>, <span style="color:var(--primary);font-weight:bold;">Docker</span>
    </p>
  </header>
  <div class="utility-grid">
    <!-- Changelog Ops + Bash Tools Installer -->
    <div class="utility-card" data-script="changelog-deps-script">
      <h3>Install Changelog & Bash Tools <span class="badge badge-bash">Bash</span> <span class="badge badge-python">Python</span> <span class="badge badge-ops">Ops</span></h3>
      <p>Install <b>all required dependencies</b> for changelog operations and daily Bash tools.</p>
      <button class="download-btn" onclick="downloadScript('changelog-deps-script', 'changelog-deps.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="changelog-deps-script">#!/bin/bash
set -e
echo ""
echo "[Changelog & Bash Tools Installer]"

echo -e "\n[1/9] Checking for Git..."
command -v git >/dev/null || { echo "  Installing git"; sudo apt-get install -y git; }
echo "  $(git --version | head -n1)"

echo -e "\n[2/9] Checking for Python3..."
command -v python3 >/dev/null || { echo "  Installing python3"; sudo apt-get install -y python3; }
echo "  Python: $(python3 --version)"

echo -e "\n[3/9] Checking for pip3..."
command -v pip3 >/dev/null || { echo "  Installing python3-pip"; sudo apt-get install -y python3-pip; }
pip3 --version

echo -e "\n[4/9] Upgrading pip..."
pip3 install --upgrade pip

echo -e "\n[5/9] Installing required Python packages:"
for pkg in gitpython click rich auto-changelog markdown; do
  echo "    $pkg"
  pip3 install --upgrade $pkg
done

echo -e "\n[6/9] Checking for Bash tools (awk, sed, grep)..."
for btool in awk sed grep; do
  command -v $btool >/dev/null || { echo "  Installing $btool"; sudo apt-get install -y $btool; }
  echo "  Found $btool: $($btool --version 2>&1 | head -n1)"
done

echo -e "\n[7/9] Checking for OpenSSL (for encryption)..."
command -v openssl >/dev/null || { echo "  Installing openssl"; sudo apt-get install -y openssl; }
echo "  $(openssl version)"

echo -e "\n[8/9] Checking for Docker (optional for containerized ops)..."
if ! command -v docker >/dev/null; then
  echo "  Docker not found. Install with: sudo apt-get install -y docker.io"
else
  echo "  $(docker --version)"
fi

echo -e "\n[9/9] Checking for Pandoc (optional for MD→HTML)..."
if ! command -v pandoc >/dev/null; then
  echo "  Pandoc not found. Install with: sudo apt-get install pandoc -y"
else
  echo "  $(pandoc --version | head -n1)"
fi

echo -e "\nAll dependencies for changelog ops and Bash tools (git, python3, pip3, gitpython, click, rich, auto-changelog, markdown, awk, sed, grep, openssl, [docker], [pandoc]) are installed or checked."
</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Symmetric File Encryption Using OpenSSL -->
    <div class="utility-card" data-script="openssl-enc-script">
      <h3>Symmetric File Encryption <span class="badge badge-bash">Bash</span> <span class="badge">OpenSSL</span></h3>
      <p>Encrypt or decrypt files using strong AES-256 with OpenSSL for secure operations.</p>
      <button class="download-btn" onclick="downloadScript('openssl-enc-script', 'openssl-enc.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="openssl-enc-script"># Encrypt
openssl enc -aes-256-cbc -salt -in input.txt -out output.enc
# Decrypt (interactive)
openssl enc -d -aes-256-cbc -in output.enc -out decrypted.txt
</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Decrypt (with inline passphrase) -->
    <div class="utility-card" data-script="openssl-decrypt-inline-script">
      <h3>Decrypt File (with Inline Passphrase) <span class="badge badge-bash">Bash</span></h3>
      <p>Decrypt a file using a passphrase inline (use with caution for scripts).</p>
      <button class="download-btn" onclick="downloadScript('openssl-decrypt-inline-script', 'decrypt-inline.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="openssl-decrypt-inline-script">openssl enc -d -aes-256-cbc -in output.enc -out decrypted.txt -pass pass:'YOUR_PASSPHRASE'</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Decrypt and Save Output to File -->
    <div class="utility-card" data-script="openssl-decrypt-save-script">
      <h3>Decrypt & Save Output <span class="badge badge-bash">Bash</span></h3>
      <p>Decrypt and pipe output (useful for chaining or logging).</p>
      <button class="download-btn" onclick="downloadScript('openssl-decrypt-save-script', 'decrypt-save.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="openssl-decrypt-save-script">openssl enc -d -aes-256-cbc -in output.enc | tee decrypted.txt</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Environment Bootstrapping Script -->
    <div class="utility-card" data-script="env-bootstrap-script">
      <h3>Environment Bootstrap <span class="badge badge-bash">Bash</span></h3>
      <p>Provision tools, directories, and permissions for a secure automation environment.</p>
      <button class="download-btn" onclick="downloadScript('env-bootstrap-script', 'bootstrap.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="env-bootstrap-script">#!/bin/bash
set -e
command -v openssl || sudo apt-get install openssl -y
command -v docker || sudo apt-get install docker.io -y
command -v git || sudo apt-get install git -y
mkdir -p files
chmod 700 files
echo "Environment ready!"</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Clean Logs Older Than 7 Days -->
    <div class="utility-card" data-script="logclean-script">
      <h3>Clean Logs Older Than 7 Days <span class="badge badge-bash">Bash</span></h3>
      <p>Remove .log files older than 7 days for secure housekeeping.</p>
      <button class="download-btn" onclick="downloadScript('logclean-script', 'clean-logs.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="logclean-script">find /var/log -type f -name "*.log" -mtime +7 -delete</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Check for Active Python Virtual Environment -->
    <div class="utility-card" data-script="venv-check-script">
      <h3>Check Python Virtual Env <span class="badge badge-bash">Bash</span></h3>
      <p>Detect if a Python virtual environment is active.</p>
      <button class="download-btn" onclick="downloadScript('venv-check-script', 'check-venv.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="venv-check-script">if [[ "$VIRTUAL_ENV" != "" ]]; then echo "venv active"; else echo "no venv"; fi</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Local GitHub Repo Backup (No API) -->
    <div class="utility-card" data-script="repo-backup-script">
      <h3>Local GitHub Repo Backup <span class="badge badge-bash">Bash</span> <span class="badge badge-git">Git</span></h3>
      <p>Make a local mirror of a GitHub repository without using the API.</p>
      <button class="download-btn" onclick="downloadScript('repo-backup-script', 'repo-backup.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="repo-backup-script">git clone --mirror https://github.com/youruser/yourrepo</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Tail Log With Timestamps -->
    <div class="utility-card" data-script="tail-ts-script">
      <h3>Tail Log With Timestamps <span class="badge badge-bash">Bash</span></h3>
      <p>Stream logs in real time, each line prepended with a timestamp.</p>
      <button class="download-btn" onclick="downloadScript('tail-ts-script', 'tail-ts.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="tail-ts-script">tail -f /var/log/syslog | while read line; do echo "[$(date +%FT%T)] $line"; done</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Convert Markdown to HTML -->
    <div class="utility-card" data-script="md-to-html-script">
      <h3>Convert Markdown to HTML <span class="badge badge-bash">Bash</span> <span class="badge badge-python">Python</span></h3>
      <p>Convert .md files to HTML using Pandoc or Python markdown module.</p>
      <button class="download-btn" onclick="downloadScript('md-to-html-script', 'md-to-html.sh')">Download Script</button>
      <button class="expand-btn" title="Expand/Minimize" onclick="toggleExpand(this)">
        <svg class="expand-icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7 10l5 5 5-5z"/></svg>
      </button>
      <div style="position:relative;flex:1 1 auto;min-height:0;">
        <pre id="md-to-html-script"># With Pandoc:
pandoc file.md -o file.html
# Python:
python3 -m markdown file.md > file.html</pre>
        <div class="fadeout"></div>
      </div>
    </div>
    <!-- Filler cards for grid balance (invisible if grid has less than 25 scripts) -->
    <div style="display:none"></div><div style="display:none"></div><div style="display:none"></div><div style="display:none"></div>
    <div style="display:none"></div><div style="display:none"></div><div style="display:none"></div><div style="display:none"></div>
    <div style="display:none"></div><div style="display:none"></div><div style="display:none"></div><div style="display:none"></div>
    <div style="display:none"></div><div style="display:none"></div><div style="display:none"></div>
  </div>
  <div class="external-links">
    <a href="https://www.openssl.org/docs/" target="_blank" rel="noopener">OpenSSL Docs</a>
    <a href="https://docs.docker.com/" target="_blank" rel="noopener">Docker Docs</a>
    <a href="https://cryptography.io/" target="_blank" rel="noopener">Python Cryptography</a>
    <a href="https://pandoc.org/" target="_blank" rel="noopener">Pandoc Docs</a>
    <a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener">FastAPI Docs</a>
  </div>
  <script>
    function downloadScript(preId, filename) {
      var code = document.getElementById(preId).textContent.replace(/\r?\n$/, "");
      var blob = new Blob([code], {type: "text/x-sh"});
      var link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      setTimeout(function(){
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
      }, 100);
    }

    // Expand/minimize feature for long scripts
    function toggleExpand(btn) {
      var card = btn.closest('.utility-card');
      var pre = card.querySelector('pre');
      var fade = card.querySelector('.fadeout');
      var icon = btn.querySelector('svg');
      if (card.classList.contains('expanded')) {
        card.classList.remove('expanded');
        // Down chevron
        icon.innerHTML = '<path fill="currentColor" d="M7 10l5 5 5-5z"/>';
        pre.scrollTop = 0;
        if (fade) fade.style.display = '';
      } else {
        card.classList.add('expanded');
        // Up chevron
        icon.innerHTML = '<path fill="currentColor" d="M7 14l5-5 5 5z"/>';
        if (fade) fade.style.display = 'none';
      }
    }

    // Automatically minimize long scripts, expand for short ones
    window.addEventListener('DOMContentLoaded', function() {
      document.querySelectorAll('.utility-card').forEach(function(card) {
        var pre = card.querySelector('pre');
        var btn = card.querySelector('.expand-btn');
        var icon = btn ? btn.querySelector('svg') : null;
        if (pre && btn) {
          var isLong = pre.textContent.split('\n').length > 12 || pre.textContent.length > 700;
          if (!isLong) {
            btn.style.display = "none";
            card.classList.add('expanded');
            var fade = card.querySelector('.fadeout');
            if (fade) fade.style.display = 'none';
            if (icon) icon.innerHTML = '<path fill="currentColor" d="M7 14l5-5 5 5z"/>';
          } else {
            // Default to minimized with down chevron
            if (icon) icon.innerHTML = '<path fill="currentColor" d="M7 10l5 5 5-5z"/>';
          }
        }
      });
    });
  </script>
</body>
</html>
