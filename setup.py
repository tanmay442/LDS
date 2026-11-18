#!/usr/bin/env python3
"""One-command setup and launcher for the Data Science environment.

Works on Windows, macOS and Linux. Requires only Python on PATH.

Usage:
    python setup.py
    PORT=8890 python setup.py          # use a different port
"""

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BOOTSTRAP = ROOT / ".bootstrap"
MARKER = VENV / ".project-path"
PORT = int(os.environ.get("PORT", "8888"))
NOTEBOOK_URL = f"http://127.0.0.1:{PORT}/lab/tree/notebooks/00_START_HERE.ipynb"

THEME_SETTINGS = """{
  "theme": "JupyterLab Dark"
}
"""

WARNING_FILTERS = '''import warnings

warnings.filterwarnings("ignore", message="distutils Version classes are deprecated")
warnings.filterwarnings("ignore", message="Passing 'N' to ListedColormap is deprecated")
'''


def tool(directory, name):
    """Return the path of an executable inside a virtual environment."""
    suffix = ".exe" if os.name == "nt" else ""
    return directory / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def project_environment():
    """Map every library cache and config directory into the project folder."""
    return {
        "JUPYTER_CONFIG_PATH": str(ROOT),
        "JUPYTER_CONFIG_DIR": str(ROOT / ".jupyter" / "config"),
        "JUPYTER_DATA_DIR": str(ROOT / ".jupyter" / "data"),
        "JUPYTERLAB_SETTINGS_DIR": str(ROOT / ".jupyter" / "lab" / "user-settings"),
        "JUPYTER_RUNTIME_DIR": str(ROOT / ".jupyter" / "runtime"),
        "IPYTHONDIR": str(ROOT / ".jupyter" / "ipython"),
        "NLTK_DATA": str(ROOT / ".data" / "nltk_data"),
        "SCIKIT_LEARN_DATA": str(ROOT / ".data" / "scikit_learn_data"),
        "KERAS_HOME": str(ROOT / ".keras"),
        "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib"),
        "HF_HOME": str(ROOT / ".cache" / "huggingface"),
    }


def write_file(path, content):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_project_settings():
    write_file(
        ROOT / ".jupyter" / "lab" / "user-settings"
        / "@jupyterlab" / "apputils-extension" / "themes.jupyterlab-settings",
        THEME_SETTINGS,
    )
    write_file(
        ROOT / ".jupyter" / "ipython" / "profile_default" / "startup"
        / "00_warning_filters.py",
        WARNING_FILTERS,
    )
    for directory in (
        ".jupyter/runtime", ".jupyter/data", ".jupyter/config",
        ".data/nltk_data", ".data/scikit_learn_data",
        ".keras", ".cache/matplotlib",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)


def run(command, env=None):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True, env=env)


def venv_is_current():
    if not tool(VENV, "python").exists() or not MARKER.exists():
        return False
    return MARKER.read_text(encoding="utf-8").strip() == str(ROOT)


def install_uv():
    """Install uv into a temporary venv so it can manage the real environment."""
    if tool(BOOTSTRAP, "uv").exists():
        return tool(BOOTSTRAP, "uv")
    print("Installing uv...")
    run([sys.executable, "-m", "venv", str(BOOTSTRAP)])
    run([
        tool(BOOTSTRAP, "python"), "-m", "pip", "install",
        "--disable-pip-version-check", "--quiet", "uv",
    ])
    return tool(BOOTSTRAP, "uv")


def build_environment():
    uv = install_uv()
    print("Installing Python 3.12 if needed...")
    run([uv, "python", "install", "3.12"])
    print("Creating the virtual environment...")
    run([uv, "venv", "--clear", "--python", "3.12", str(VENV)])
    print("Installing the libraries (this takes a few minutes on first run)...")
    run([uv, "sync"])
    MARKER.write_text(str(ROOT), encoding="utf-8")
    shutil.rmtree(BOOTSTRAP, ignore_errors=True)


def server_is_running():
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=2) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def start_server(env):
    print("Starting JupyterLab...")
    log_path = ROOT / ".jupyter" / "jupyter.log"
    arguments = [
        str(tool(VENV, "python")), "-m", "jupyterlab",
        "--no-browser",
        "--ip=127.0.0.1",
        f"--port={PORT}",
        "--port-retries=0",
        f"--notebook-dir={ROOT}",
    ]
    options = {
        "cwd": ROOT,
        "env": env,
        "stdout": log_path.open("w"),
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        options["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        options["start_new_session"] = True
    subprocess.Popen(arguments, **options)

    for _ in range(180):
        if server_is_running():
            return
        time.sleep(1)
    raise RuntimeError(f"JupyterLab did not start. See {log_path}")


def main():
    environment = os.environ.copy()
    environment.update(project_environment())
    create_project_settings()

    if not venv_is_current():
        print("First run, or the folder moved. Setting up the environment.")
        build_environment()

    if not (ROOT / "data" / "iris.csv").exists():
        print("Downloading the datasets...")
        run([tool(VENV, "python"), ROOT / "scripts" / "download_data.py", "--nltk"], environment)

    if not server_is_running():
        start_server(environment)

    print(f"Opening {NOTEBOOK_URL}")
    if not webbrowser.open(NOTEBOOK_URL):
        print("No browser was found. Paste the URL above into your browser.")
    print("Dark theme is on. No token or password is required.")


if __name__ == "__main__":
    main()
