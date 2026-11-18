import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def apply():
    """Keep every library cache and config directory inside the project."""
    for name, path in {
        "JUPYTER_CONFIG_PATH": ROOT,
        "JUPYTER_CONFIG_DIR": ROOT / ".jupyter" / "config",
        "JUPYTER_DATA_DIR": ROOT / ".jupyter" / "data",
        "JUPYTERLAB_SETTINGS_DIR": ROOT / ".jupyter" / "lab" / "user-settings",
        "JUPYTER_RUNTIME_DIR": ROOT / ".jupyter" / "runtime",
        "IPYTHONDIR": ROOT / ".jupyter" / "ipython",
        "NLTK_DATA": ROOT / ".data" / "nltk_data",
        "SCIKIT_LEARN_DATA": ROOT / ".data" / "scikit_learn_data",
        "KERAS_HOME": ROOT / ".keras",
        "MPLCONFIGDIR": ROOT / ".cache" / "matplotlib",
        "HF_HOME": ROOT / ".cache" / "huggingface",
    }.items():
        os.environ.setdefault(name, str(path))
