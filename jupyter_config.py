import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for name, path in {
    "IPYTHONDIR": ROOT / ".jupyter" / "ipython",
    "JUPYTER_CONFIG_DIR": ROOT / ".jupyter" / "config",
    "JUPYTER_DATA_DIR": ROOT / ".jupyter" / "data",
    "NLTK_DATA": ROOT / ".data" / "nltk_data",
    "SCIKIT_LEARN_DATA": ROOT / ".data" / "scikit_learn_data",
    "KERAS_HOME": ROOT / ".keras",
    "MPLCONFIGDIR": ROOT / ".cache" / "matplotlib",
    "HF_HOME": ROOT / ".cache" / "huggingface",
}.items():
    os.environ.setdefault(name, str(path))

c.ServerApp.root_dir = str(ROOT)
c.ServerApp.ip = "127.0.0.1"
c.ServerApp.open_browser = True
c.ServerApp.token = ""
c.ServerApp.password = ""
c.ServerApp.allow_remote_access = False
c.IdentityProvider.token = ""
