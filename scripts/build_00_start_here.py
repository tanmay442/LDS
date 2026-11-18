"""Build the 00_START_HERE.ipynb index notebook."""

from nbhelper import code, md, write_notebook

cells = [
    md("""
# Welcome — Data Science learning environment

This notebook is the entry point for the whole course. It explains how the
environment is organised, maps every notebook to your syllabus, and gives you
short cheat sheets you can come back to at any time.

The environment is a local Python 3.12 virtual environment created with
**uv** and driven by **JupyterLab**. All datasets are local, so the notebooks
work without an internet connection after the first download.
"""),
    md("""
## Learning objectives

- Start JupyterLab and select the correct kernel.
- Understand the directory layout and what each notebook covers.
- Locate a dataset and load it from `data/`.
- Know the `uv` commands needed to maintain the environment.
- Find the official documentation for every library in the syllabus.
"""),
    md("""
## 1. Start the environment

From the project folder, run one command:

```bash
python setup.py
```

On Windows, use `py setup.py` if `python` is not on PATH.

That single script creates the virtual environment, installs Python 3.12 and
all libraries if needed, downloads the datasets, starts JupyterLab, and opens
this notebook in your browser at http://127.0.0.1:8888 (or the `PORT` you set).
No token or password is needed. The kernel is named **python3** and is chosen
automatically. Running the command again only starts JupyterLab.

To use a different port:

```bash
PORT=8890 python setup.py
```

PowerShell: `$env:PORT = 8890; python setup.py`
"""),
    md("""
## 2. Directory layout

```
DS/
├── .venv/                  # isolated Python 3.12 environment (do not edit)
├── .data/                  # NLTK and scikit-learn caches
├── .jupyter/               # Jupyter config, settings, runtime, IPython profile
├── .keras/                 # Keras configuration
├── .cache/matplotlib/      # Matplotlib font cache
├── pyproject.toml          # the dependency list you can edit
├── uv.lock                 # exact locked versions (managed by uv)
├── README.md               # quick reference
├── data/                   # all CSV datasets + dataset_index.csv
├── scripts/
│   ├── nbhelper.py          # notebook generator helper
│   ├── download_data.py     # dataset downloader
│   └── build_*.py           # scripts that generated the notebooks
└── notebooks/
    ├── 00_START_HERE.ipynb
    ├── lab_practice.ipynb
    ├── python-fundamentals/   # Units I–III
    ├── data-science/          # Units IV–V: core DS, ML, NLP, datasets
    └── deep-learning/         # TensorFlow, Keras, PyTorch
```
"""),
    md("""
## 3. Syllabus map — where to study each topic

| Syllabus unit | Syllabus topic | Notebook |
| --- | --- | --- |
| Unit I | Python introduction, data types, operators, strings, conditionals, loops, functions, recursion, lambda | `python-fundamentals/unit1_python_fundamentals.ipynb` |
| Unit II | Lists, tuples, sets, dictionaries, `filter/map/reduce/zip`, modules | `python-fundamentals/unit2_data_structures.ipynb` |
| Unit III | OOP, inheritance, constructors, exceptions, files, regular expressions, SQL, MySQL/PostgreSQL | `python-fundamentals/unit3_oops_files_regex_sql.ipynb` |
| Unit IV | NumPy, SciPy, pandas | `data-science/unit4_numpy_scipy_pandas.ipynb` |
| Unit IV | Matplotlib, Seaborn, SymPy, Statsmodels (plus Plotly/Bokeh) | `data-science/unit4_plotting_sympy_statsmodels.ipynb` |
| Unit V | Scikit-Learn and traditional ML | `data-science/unit5_scikit_learn.ipynb` |
| Unit V | TensorFlow, Keras, PyTorch | `deep-learning/unit5_deep_learning.ipynb` |
| Unit V | SQLAlchemy, NLTK, Gensim (plus TextBlob, DuckDB) | `data-science/unit5_nlp_databases.ipynb` |
| Reference | Datasets and loaders | `data-science/datasets_catalog.ipynb` |
| Lab work | 20 programs, 4 per unit group | `lab_practice.ipynb` |

Your syllabus also mentions **IPython, Jupyter and Google Colab**. This whole
setup is the local Jupyter alternative to Colab. Colab needs no installation —
it runs in a browser at https://colab.research.google.com.
"""),
    md("""
## 4. Environment check

Run this cell to confirm every major library imports and to see its version.
"""),
    code("""
from pathlib import Path
import sys

print("Python:", sys.version.split()[0])

packages = [
    "numpy", "pandas", "scipy", "sympy", "statsmodels",
    "matplotlib", "seaborn", "plotly", "bokeh",
    "sklearn", "xgboost", "lightgbm", "catboost", "imblearn",
    "tensorflow", "keras", "torch",
    "nltk", "gensim", "textblob", "sqlalchemy", "duckdb",
]

for name in packages:
    try:
        module = __import__(name)
        print(f"  {name:<14} {getattr(module, '__version__', 'installed')}")
    except Exception as error:
        print(f"  {name:<14} NOT AVAILABLE: {type(error).__name__}: {error}")
"""),
    md("""
## 5. Load your first dataset

This index notebook sits directly inside `notebooks/`, so `../data` points to the shared datasets. Grouped notebooks use `../../data` for the same folder:

```python
from pathlib import Path
DATA = Path("../data")
```
"""),
    code("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path("../data")

iris = pd.read_csv(DATA / "iris.csv")
print(iris.shape)
iris.head()
"""),
    code("""
sns.set_theme(style="whitegrid")
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species")
plt.title("Iris — first look")
plt.show()
"""),
    md("""
## 6. uv command cheat sheet

| Goal | Command |
| --- | --- |
| Install everything from `pyproject.toml` | `uv sync` |
| Add a package | `uv add package_name` |
| Remove a package | `uv remove package_name` |
| Install a one-off tool into the environment | `uv pip install package_name` |
| List installed packages | `uv pip list` |
| Update locked versions | `uv lock --upgrade` |
| Run a command in the environment | `uv run python script.py` |
| Start JupyterLab | `uv run jupyter lab --notebook-dir .` |
| Install another Python interpreter | `uv python install 3.13` |
| Activate the environment in a shell | `source .venv/bin/activate` |
"""),
    md("""
## 7. Python cheat sheet

| Task | Code |
| --- | --- |
| Import the core stack | `import numpy as np, pandas as pd, matplotlib.pyplot as plt` |
| Set a random seed | `np.random.seed(42)` or `rng = np.random.default_rng(42)` |
| Read a CSV | `pd.read_csv(DATA / "name.csv")` |
| Quick look at data | `df.head()`, `df.info()`, `df.describe()` |
| Select columns | `df[["a", "b"]]`, `df.loc[:, "a":"b"]`, `df.iloc[:, 0:2]` |
| Filter rows | `df[df["age"] > 30]` |
| Group and aggregate | `df.groupby("species")["sepal_length"].mean()` |
| Make a figure | `fig, ax = plt.subplots(figsize=(8, 5))` |
| Seaborn theme | `sns.set_theme(style="whitegrid", context="notebook")` |
| Train/test split | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` |
"""),
    md("""
## 8. Available datasets

`data/dataset_index.csv` is the complete list. Highlights:

| Dataset | File | Use it for |
| --- | --- | --- |
| Iris | `iris.csv` | classification, EDA, plotting |
| Diamonds | `diamonds.csv` | regression, large-data EDA, facets |
| Planets | `planets.csv` | missing values, categorical plots |
| Tips | `tips.csv` | Seaborn examples, group comparisons |
| Titanic | `titanic.csv` | classification with mixed types |
| Penguins | `penguins.csv` | classification, faceting |
| Anscombe | `anscombe.csv` | why visualisation matters |
| Wine / Breast cancer / Digits | `sk_wine.csv`, `sk_breast_cancer.csv`, `digits.csv` | ML benchmarks |
| Diabetes | `sk_diabetes.csv` | regression |
| Blobs / Moons / Circles | `blobs.csv`, `moons.csv`, `circles.csv` | clustering, non-linear models |

`data-science/datasets_catalog.ipynb` documents all of them plus the online repositories.
"""),
    md("""
## 9. Official documentation

- Python — https://docs.python.org/3/tutorial/
- IPython — https://ipython.readthedocs.io/en/stable/
- Jupyter — https://docs.jupyter.org/en/latest/
- NumPy — https://numpy.org/doc/stable/
- SciPy — https://docs.scipy.org/doc/scipy/tutorial/index.html
- pandas — https://pandas.pydata.org/docs/user_guide/index.html
- Matplotlib — https://matplotlib.org/stable/tutorials/index.html
- Seaborn — https://seaborn.pydata.org/tutorial.html
- SymPy — https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html
- Statsmodels — https://www.statsmodels.org/stable/user-guide.html
- scikit-learn — https://scikit-learn.org/stable/getting_started.html
- TensorFlow — https://www.tensorflow.org/tutorials
- Keras — https://keras.io/getting_started/
- PyTorch — https://pytorch.org/tutorials/beginner/basics/intro.html
- NLTK — https://www.nltk.org/book/
- Gensim — https://radimrehurek.com/gensim/
- SQLAlchemy — https://docs.sqlalchemy.org/en/20/tutorial/
- DuckDB — https://duckdb.org/docs/stable/clients/python/overview.html
"""),
    md("""
## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named ...` | Run `uv sync` to reinstall dependencies. |
| Kernel not found | Select **Python 3 (DS uv env)**, or run `uv run ipykernel install --user --name ds-env`. |
| A dataset file is missing | Re-run `.venv/bin/python scripts/download_data.py`. |
| Import of TensorFlow fails on a new machine | Re-run `uv sync`; check `uv pip list`. |
| `Permission denied` writing a plot | Nothing in the notebooks writes files by default. |
| Notebook runs slowly | Restart kernel and run cells in order; training examples are kept deliberately small. |
"""),
    md("""
## 11. Suggested study order

1. `python-fundamentals/unit1_python_fundamentals.ipynb`
2. `python-fundamentals/unit2_data_structures.ipynb`
3. `python-fundamentals/unit3_oops_files_regex_sql.ipynb`
4. `data-science/unit4_numpy_scipy_pandas.ipynb`
5. `data-science/unit4_plotting_sympy_statsmodels.ipynb`
6. `data-science/unit5_scikit_learn.ipynb`
7. `deep-learning/unit5_deep_learning.ipynb`
8. `data-science/unit5_nlp_databases.ipynb`
9. `data-science/datasets_catalog.ipynb` whenever you need data.

Each notebook contains short explanations, runnable examples, exercises with
solutions, and a cheat sheet at the end.
"""),
]

if __name__ == "__main__":
    write_notebook("notebooks/00_START_HERE.ipynb", cells)
