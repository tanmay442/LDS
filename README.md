# LDS — Local Data Science learning environment

A self-contained JupyterLab environment for learning Python and data science.
It follows the HBTU NWS-101/102 syllabus (Units I–V) and works on **Windows,
macOS and Linux**. Only Python needs to be installed beforehand.

## Quick start

```bash
cd LDS
python setup.py
```

On Windows, use `py setup.py` if `python` is not on PATH.

`setup.py` is the only script you need. On first run it:

1. Installs `uv` into a temporary virtual environment.
2. Installs Python 3.12 if your machine does not have it.
3. Creates `.venv/` and installs every library from `pyproject.toml`.
4. Downloads the datasets and NLTK corpora into the project folder.
5. Starts JupyterLab on `http://127.0.0.1:8888` with a dark theme, no token and
   no password, and opens `notebooks/00_START_HERE.ipynb` in your browser.

Later runs detect the existing environment and only start JupyterLab.

Use a different port if 8888 is busy:

```bash
PORT=8890 python setup.py          # macOS / Linux
```

```powershell
$env:PORT = 8890; python setup.py  # Windows PowerShell
```

The first run needs an internet connection. Everything after that works offline.
The server binds only to `127.0.0.1`, so it is not reachable from other machines.

## Start here

Open `notebooks/00_START_HERE.ipynb`. It contains the syllabus map, command
cheat sheets, the dataset list, and links to official documentation.

## Notebooks

`notebooks/python-fundamentals/` — learning Python itself (Units I–III)

| File | Syllabus unit | Contents |
| --- | --- | --- |
| `unit1_python_fundamentals.ipynb` | Unit I | Data types, operators, strings, conditionals, loops, functions, recursion, lambda |
| `unit2_data_structures.ipynb` | Unit II | Lists, tuples, sets, dicts, `filter/map/reduce/zip`, modules |
| `unit3_oops_files_regex_sql.ipynb` | Unit III | OOP, exceptions, files, regex, SQLite/SQLAlchemy; MySQL/PostgreSQL templates |

`notebooks/data-science/` — analysis, classical ML, NLP, databases (Units IV–V)

| File | Syllabus unit | Contents |
| --- | --- | --- |
| `unit4_numpy_scipy_pandas.ipynb` | Unit IV | NumPy, SciPy, pandas |
| `unit4_plotting_sympy_statsmodels.ipynb` | Unit IV | Matplotlib, Seaborn, Plotly, Bokeh, SymPy, Statsmodels |
| `unit5_scikit_learn.ipynb` | Unit V | ML workflow, preprocessing, models, evaluation, tuning, clustering, PCA |
| `unit5_nlp_databases.ipynb` | Unit V | NLTK, Gensim, TextBlob, SQLAlchemy, DuckDB |
| `datasets_catalog.ipynb` | reference | Every dataset and every way to load it |

`notebooks/deep-learning/` — neural networks (Unit V)

| File | Syllabus unit | Contents |
| --- | --- | --- |
| `unit5_deep_learning.ipynb` | Unit V | TensorFlow/Keras and PyTorch fundamentals |

`notebooks/` root

| File | Syllabus unit | Contents |
| --- | --- | --- |
| `00_START_HERE.ipynb` | overview | Startup, cheat sheets, syllabus map |
| `lab_practice.ipynb` | Lab work | 20 programs, 4 per unit group, with problem statements and solutions |

Every notebook has learning objectives, official documentation links, runnable
examples, exercises with solutions, and a cheat sheet.

## Data paths

Top-level notebooks use `Path("../data")`; notebooks in subfolders use
`Path("../../data")`. Both resolve to the project's `data/` folder from the
notebook's own directory. MySQL and PostgreSQL examples are markdown templates
that need a running server and credentials.

## Installed libraries

| Group | Packages |
| --- | --- |
| Jupyter | jupyterlab, ipykernel, ipython, nbformat, nbconvert, ipywidgets |
| Scientific core | numpy, pandas, scipy, sympy, statsmodels |
| Visualisation | matplotlib, seaborn, plotly, bokeh, kaleido |
| Machine learning | scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost, mlxtend, yellowbrick, shap, optuna, joblib |
| Deep learning | tensorflow, keras, tensorboard, torch (CPU build), torchvision |
| NLP | nltk, gensim, textblob, wordcloud |
| Databases | sqlalchemy, duckdb, psycopg2-binary, pymysql |
| Utilities | openpyxl, pyarrow, requests, tqdm, regex, missingno, kagglehub, tabulate, python-dotenv |

PyTorch is pinned to the CPU index. A CUDA build aborts the kernel when
TensorFlow is imported in the same notebook.

## Datasets

`data/` contains 22 CSV files from the
[seaborn example data repository](https://github.com/mwaskom/seaborn-data)
(including `iris.csv`, `diamonds.csv`, `planets.csv`, `tips.csv`, `titanic.csv`),
all scikit-learn toy datasets exported as CSV, and synthetic datasets from
`make_blobs`, `make_moons`, `make_circles`, `make_classification` and
`make_regression`. `data/dataset_index.csv` lists every file with row counts and
descriptions. `scripts/download_data.py --nltk` re-downloads anything missing.

## Everything stays in this folder

`setup.py` and `jupyter_config.py` point every cache and config directory inside
the project. Nothing is written to your home directory.

| Content | Location |
| --- | --- |
| NLTK corpora | `.data/nltk_data/` |
| scikit-learn downloads | `.data/scikit_learn_data/` |
| Keras config | `.keras/` |
| Matplotlib font cache | `.cache/matplotlib/` |
| Jupyter settings, state, runtime | `.jupyter/` |
| IPython profile and kernel startup filters | `.jupyter/ipython/` |
| Hugging Face cache | `.cache/huggingface/` |

Generated folders are listed in `.gitignore` and are not part of the repository.

## Verification

```bash
python3 scripts/validate_notebooks.py          # JSON, structure, Python syntax
.venv/bin/python scripts/execute_notebooks.py   # run every notebook end to end
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.
Executed copies with outputs are written to `reports/executed/`; the notebooks
in `notebooks/` stay clean and unexecuted. All 11 notebooks pass both checks with
zero error outputs and zero warning outputs.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `python` not found | Install Python 3.9+ from https://www.python.org/downloads/ and tick "Add Python to PATH" on Windows. |
| First run fails to download | Check your internet connection and run `python setup.py` again. |
| Port already in use | Use `PORT=8890 python setup.py`. |
| No browser opens | Paste the printed URL into your browser. |
| Notebook shows an import error | Delete `.venv/` and run `python setup.py` again. |
| Jupyter stops | Run `python setup.py` again. |

## Official documentation

- Python — https://docs.python.org/3/tutorial/
- NumPy — https://numpy.org/doc/stable/
- pandas — https://pandas.pydata.org/docs/user_guide/index.html
- SciPy — https://docs.scipy.org/doc/scipy/tutorial/index.html
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
