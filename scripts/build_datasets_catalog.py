"""Builder for notebooks/data-science/datasets_catalog.ipynb (stdlib only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nbhelper import code, md, write_notebook

cells = []

cells.append(md("""
# Dataset Catalogue and Data-Loading Reference

This notebook is the single place to look up **every dataset available in this environment and every way to load it**. It covers the local CSV library in `data/`, seaborn's sample-data repository, all six scikit-learn toy loaders, the synthetic `make_*` generators, OpenML fetching, and the URL patterns for external sources such as UCI, Kaggle, Hugging Face, Our World in Data and NASA. Each section states the exact function signature and the offline-safe way to reproduce the example.

The environment is a uv-managed Python 3.12 project. Grouped notebooks live in subfolders of `notebooks/`, and `data/` is two levels up from each notebook, and every dataset-reading cell begins with `from pathlib import Path` and `DATA = Path("../../data")`. All code cells run top-to-bottom without internet: network calls appear only inside markdown code fences.

## Learning objectives

- **Learning objectives:** after working through this notebook you can find, load and inspect any dataset in the environment.
- Load every local CSV with `pandas.read_csv` using `Path("../../data") / "name.csv"`, and inspect it with `head`, `shape`, `columns`, `dtypes` and `info`.
- Recall the full `pandas.read_csv` parameter set for custom delimiters, headers, column selection, dtypes, dates and missing values.
- List seaborn's 22 sample datasets, explain `get_dataset_names`, `load_dataset(name, cache=True)` and the cache directory, and load seaborn datasets from local copies offline.
- Use all six scikit-learn toy loaders and explain every `Bunch` attribute, plus the `as_frame=True` and `return_X_y=True` variants.
- Generate controlled data with the eight `make_*` generators and state what each one returns.
- Write `fetch_openml` calls with `name`, `data_id`, `as_frame` and describe the `~/scikit_learn_data` cache.
- Build URL templates for UCI, Kaggle/kagglehub, Hugging Face, World Bank, Our World in Data and the NASA Exoplanet Archive.
- Choose the right dataset for classification, regression, clustering, time series, NLP and visualization using the decision table.
- Add a new CSV to `data/` and document it reproducibly.
"""))

cells.append(md("""
## Official documentation & references

- [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) — the main local-CSV loader.
- [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) — `head`, `shape`, `columns`, `dtypes`, `info`.
- [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html) — sample-data loader.
- [seaborn-data GitHub repository](https://github.com/mwaskom/seaborn-data) — the 22 CSV files and their sources.
- [scikit-learn datasets](https://scikit-learn.org/stable/datasets.html) — overview of all dataset utilities.
- [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) — the six `load_*` functions.
- [scikit-learn synthetic datasets](https://scikit-learn.org/stable/datasets/generating_synthetic.html) — the `make_*` generators.
- [sklearn.datasets.fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) — OpenML access.
- [OpenML](https://www.openml.org/) — the dataset registry behind `fetch_openml`.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — the oldest public dataset archive.
- [Python pathlib](https://docs.python.org/3/library/pathlib.html) — `Path`, `/` operator, `glob`.
"""))

cells.append(md("""
## Roadmap

| Part | Topic | Key idea |
| --- | --- | --- |
| 1 | Local CSVs in `data/` | `DATA = Path("../../data")`, then `pd.read_csv(DATA / "name.csv")` |
| 2 | Seaborn sample data | `get_dataset_names`, `load_dataset`, offline via local CSVs |
| 3 | scikit-learn datasets | `Bunch`, six loaders, eight generators |
| 4 | OpenML | `fetch_openml(name=...)` / `fetch_openml(data_id=...)` |
| 5 | Other sources | UCI, Kaggle, Hugging Face, World Bank, OWID, NASA URL templates |
| 6 | Choosing a dataset | task-to-dataset decision table |
| 7 | Adding a dataset | download, save, document, verify |
"""))

cells.append(code('''
from pathlib import Path
DATA = Path("../../data")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets as skdatasets
from sklearn.datasets import get_data_home

print("pandas", pd.__version__, "| numpy", np.__version__, "| seaborn", sns.__version__)
print("Data folder:", DATA.resolve())
'''))

cells.append(md("""
---

# Part 1 — The local CSV library in `data/`

`DATA = Path("../../data")` works because a grouped notebook's working directory is its own subfolder inside `notebooks/`. The `/` operator joins path pieces, and `glob("*.csv")` discovers files without hard-coding a list. A missing file is reported, not treated as an error, so this section still runs while other agents are still populating `data/`.
"""))

cells.append(code('''
csv_files = sorted(DATA.glob("*.csv"))
print(f"Folder: {DATA.resolve()}")
print(f"CSV files found: {len(csv_files)}")
for path in csv_files:
    print(f"  {path.name:>24}  {path.stat().st_size / 1024:8.1f} KB")
'''))

cells.append(md("""
### Complete inventory of the local CSVs

These are the 26 CSV files provided in `data/`. Four of them (`wine`, `breast_cancer`, `diabetes`, `digits`) are also scikit-learn toy datasets, stored locally so they can be used offline.

| File | One-line description | Typical use |
| --- | --- | --- |
| `iris.csv` | 150 irises with sepal/petal measurements and species | classification, clustering, EDA |
| `diamonds.csv` | ~54k diamond prices plus the 4C attributes and cut/colour/clarity | regression, categorical EDA |
| `planets.csv` | NASA exoplanet candidates with orbital and discovery attributes | scatter plots, filtering |
| `tips.csv` | Restaurant bill, tip, sex, smoker, day, time, party size | small categorical EDA |
| `titanic.csv` | Passenger survival, class, sex, age, fare and embarkation | classification, feature engineering |
| `penguins.csv` | Penguin species, island, flipper/bill measurements, sex, year | classification, visualization |
| `mpg.csv` | Car fuel economy with weight, horsepower, origin, model year | regression, grouped summaries |
| `flights.csv` | Monthly number of flights per year and airline | time series, line plots |
| `fmri.csv` | fMRI signal by timepoint, event, region and subject | repeated measures, line plots |
| `anscombe.csv` | Four x/y groups with identical regression statistics | anscombe's quartet, visualization |
| `car_crashes.csv` | US state crash rates with alcohol and speeding indicators | regression, correlation |
| `geyser.csv` | Old Faithful eruption duration and waiting time | bivariate distribution |
| `seaice.csv` | Arctic sea-ice extent by year, month and hemisphere | time series, heatmaps |
| `dowjones.csv` | Daily Dow Jones Industrial Average values across decades | time series, resampling |
| `exercise.csv` | Exercise/heart-rate measurements under conditions | repeated measures |
| `attention.csv` | Attention scores under conditions and age groups | psychology experiments |
| `brain_networks.csv` | Wide-format fMRI brain-region connectivity matrix | network/heatmap visualization |
| `dots.csv` | Dot-motion psychophysics judgments (condition, subject, response) | reaction-time EDA |
| `glue.csv` | Scores from the GLUE natural-language benchmark tasks | model comparison tables |
| `healthexp.csv` | Life expectancy vs health spending by country and year | scatter plots, regression |
| `anagrams.csv` | Anagram solution times by team size and condition | experimental psychology |
| `taxis.csv` | NYC taxi trips with pickup/dropoff, fare, tips and duration | geospatial and time EDA |
| `wine.csv` | 178 wines with 13 chemical features and cultivar class | multiclass classification |
| `breast_cancer.csv` | 569 tumours with 30 cell-nucleus features and diagnosis | binary classification |
| `diabetes.csv` | 442 patients with 10 features and disease progression | regression |
| `digits.csv` | 1797 handwritten-digit images as 64 pixel features | multiclass classification |

### A safe loader helper

`load_local` centralises the path construction and the missing-file message. Passing `**kwargs` through to `pd.read_csv` means every option from the next section still works.
"""))

cells.append(code('''
def load_local(name, **kwargs):
    """Load ../../data/<name>.csv, or return None with a clear message if absent."""
    path = DATA / f"{name}.csv"
    if not path.exists():
        print(f"NOT YET AVAILABLE: {path}")
        return None
    return pd.read_csv(path, **kwargs)


iris_local = load_local("iris")
print("Type:", type(iris_local))
'''))

cells.append(md("""
### Inspecting a dataset: the five standard questions

1. `head()` — what do the first rows look like?
2. `shape` — how many rows and columns?
3. `columns` — what are the exact column names (needed for indexing)?
4. `dtypes` — which columns are numeric, text or datetime?
5. `info()` — a compact summary of dtypes and non-null counts.

Apply them to `iris.csv` below.
"""))

cells.append(code('''
iris_local = load_local("iris")
if iris_local is not None:
    print("--- head() ---")
    print(iris_local.head())
    print("--- shape ---", iris_local.shape)
    print("--- columns ---", list(iris_local.columns))
    print("--- dtypes ---")
    print(iris_local.dtypes)
    print("--- info() ---")
    iris_local.info()
'''))

cells.append(md("""
`titanic.csv` is a good mixed-type example: numeric columns (`age`, `fare`) sit next to categorical strings (`sex`, `embarked`) and missing values. `select_dtypes` separates them.
"""))

cells.append(code('''
titanic_local = load_local("titanic")
if titanic_local is not None:
    print("shape:", titanic_local.shape)
    print(titanic_local.head())
    print(titanic_local.dtypes)
    print("numeric columns:", list(titanic_local.select_dtypes("number").columns))
    print("object columns:", list(titanic_local.select_dtypes(include=["object", "str"]).columns))
    print("missing per column:")
    print(titanic_local.isna().sum())
'''))

cells.append(md("""
`diamonds.csv` is the largest local table, so print only a compact profile instead of all rows.
"""))

cells.append(code('''
diamonds_local = load_local("diamonds")
if diamonds_local is not None:
    print("shape:", diamonds_local.shape)
    print(diamonds_local.head(3))
    print(diamonds_local.dtypes)
    print("price summary:")
    print(diamonds_local["price"].describe())
'''))

cells.append(md("""
### One-glance profile of the main datasets

Building a small profile table is faster than opening each file by hand. This is the pattern to reuse when you add a dataset later.
"""))

cells.append(code('''
profile_names = ["iris", "titanic", "diamonds", "penguins", "tips", "mpg", "flights", "wine", "breast_cancer"]
rows = []
for name in profile_names:
    frame = load_local(name)
    if frame is not None:
        rows.append({
            "dataset": name,
            "rows": len(frame),
            "cols": frame.shape[1],
            "numeric_cols": frame.select_dtypes("number").shape[1],
            "object_cols": frame.select_dtypes(include=["object", "str"]).shape[1],
            "memory_KB": round(frame.memory_usage(deep=True).sum() / 1024, 1),
        })
profile = pd.DataFrame(rows)
print(profile.to_string(index=False))
'''))

cells.append(md("""
### `pandas.read_csv` parameters worth knowing

`pd.read_csv(filepath_or_buffer, *, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None, dtype=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None, encoding_errors='strict', dialects=None, on_bad_lines='error', delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None, dtype_backend=<no_default>)`.

| Parameter | Purpose |
| --- | --- |
| `sep` / `delimiter` | Column separator; use `'\t'` for TSV, `';'` for European CSV |
| `header` | Row used as column names; `None` when the file has no header |
| `names` | Supply column names yourself (pair with `header=None`) |
| `index_col` | Column number/name to use as the row index |
| `usecols` | Read only a subset of columns, saving memory |
| `dtype` | Force types, e.g. `{'id': str, 'age': 'float32'}` |
| `parse_dates` | Parse columns to `datetime64`, e.g. `['date']` |
| `dayfirst` | Interpret ambiguous dates as day-first |
| `na_values` | Extra strings to treat as missing, e.g. `['?', 'NA']` |
| `nrows` / `skiprows` / `skipfooter` | Read a slice of a large file |
| `comment` | Ignore lines starting with this character |
| `thousands` / `decimal` | Number formatting, e.g. `','` and `'.'` |
| `encoding` | File encoding, e.g. `'utf-8'`, `'latin-1'` |
| `chunksize` | Return an iterator of DataFrames for out-of-core reading |
| `on_bad_lines` | `'error'`, `'warn'` or `'skip'` for malformed rows |
| `compression` | `'infer'`, `'gzip'`, `'zip'`, `'bz2'` for compressed files |
"""))

cells.append(code('''
iris_path = DATA / "iris.csv"
if iris_path.exists():
    print("nrows=3 ->", pd.read_csv(iris_path, nrows=3).shape)
    print("usecols=[0, 1] ->", pd.read_csv(iris_path, usecols=[0, 1]).shape)
    print("header=None, names= ->")
    print(pd.read_csv(iris_path, header=None, names=["a", "b", "c", "d", "species"], nrows=2))
    print("dtype on one column ->", pd.read_csv(iris_path, dtype={"sepal_length": "float32"}).dtypes.iloc[0])
else:
    print("Place iris.csv in ../../data to run the pd.read_csv option demos.")
'''))

cells.append(md("""
---

# Part 2 — Seaborn's sample-data repository

Seaborn ships a helper for small example datasets. **It requires internet** for two reasons: `get_dataset_names()` reads an index file from the repository, and `load_dataset()` downloads `<name>.csv` when it is not cached. The repository is [github.com/mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data); the raw URL pattern is `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/<name>.csv`.

```python
# Requires internet — shown as text so this notebook stays fully offline-runnable.
import seaborn as sns

names = sns.get_dataset_names()          # list of 22 names from the online index
tips = sns.load_dataset("tips")          # downloads tips.csv on first use
tips = sns.load_dataset("tips", cache=True)      # explicit default: cache after download
tips = sns.load_dataset("tips", cache=False)     # always fetch fresh, never cache
tips = sns.load_dataset("tips", data_home="/tmp/seaborn-cache")  # custom cache dir
tips = sns.load_dataset("tips", usecols=["total_bill", "tip"])   # **kws go to pd.read_csv
print(sns.get_data_home())               # where the cache lives
```

`load_dataset(name, cache=True, data_home=None, **kws)` returns a `pandas.DataFrame`. Extra keyword arguments such as `usecols` or `nrows` are passed straight through to `pd.read_csv`. In seaborn 0.13 the cache directory comes from `sns.get_data_home()` (on Linux usually `~/.cache/seaborn`; older releases used `~/seaborn-data`).

The cell below calls `get_dataset_names()` when a connection exists and otherwise falls back to the bundled list, so it always runs.
"""))

cells.append(code('''
SEABORN_NAMES_OFFLINE = [
    "anagrams", "anscombe", "attention", "brain_networks", "car_crashes",
    "diamonds", "dots", "dowjones", "exercise", "flights", "fmri", "geyser",
    "glue", "healthexp", "iris", "mpg", "penguins", "planets", "seaice",
    "taxis", "tips", "titanic",
]

try:
    sns_names = sns.get_dataset_names()
    print(f"Fetched {len(sns_names)} names from the seaborn-data index (online).")
except Exception as exc:
    sns_names = SEABORN_NAMES_OFFLINE
    print(f"Offline ({type(exc).__name__}) - using the bundled list of {len(sns_names)} names.")

print(sorted(sns_names))
'''))

cells.append(md("""
### All 22 seaborn datasets

The descriptions below are distilled from the official [seaborn-data README](https://github.com/mwaskom/seaborn-data).

| Name | One-line description | Typical use |
| --- | --- | --- |
| `anagrams` | Anagram solving times by team size and condition | experimental psychology, catplot |
| `anscombe` | Four x/y groups with equal regression statistics | anscombe's quartet, visualization |
| `attention` | Attention task scores by condition and age group | repeated measures |
| `brain_networks` | Wide-format fMRI region connectivity matrix | heatmap, network EDA |
| `car_crashes` | US state crash rates with alcohol and speeding factors | regression, correlation |
| `diamonds` | Diamond prices plus carat, cut, colour, clarity | regression, categorical EDA |
| `dots` | Dot-motion psychophysics responses by condition | reaction-time EDA |
| `dowjones` | Daily Dow Jones average over many decades | time series, resampling |
| `exercise` | Exercise and heart-rate readings under conditions | repeated measures |
| `flights` | Monthly flight counts by year and airline | line plots, time series |
| `fmri` | Signal by timepoint, event, region and subject | lineplot with error bands |
| `geyser` | Old Faithful eruption duration and waiting time | bivariate distribution |
| `glue` | GLUE benchmark scores across NLP tasks | model comparison tables |
| `healthexp` | Life expectancy vs health spending by country/year | scatter, regression |
| `iris` | 150 irises, four measurements and species | classification, pairplot |
| `mpg` | Car fuel economy, weight, horsepower, origin, year | regression, grouping |
| `penguins` | Species, island, bill/flipper measurements, sex | classification, visualization |
| `planets` | Exoplanet orbital and discovery attributes | scatter, filtering |
| `seaice` | Arctic sea-ice extent by year, month and hemisphere | time series, heatmap |
| `taxis` | NYC taxi trips with fares, tips and durations | geospatial and time EDA |
| `tips` | Restaurant bill, tip and categorical context | small categorical EDA |
| `titanic` | Passenger survival, class, sex, age, fare | classification, EDA |
"""))

cells.append(md("""
### Offline equivalent: loop over the local copies

Every seaborn dataset also exists as a local CSV in `data/` (and `wine`, `breast_cancer`, `diabetes`, `digits` are scikit-learn extras). `pd.read_csv(DATA / f"{name}.csv")` reproduces `sns.load_dataset(name)` without a network call. Note that seaborn applies a small amount of categorical reordering in `load_dataset`, which the raw CSV does not.
"""))

cells.append(code('''
for name in ["iris", "tips", "titanic", "planets", "flights"]:
    frame = load_local(name)
    if frame is not None:
        print(f"{name:>10}: shape={frame.shape}, columns={list(frame.columns)}")
'''))

cells.append(md("""
---

# Part 3 — scikit-learn datasets

## The `Bunch` object

Each `load_*` function returns a `sklearn.utils.Bunch`: a dict-like object whose keys are also available as attributes. Printing `bunch.keys()` is the fastest way to see what a loader offers.
"""))

cells.append(code('''
iris_bunch = skdatasets.load_iris()
print("type:", type(iris_bunch))
print("keys:", list(iris_bunch.keys()))
print("data shape:", iris_bunch.data.shape, "| target shape:", iris_bunch.target.shape)
print("feature_names:", iris_bunch.feature_names)
print("target_names:", iris_bunch.target_names.tolist())
print("DESCR first lines:")
print("\\n".join(iris_bunch.DESCR.splitlines()[:6]))
'''))

cells.append(md("""
### `Bunch` attributes

| Attribute | Always present? | Meaning |
| --- | --- | --- |
| `data` | yes | Feature matrix, shape `(n_samples, n_features)`; ndarray by default |
| `target` | yes | Labels, shape `(n_samples,)` or `(n_samples, n_targets)` for multi-output |
| `feature_names` | yes | List of column names for `data` |
| `target_names` | for classifiers | Names of the classes; continuous regression targets may omit it |
| `DESCR` | yes | Full dataset description, provenance and references |
| `frame` | only with `as_frame=True` | A pandas DataFrame holding features **and** target |
| `filename` | some loaders | Path to the compressed source file shipped with scikit-learn |

Bunch can be unpacked like a dict: `X, y = bunch["data"], bunch["target"]`.

### `as_frame=True` and `return_X_y=True`

- `as_frame=False` (default) → NumPy arrays.
- `as_frame=True` → `data` and `target` become pandas objects, and a new `frame` attribute appears.
- `return_X_y=True` → the loader returns the tuple `(X, y)` instead of a Bunch. It combines with `as_frame=True`.
"""))

cells.append(code('''
iris_frame_bunch = skdatasets.load_iris(as_frame=True)
print("keys now:", list(iris_frame_bunch.keys()))
print("frame type:", type(iris_frame_bunch.frame))
print("data type:", type(iris_frame_bunch.data), "| target type:", type(iris_frame_bunch.target))
print(iris_frame_bunch.frame.head())
'''))

cells.append(code('''
X, y = skdatasets.load_iris(return_X_y=True)
print("arrays:", type(X).__name__, X.shape, "|", type(y).__name__, y.shape)
X_df, y_series = skdatasets.load_iris(return_X_y=True, as_frame=True)
print("frame parts:", type(X_df).__name__, X_df.shape, "|", type(y_series).__name__, y_series.shape)
'''))

cells.append(md("""
### The six toy loaders

| Loader | Samples | Features | Task | Notes |
| --- | --- | --- | --- | --- |
| `load_iris` | 150 | 4 | 3-class classification | classic small dataset |
| `load_diabetes` | 442 | 10 | regression | target is a continuous progression score |
| `load_digits` | 1797 | 64 | 10-class classification | 8x8 pixel images, also in `.images` |
| `load_wine` | 178 | 13 | 3-class classification | chemical analysis of Italian wines |
| `load_breast_cancer` | 569 | 30 | binary classification | target 0 = malignant, 1 = benign |
| `load_linnerud` | 20 | 3 | multi-output regression | 3 exercise features, 3 physiological targets |
"""))

cells.append(code('''
loader_names = ["load_iris", "load_diabetes", "load_digits", "load_wine",
                "load_breast_cancer", "load_linnerud"]
records = []
for loader_name in loader_names:
    bunch = getattr(skdatasets, loader_name)()
    records.append({
        "loader": loader_name,
        "n_samples": bunch.data.shape[0],
        "n_features": bunch.data.shape[1],
        "n_targets": bunch.target.shape[1] if bunch.target.ndim > 1 else 1,
        "n_classes": len(bunch.target_names) if hasattr(bunch, "target_names") else "-",
        "DESCR_first_line": bunch.DESCR.strip().splitlines()[0][:60],
    })
print(pd.DataFrame(records).to_string(index=False))
'''))

cells.append(md("""
`load_digits` is special: besides the flattened `data` it exposes `.images` with shape `(n_samples, 8, 8)`, which is convenient for image plotting.
"""))

cells.append(code('''
digits = skdatasets.load_digits()
print("data:", digits.data.shape, "| images:", digits.images.shape, "| target:", digits.target.shape)
print("feature_names[:6]:", digits.feature_names[:6])
digits_frame = skdatasets.load_digits(as_frame=True).frame
print("frame columns[:6]:", list(digits_frame.columns[:6]))
print(digits_frame.iloc[:3, :6])
'''))

cells.append(md("""
The remaining regression and classification loaders in one pass. Notice that `load_diabetes` has **no** `target_names`, while `load_linnerud` has three targets.
"""))

cells.append(code('''
for loader_name in ["load_diabetes", "load_wine", "load_breast_cancer", "load_linnerud"]:
    bunch = getattr(skdatasets, loader_name)()
    print(f"\\n{loader_name}: data={bunch.data.shape}, target={bunch.target.shape}")
    print("  feature_names:", bunch.feature_names)
    print("  target_names:", getattr(bunch, "target_names", "not provided (continuous target)"))
'''))

cells.append(md("""
> **Note on Boston housing.** `load_boston` was deprecated in scikit-learn 1.0 and **removed in 1.2** because of ethical concerns about one of its features. It is not available here. If you need it, use `fetch_openml(name="boston", version=1, as_frame=True)` or a local CSV; discuss its limitations before modelling.
"""))

cells.append(md("""
## Synthetic data: the `make_*` generators

Generators return plain NumPy arrays `(X, y)` (sometimes an extra array). They are ideal when you need a known ground truth, a chosen difficulty level, or more rows than a toy dataset provides. Every generator accepts `random_state` for reproducibility.

Parameter notes:

- `make_classification(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, n_clusters_per_class, weights, flip_y, class_sep, shuffle, random_state)` → `(X, y)`. `flip_y` adds label noise and `class_sep` controls how separable the classes are.
- `make_regression(n_samples, n_features, n_informative, n_targets, bias, effective_rank, tail_strength, noise, shuffle, coef, random_state)` → `(X, y)`, or `(X, y, coef)` when `coef=True`.
- `make_blobs(n_samples, n_features, centers, cluster_std, center_box, shuffle, random_state, return_centers)` → `(X, y)`, or `(X, y, centers)` when `return_centers=True`.
- `make_moons(n_samples, noise, random_state)` → `(X, y)` with two features and two non-linearly separable classes.
- `make_circles(n_samples, factor, noise, random_state)` → `(X, y)`; `factor` is the inner/outer radius ratio.
- `make_swiss_roll(n_samples, noise, random_state)` → `(X, t)`, a 2D manifold in 3D where `t` is the position along the roll.
- `make_multilabel_classification(n_samples, n_features, n_classes, n_labels, length, allow_unlabeled, sparse, return_indicator, random_state)` → `(X, Y)` with a binary indicator matrix `Y`.
- `make_sparse_uncorrelated(n_samples, n_features, random_state)` → `(X, y)` with a sparse, uncorrelated design matrix.

### Generator quick reference

| Generator | Returns | Main use |
| --- | --- | --- |
| `make_classification` | `(X, y)` | controllable multiclass classification |
| `make_regression` | `(X, y)` or `(X, y, coef)` | regression with known coefficients |
| `make_blobs` | `(X, y)` or `(X, y, centers)` | Gaussian clustering clusters |
| `make_moons` | `(X, y)` | non-linear two-class separation |
| `make_circles` | `(X, y)` | concentric non-linear classes |
| `make_swiss_roll` | `(X, t)` | manifold / dimensionality reduction |
| `make_multilabel_classification` | `(X, Y)` | multilabel classification |
| `make_sparse_uncorrelated` | `(X, y)` | sparse regression design |
"""))

cells.append(code('''
# Classification: informative vs redundant features, class imbalance and label noise.
X, y = skdatasets.make_classification(
    n_samples=200, n_features=20, n_informative=5, n_redundant=2,
    n_classes=3, n_clusters_per_class=1, class_sep=1.5,
    weights=[0.7, 0.2, 0.1], flip_y=0.02, random_state=42,
)
print("classification X:", X.shape, "| y:", y.shape, "| counts:", np.bincount(y))

# Regression: coef=True also returns the true coefficients used to build y.
X, y, coef = skdatasets.make_regression(
    n_samples=100, n_features=5, n_informative=3, noise=8.0,
    bias=2.5, coef=True, random_state=0,
)
print("regression X:", X.shape, "| y:", y.shape, "| coef:", np.round(coef, 2))

# Blobs: Gaussian clusters with known centres.
X, y, centers = skdatasets.make_blobs(
    n_samples=300, n_features=2, centers=3, cluster_std=0.8,
    return_centers=True, random_state=7,
)
print("blobs X:", X.shape, "| y:", y.shape, "| centers:", centers.shape,
      "| counts:", np.bincount(y))
'''))

cells.append(code('''
# Two low-dimensional non-linearly separable datasets.
for maker in [skdatasets.make_moons, skdatasets.make_circles]:
    features, labels = maker(n_samples=200, noise=0.07, random_state=1)
    print(maker.__name__, features.shape, labels.shape, "counts:", np.bincount(labels))

features, labels = skdatasets.make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=1)
print("make_circles(factor=0.5) ->", features.shape, labels.shape)

# Swiss roll: t is a colour axis along the manifold.
X, t = skdatasets.make_swiss_roll(n_samples=1000, noise=0.05, random_state=3)
print("swiss_roll X:", X.shape, "| t:", t.shape,
      "| t range:", round(t.min(), 2), "to", round(t.max(), 2))
'''))

cells.append(code('''
# Multilabel targets: several classes can be active for one sample.
X, Y = skdatasets.make_multilabel_classification(
    n_samples=150, n_features=20, n_classes=4, n_labels=2,
    allow_unlabeled=False, random_state=5,
)
print("multilabel X:", X.shape, "| Y:", Y.shape,
      "| mean labels/sample:", round(Y.sum(axis=1).mean(), 2))

# Sparse uncorrelated design for regression experiments.
Xs, ys = skdatasets.make_sparse_uncorrelated(n_samples=100, n_features=10, random_state=11)
print("sparse_uncorrelated X:", Xs.shape, "| y:", ys.shape,
      "| nonzero entries:", np.count_nonzero(Xs))
'''))

cells.append(md("""
---

# Part 4 — OpenML with `fetch_openml`

OpenML is an online registry of thousands of datasets. scikit-learn's fetcher caches downloads under `~/scikit_learn_data` by default (change with `data_home`). All calls below are shown inside a code fence because they need internet.

```python
from sklearn.datasets import fetch_openml

# Fetch by human-readable name (optionally pin a version).
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data, mnist.target.astype(int)

# Fetch by numeric OpenML dataset id — unambiguous when names collide.
adult = fetch_openml(data_id=2, as_frame=True, parser="auto")

# Return (X, y) directly, and control caching / retries.
X, y = fetch_openml(name="iris", version=1, return_X_y=True,
                    as_frame=True, cache=True, n_retries=3, delay=1.0)

# Where the cache lives.
from sklearn.datasets import get_data_home
print(get_data_home())       # ~/scikit_learn_data
```

Parameter highlights: `name`, `version` (default `'active'`), `data_id`, `as_frame` (default `'auto'`), `parser` (`'auto'`, `'pandas'`, `'liac-arff'`), `target_column`, `return_X_y`, `cache`, `data_home`, `n_retries`, `delay`, `read_csv_kwargs`. When `as_frame=True`, the result is a `Bunch` whose `frame` contains features plus target, exactly like the toy loaders. OpenML pages live at `https://www.openml.org/d/<data_id>`.
"""))

cells.append(code('''
print("sklearn data home:", get_data_home())
print("seaborn cache home:", sns.get_data_home())
'''))

cells.append(md("""
---

# Part 5 — Other data sources and their URL templates

None of the following cells download anything; they only print the URL patterns you can paste into `pd.read_csv`, `requests.get`, or a dedicated loader.

## UCI Machine Learning Repository

Homepage: [archive.ics.uci.edu](https://archive.ics.uci.edu/). Dataset pages follow `https://archive.ics.uci.edu/dataset/<id>/<slug>` (for example the [iris dataset](https://archive.ics.uci.edu/dataset/53/iris)). The `ucimlrepo` package offers `fetch_ucirepo(id=53)` and returns `.data.features` and `.data.targets`.

```python
# Requires the optional ucimlrepo package.
from ucimlrepo import fetch_ucirepo
iris = fetch_ucirepo(id=53)
X, y = iris.data.features, iris.data.targets
```

## Kaggle / kagglehub

Dataset pages: `https://www.kaggle.com/datasets/<owner>/<slug>`. The `kagglehub` package (installed here) downloads without the CLI when credentials are configured.

```python
import kagglehub
path = kagglehub.dataset_download("uciml/iris")   # returns a local folder path
competition = kagglehub.competition_download("titanic")
```

## Hugging Face datasets

Pages: `https://huggingface.co/datasets/<org>/<name>`. The `datasets` package is **not** installed in this environment, so the example is text only.

```python
# Would require: pip install datasets
from datasets import load_dataset
ds = load_dataset("rajpurkar/squad", split="train")
```

## World Bank and Our World in Data

- World Bank API: `https://api.worldbank.org/v2/country/<ISO2>/indicator/<INDICATOR>?format=json`, e.g. `NY.GDP.MKTP.CD` for GDP.
- Our World in Data grapher CSV: `https://ourworldindata.org/grapher/<slug>.csv`, e.g. `life-expectancy.csv`.

```python
import pandas as pd
gdp = pd.read_json("https://api.worldbank.org/v2/country/IND/indicator/NY.GDP.MKTP.CD?format=json")
life = pd.read_csv("https://ourworldindata.org/grapher/life-expectancy.csv")
```

## NASA Exoplanet Archive

The TAP service accepts ADQL queries and returns CSV. The seaborn `planets` dataset was derived from this catalogue.

```python
import urllib.parse, pandas as pd
adql = "SELECT TOP 20 pl_name, disc_year, sy_dist FROM ps WHERE default_flag=1"
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + urllib.parse.quote(adql) + "&format=csv"
planets = pd.read_csv(url)
```
"""))

cells.append(code('''
url_templates = {
    "OpenML page": "https://www.openml.org/d/{data_id}",
    "UCI page": "https://archive.ics.uci.edu/dataset/{dataset_id}/{slug}",
    "Kaggle dataset": "https://www.kaggle.com/datasets/{owner}/{slug}",
    "Hugging Face dataset": "https://huggingface.co/datasets/{org}/{name}",
    "OWID grapher CSV": "https://ourworldindata.org/grapher/{slug}.csv",
    "World Bank API": "https://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}?format=json",
    "NASA Exoplanet TAP": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={adql}&format=csv",
}
for label, template in url_templates.items():
    print(f"{label:24} {template}")
'''))

cells.append(md("""
---

# Part 6 — Which dataset should I use?

| Task | Recommended datasets | Why |
| --- | --- | --- |
| Classification | `iris`, `digits`, `wine`, `breast_cancer`, `titanic` | labelled classes, clean, small enough to iterate quickly |
| Regression | `diabetes`, `diamonds` (`price` as target), `mpg` (`mpg` as target) | continuous targets with mixed feature types |
| Clustering | `make_blobs`, `iris` with the `species` column dropped | known cluster structure with/without ground truth |
| Time series | `dowjones`, `flights`, `seaice`, `fmri` | ordered by date/time or timepoint |
| NLP | text corpora built in the notebook (NLTK / gensim) | raw text needs tokenization before modelling |
| Visualization practice | `tips`, `fmri`, `planets`, `penguins`, `anscombe` | categorical, repeated-measures and quartet examples |

Two cautions: `diamonds` has ~54k rows, so start with `nrows=2000` while experimenting; and the old Boston housing dataset is unavailable in current scikit-learn — use OpenML or a local CSV if you need it.
"""))

cells.append(code('''
task_to_datasets = {
    "classification": ["iris", "digits", "wine", "breast_cancer", "titanic"],
    "regression": ["diabetes", "diamonds.price", "mpg", "boston_via_openml"],
    "clustering": ["make_blobs", "iris_without_target"],
    "time_series": ["dowjones", "flights", "seaice", "fmri"],
    "nlp": ["NLTK / gensim corpora built in notebook"],
    "visualization": ["tips", "fmri", "planets", "penguins", "anscombe"],
}
for task, choices in task_to_datasets.items():
    print(f"{task:15} -> {', '.join(choices)}")
'''))

cells.append(md("""
---

# Part 7 — Try it yourself

### Try it yourself 1 — Local file inspection

Using only local files, load `penguins.csv` with `load_local`. Print its shape, column names, dtypes and the number of missing values per column. Then report which column has the most missing values.
"""))

cells.append(code('''
# Solution 1
penguins = load_local("penguins")
if penguins is not None:
    print("shape:", penguins.shape)
    print("columns:", list(penguins.columns))
    print(penguins.dtypes)
    missing = penguins.isna().sum()
    print("missing per column:")
    print(missing)
    print("most missing:", missing.idxmax(), "->", int(missing.max()))
'''))

cells.append(md("""
### Try it yourself 2 — Compare two classifiers' loaders

Use `load_wine()` and `load_breast_cancer()`. Print which has more features and both `target_names`. Then load each with `as_frame=True` and display the first row of each `frame`.
"""))

cells.append(code('''
# Solution 2
wine = skdatasets.load_wine()
cancer = skdatasets.load_breast_cancer()
print("wine features:", wine.data.shape[1], "| cancer features:", cancer.data.shape[1])
print("wine target_names:", wine.target_names.tolist())
print("cancer target_names:", cancer.target_names.tolist())
wine_frame = skdatasets.load_wine(as_frame=True).frame
cancer_frame = skdatasets.load_breast_cancer(as_frame=True).frame
print("wine first row (first 6 columns):")
print(wine_frame.iloc[0, :6].to_string())
print("cancer first row (first 6 columns):")
print(cancer_frame.iloc[0, :6].to_string())
'''))

cells.append(md("""
### Try it yourself 3 — Generate and plot a non-linear dataset

Generate a two-moons dataset with 400 samples, `noise=0.1` and `random_state=42`. Print the class counts and plot the points with matplotlib, colouring by class.
"""))

cells.append(code('''
# Solution 3
moons_X, moons_y = skdatasets.make_moons(n_samples=400, noise=0.1, random_state=42)
print("class counts:", np.bincount(moons_y))
plt.figure(figsize=(5, 4))
plt.scatter(moons_X[:, 0], moons_X[:, 1], c=moons_y, cmap="coolwarm", s=15)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.title("make_moons(noise=0.1)")
plt.tight_layout()
plt.show()
'''))

cells.append(md("""
---

# Adding a new dataset to `data/` — reproducible recipe

1. **Acquire** the file from an official source. Use one of the URL templates from Part 5, `kagglehub.dataset_download`, `fetch_openml`, or a manual download. Record the exact source URL.
2. **Save** it as `data/<snake_case_name>.csv`, using UTF-8 and a header row. No notebooks may write outside `notebooks/` and `data/`.
3. **Verify** it loads: `pd.read_csv(DATA / "new_name.csv")`, then check `head`, `shape`, `columns`, `dtypes` and missing values.
4. **Document** it by adding one row to the inventory table in Part 1 (description and typical use) and by keeping a source/licence note next to the file.
5. **Reuse** it with `load_local("new_name")`; no path logic needs to change anywhere else.

The helper below prints the record to paste into the catalogue.
"""))

cells.append(code('''
def document_new_dataset(name, source_url, license_note, notes=""):
    """Verify a new local CSV and return its catalogue record."""
    path = DATA / f"{name}.csv"
    frame = pd.read_csv(path) if path.exists() else None
    print(f"file exists: {path.exists()} -> {path}")
    if frame is not None:
        print("shape:", frame.shape)
        print("dtypes:")
        print(frame.dtypes)
    return {
        "file": f"{name}.csv",
        "source": source_url,
        "license": license_note,
        "rows": None if frame is None else len(frame),
        "columns": None if frame is None else list(frame.columns),
        "notes": notes,
    }


record = document_new_dataset(
    "iris",
    "https://archive.ics.uci.edu/dataset/53/iris",
    "CC BY 4.0",
    "example entry only - the file is already in the catalogue",
)
print(record)
'''))

cells.append(md("""
---

# Cheat sheet / key takeaways

| Need | Use |
| --- | --- |
| Local CSV | `DATA = Path("../../data")`; `pd.read_csv(DATA / "name.csv")` |
| Discover local files | `sorted(DATA.glob("*.csv"))` |
| Inspect a table | `head()`, `shape`, `columns`, `dtypes`, `info()`, `isna().sum()` |
| Seaborn list | `sns.get_dataset_names()` (online, 22 names) |
| Seaborn load | `sns.load_dataset(name, cache=True, data_home=None, **kws)` |
| Seaborn offline | `pd.read_csv(DATA / f"{name}.csv")` |
| Bunch contents | `data`, `target`, `feature_names`, `target_names`, `DESCR`, `frame` (with `as_frame=True`) |
| Six loaders | `load_iris`, `load_diabetes`, `load_digits`, `load_wine`, `load_breast_cancer`, `load_linnerud` |
| DataFrame output | `load_iris(as_frame=True).frame` |
| Array output | `load_iris(return_X_y=True)` |
| Eight generators | `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`, `make_multilabel_classification`, `make_sparse_uncorrelated` |
| OpenML | `fetch_openml(name="mnist_784", version=1)` or `fetch_openml(data_id=2)` |
| Cache locations | seaborn: `sns.get_data_home()`; sklearn: `get_data_home()` → `~/scikit_learn_data` |
| Boston housing | removed from scikit-learn 1.2; use OpenML or a local CSV |
| Add a dataset | acquire → save UTF-8 CSV → verify → document → `load_local(name)` |
"""))

cells.append(md("""
# Further reading

- [scikit-learn dataset loading utilities](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) — full API reference for loaders and fetchers.
- [scikit-learn `Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html) — the dict-like container returned by loaders.
- [seaborn dataset tutorial](https://seaborn.pydata.org/tutorial/introduction.html#seaborn-datasets) — how the example datasets are used in plotting guides.
- [pandas I/O tools](https://pandas.pydata.org/docs/user_guide/io.html) — CSV, Excel, Parquet, SQL and JSON reading.
- [UCI dataset search](https://archive.ics.uci.edu/datasets) — browse datasets by task and size.
- [OpenML search](https://www.openml.org/search?type=data) — thousands of datasets with numeric IDs.
- [Kaggle Datasets API / kagglehub docs](https://github.com/Kaggle/kagglehub) — programmatic downloads.
- [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/) — text, image and audio datasets.
- [Our World in Data grapher API](https://ourworldindata.org/grapher-explorer) — CSV endpoints for global indicators.
- [World Bank API](https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information) — indicator and country endpoints.
- [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/docs/TAP.html) — ADQL query service.
"""))

write_notebook("notebooks/data-science/datasets_catalog.ipynb", cells)
