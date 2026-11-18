"""Builder for the Unit IV part 1 reference notebook (NumPy, SciPy, pandas)."""

from nbhelper import md, code, write_notebook

cells = []

# ----------------------------------------------------------------------
# 1. Title / objectives
# ----------------------------------------------------------------------
cells.append(md(r"""
# Unit IV, Part 1 — NumPy, SciPy and pandas in Depth

This reference notebook covers the three core numerical and tabular libraries of the Python data-science stack. NumPy supplies the `ndarray`, a homogeneous, vectorised, in-memory array that underlies almost every other library. SciPy builds on NumPy with ready-made algorithms for statistics, optimisation, linear algebra, signal processing, integration, interpolation, clustering and image processing. pandas builds on both with the `Series` and `DataFrame`, giving labelled, table-oriented data wrangling. We work through array creation, indexing, broadcasting, ufuncs, aggregation, linear algebra, hypothesis testing, model fitting, group-by aggregation, reshaping, time series, file I/O and memory tuning, using the local CSV files in `../../data`. The plotting, symbolic-math and statistics-modelling libraries are covered in a separate notebook.

**Learning objectives**

- Create NumPy arrays with `array`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace` and the `random` module; explain `dtype`, `shape`, `ndim` and `size`.
- Index and slice arrays; distinguish views from copies; use fancy indexing and boolean masks.
- Reshape, transpose, stack and split arrays; state and apply the broadcasting rules.
- Use ufuncs and vectorisation, aggregation with the `axis` parameter, sorting and structured arrays.
- Generate reproducible random numbers with seeds and `Generator`; solve linear algebra with `np.linalg` (`dot`, `matmul`, `inv`, `det`, `eig`, `solve`, `svd`, `norm`); save and load `.npy` files.
- Navigate SciPy's subpackages and apply `scipy.stats` distributions, descriptive statistics and hypothesis tests (`ttest_ind`, `ttest_1samp`, `chisquare`, `pearsonr`, `spearmanr`, `f_oneway`, `shapiro`, `mannwhitneyu`).
- Optimise and fit with `scipy.optimize` (`minimize`, `curve_fit`, `root`, `linprog`), and use `scipy.linalg`, `scipy.sparse`, `scipy.signal`, `scipy.integrate`, `scipy.interpolate`, `scipy.cluster.hierarchy`, `scipy.spatial.distance` and `scipy.ndimage`.
- Build pandas `Series` and `DataFrame`, inspect them, and select data with `loc`, `iloc`, `at`, `iat`, boolean filters, `isin` and `query`.
- Handle missing data and duplicates; transform columns with `map`, `apply` and `DataFrame.map` (the current name for `applymap`); aggregate with `groupby` (`agg`, `transform`, `filter`, `pivot_table`).
- Combine data with `merge`, `join` and `concat`; reshape with `melt`, `stack`, `unstack` and MultiIndex; sort and rank; bin with `cut`/`qcut` and categorical dtype.
- Use `.str` string methods, datetime conversion with `to_datetime`/`.dt`, `resample` and `rolling`; read and write CSV, Excel, JSON and parquet; optimise memory and style a DataFrame.
"""))

# ----------------------------------------------------------------------
# 2. Official documentation
# ----------------------------------------------------------------------
cells.append(md(r"""
## Official documentation & references

- [NumPy: absolute beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy: broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy reference and API index](https://numpy.org/doc/stable/reference/index.html)
- [SciPy tutorial](https://docs.scipy.org/doc/scipy/tutorial/index.html)
- [SciPy reference](https://docs.scipy.org/doc/scipy/reference/index.html)
- [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [pandas reference](https://pandas.pydata.org/docs/reference/index.html)
"""))

# ----------------------------------------------------------------------
# 3. Setup / imports
# ----------------------------------------------------------------------
cells.append(code(r"""
from pathlib import Path

# Data lives two levels above this notebook.
DATA = Path("../../data")

import time
import tempfile
import warnings
warnings.filterwarnings("ignore")   # keep teaching output tidy

import numpy as np
import pandas as pd
import scipy
from scipy import stats, optimize, linalg, signal, integrate, interpolate, sparse, ndimage
from scipy.optimize import minimize, curve_fit, root, linprog
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster

print("numpy", np.__version__, "| pandas", pd.__version__, "| scipy", scipy.__version__)
print("data dir exists:", DATA.exists())
"""))

# ----------------------------------------------------------------------
# NumPy
# ----------------------------------------------------------------------
cells.append(md(r"""
## 1. NumPy: the `ndarray`

An `ndarray` is a fixed-size, homogeneous block of values in contiguous memory. That single design choice is why NumPy is fast: arithmetic runs in compiled C loops instead of Python loops, and the array carries its own shape and dtype metadata. Everything below works on the same object, so the operations compose freely.
"""))

cells.append(code(r"""
# Creation routines: each returns an ndarray with a given shape.
a = np.array([1, 2, 3, 4])                  # from a Python list
b = np.array([[1, 2, 3], [4, 5, 6]])        # 2-D / nested list
z = np.zeros((2, 3))                        # all zeros
o = np.ones((3, 2), dtype=int)              # all ones as int
f = np.full((2, 2), 7)                      # filled with a constant
e = np.eye(4)                                # 4x4 identity matrix
r = np.arange(0, 20, 2)                     # like range(): start, stop, step
l = np.linspace(0, 1, 11)                   # 11 evenly spaced values incl. both ends

print("array:", a)
print("2-D array:\n", b)
print("zeros:\n", z)
print("ones (int):\n", o)
print("full:\n", f)
print("eye:\n", e)
print("arange:", r)
print("linspace:", np.round(l, 2))
"""))

cells.append(md(r"""
### dtype, shape, ndim and size

Every array has a **dtype** (the type of each element), a **shape** (length per axis), **ndim** (number of axes) and **size** (total element count). `itemsize` gives bytes per element and `nbytes` gives the whole buffer. `astype` casts into a new array with a different dtype.
"""))

cells.append(code(r"""
x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print("dtype:", x.dtype)
print("shape:", x.shape)
print("ndim:", x.ndim)
print("size:", x.size)
print("itemsize (bytes):", x.itemsize)
print("nbytes:", x.nbytes)

xi = x.astype(np.int32)                     # casting creates a new array
print("cast dtype:", xi.dtype, "values:\n", xi)
"""))

cells.append(md(r"""
### Indexing and slicing

Indexing uses zero-based positions and square brackets. For 2-D arrays the first index is the row, the second the column. A slice `start:stop:step` returns a view (see the next section); negative indices count from the end.
"""))

cells.append(code(r"""
v = np.arange(10)
print("v[3]:", v[3], " v[-1]:", v[-1])
print("v[2:6]:", v[2:6])
print("v[::2]:", v[::2])
print("v[::-1]:", v[::-1])                  # reverse

m = np.arange(12).reshape(3, 4)
print("m:\n", m)
print("m[1, 2]:", m[1, 2])                  # row 1, col 2
print("m[0, :]:", m[0, :])                  # first row
print("m[:, -1]:", m[:, -1])               # last column
print("m[0:2, 1:3]:\n", m[0:2, 1:3])
"""))

cells.append(md(r"""
### Views vs copies

Basic slicing returns a **view**: a new array object that shares the original data buffer, so modifying it changes the source. Fancy indexing and `.copy()` return **copies** with their own buffer. `np.shares_memory` tells you which you have. Knowing this prevents subtle bugs and unnecessary memory use.
"""))

cells.append(code(r"""
a = np.arange(10)
view = a[2:6]                                # basic slice -> view
view[0] = 999
print("after view edit, a:", a)

fancy = a[[2, 3, 4]]                         # fancy indexing -> copy
fancy[0] = -1
print("after fancy edit, a (unchanged):", a)

explicit = a[2:6].copy()                     # explicit copy
explicit[0] = -5
print("after copy edit, a (unchanged):", a)
print("shares memory with view:", np.shares_memory(a, view))
print("shares memory with fancy:", np.shares_memory(a, fancy))
"""))

cells.append(md(r"""
### Fancy indexing and boolean masking

Fancy indexing uses an integer array of positions; boolean masking uses a same-shaped array of `True`/`False`. Masks can be combined with `&` (and), `|` (or) and `~` (not) — use parentheses because `&` binds tighter than comparison operators. `np.where` is the vectorised if-then-else.
"""))

cells.append(code(r"""
a = np.arange(10, 60, 10)                    # [10 20 30 40 50]
print("fancy:", a[[0, 2, 4]])
print("repeated fancy:", a[[1, 1, 3]])

mask = a > 25
print("mask:", mask)
print("boolean select:", a[mask])
print("combined:", a[(a > 15) & (a < 45)])
print("negated:", a[~mask])
print("np.where:", np.where(a > 25, a, -1))
"""))

cells.append(md(r"""
### Reshaping, transpose, stacking and splitting

`reshape` reinterprets the same buffer with a new shape (using `-1` lets NumPy infer one axis). `ravel` flattens to 1-D. `.T` / `transpose` permutes axes. `vstack`, `hstack` and `stack` join arrays; `split`, `hsplit` and `vsplit` break them apart.
"""))

cells.append(code(r"""
a = np.arange(12)
print("reshape 3x4:\n", a.reshape(3, 4))
print("reshape 2x-1:\n", a.reshape(2, -1))
print("ravel:", a.reshape(3, 4).ravel())

m = np.arange(6).reshape(2, 3)
print("m:\n", m)
print("m.T:\n", m.T)
print("transpose (1, 0):\n", np.transpose(m, (1, 0)))
"""))

cells.append(code(r"""
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
print("vstack:\n", np.vstack([u, v]))
print("hstack:", np.hstack([u, v]))
print("stack axis=0:\n", np.stack([u, v], axis=0))
print("stack axis=1:\n", np.stack([u, v], axis=1))

m = np.arange(12).reshape(3, 4)
print("hsplit into 2:", np.hsplit(m, 2))
print("vsplit into 3 first part:\n", np.vsplit(m, 3)[0])
print("split axis=1:\n", np.split(m, 2, axis=1)[0])
"""))

cells.append(md(r"""
### Broadcasting rules

When two arrays are combined elementwise, NumPy aligns their shapes from the **right**. Two dimensions are compatible when they are equal or one of them is `1`; the size-1 axis is stretched. The result shape takes the larger size on each axis. Incompatible shapes raise `ValueError`. This is what lets you subtract a column of means from a whole matrix without writing a loop.
"""))

cells.append(code(r"""
col = np.arange(3).reshape(3, 1)            # shape (3, 1)
row = np.arange(4).reshape(1, 4)            # shape (1, 4)
print("(3,1) + (1,4) -> (3,4):\n", col + row)

vector = np.array([10, 20, 30])             # shape (3,)
matrix = np.ones((2, 3))                    # shape (2, 3)
print("(2,3) + (3,):\n", matrix + vector)

try:
    np.ones((2, 3)) + np.ones((4,))          # trailing sizes 3 vs 4 -> error
except ValueError as exc:
    print("expected broadcasting error:", exc)
"""))

cells.append(md(r"""
### ufuncs and vectorisation

A **ufunc** (universal function) is a compiled, elementwise function that supports broadcasting, casting and optional output arrays: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `add`, `multiply`, and many more. Vectorising means expressing work as ufunc calls over whole arrays rather than Python `for` loops. The speed difference is typically one to two orders of magnitude.
"""))

cells.append(code(r"""
x = np.linspace(0, 10, 1_000_000)
y = np.sin(x) + np.cos(x) ** 2                # ufuncs compose elementwise
print("ufunc result head:", np.round(y[:4], 3))

add_one = np.frompyfunc(lambda value: value + 1, 1, 1)   # custom ufunc
print("custom ufunc:", add_one(np.array([1, 2, 3])))

start = time.perf_counter()
_ = np.sqrt(x)
vector_time = time.perf_counter() - start

start = time.perf_counter()
_ = [value ** 0.5 for value in x]
loop_time = time.perf_counter() - start
print(f"vectorised {vector_time:.4f}s vs Python loop {loop_time:.4f}s")
"""))

cells.append(md(r"""
### Aggregation and the `axis` parameter

Reductions collapse an array to a smaller one. `axis=0` reduces **down each column** (one result per column); `axis=1` reduces **across each row**. Without `axis`, the whole array is reduced. `argmin`/`argmax` return the flat position of the extreme value. The `nan*` variants skip missing values.
"""))

cells.append(code(r"""
a = np.arange(1, 13).reshape(3, 4)
print("array:\n", a)
print("sum:", a.sum(), " mean:", round(a.mean(), 3))
print("std:", round(a.std(), 3), " var:", round(a.var(), 3))
print("min:", a.min(), " max:", a.max())
print("sum axis=0 (per column):", a.sum(axis=0))
print("sum axis=1 (per row):", a.sum(axis=1))
print("mean axis=0:", np.round(a.mean(axis=0), 2))
print("argmin:", a.argmin(), " argmax:", a.argmax())
print("argmax axis=0:", a.argmax(axis=0))

b = a.astype(float)
b[0, 0] = np.nan
print("plain mean:", b.mean(), " nanmean:", np.nanmean(b))
"""))

cells.append(md(r"""
### Sorting and searching

`sort` returns sorted values along an axis; `argsort` returns the indices that would sort the array, which is how you reorder other arrays in the same way. `argmin`/`argmax` report positions, and `nonzero` reports indices of truthy elements.
"""))

cells.append(code(r"""
a = np.array([3, 1, 4, 1, 5, 9])
print("sort:", np.sort(a))
print("argsort:", np.argsort(a))
print("values via argsort:", a[np.argsort(a)])
print("argmax:", a.argmax(), " argmin:", a.argmin())
print("nonzero:", np.nonzero(a > 4)[0])

m = np.array([[3, 1, 2], [9, 7, 8]])
print("sort axis=1 (rows):\n", np.sort(m, axis=1))
print("sort axis=0 (cols):\n", np.sort(m, axis=0))
"""))

cells.append(md(r"""
### Structured arrays

A structured array stores records with **named fields** that may have different dtypes, similar to a C struct or a database row. You define a compound dtype as a list of `(field_name, dtype)` pairs and access columns by name. For most tabular work pandas is richer, but structured arrays are useful when you need NumPy speed with heterogeneous fields.
"""))

cells.append(code(r"""
person_dtype = [("name", "U10"), ("age", "i4"), ("score", "f4")]
people = np.array([("Alice", 30, 9.5), ("Bob", 25, 8.0),
                   ("Carol", 41, 9.1)], dtype=person_dtype)
print("array:", people)
print("field names:", people.dtype.names)
print("ages:", people["age"])
print("mean score:", people["score"].mean())
print("first record:", people[0])
"""))

cells.append(md(r"""
### Random numbers: seeds and `Generator`

`np.random.default_rng(seed)` creates a modern `Generator`. It is preferred over the legacy global `np.random.seed(...)` + `np.random.rand()` API because it avoids shared global state and offers better distributions. The same seed on freshly created generators reproduces the same stream. Common methods: `random`, `integers`, `normal`, `uniform`, `choice`, `permutation`.
"""))

cells.append(code(r"""
rng = np.random.default_rng(42)              # modern Generator API
print("uniform [0,1):", np.round(rng.random(5), 3))
print("integers 1..10:", rng.integers(1, 11, size=5))
print("normal(0,1):", np.round(rng.normal(loc=0, scale=1, size=5), 3))
print("choice:", rng.choice(["a", "b", "c"], size=5))

r1 = np.random.default_rng(42)               # same seed -> same stream
r2 = np.random.default_rng(42)
print("reproducible:", np.array_equal(r1.random(5), r2.random(5)))

np.random.seed(0)                            # legacy API still works
print("legacy seed:", np.random.rand(3))
"""))

cells.append(md(r"""
### Linear algebra with `np.linalg`

`np.linalg` implements matrix operations. `dot`/`@`/`matmul` multiply matrices, `solve` solves `Ax = b` (prefer it over computing the inverse), `inv` inverts, `det` gives the determinant, `eig` gives eigenvalues and eigenvectors, `svd` gives the singular value decomposition, and `norm` computes vector/matrix norms.
"""))

cells.append(code(r"""
A = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)
x = np.linalg.solve(A, b)

print("dot:", np.dot(A, A))
print("matmul:", np.matmul(A, A), " (A @ A:", A @ A, ")")
print("det:", np.linalg.det(A))
print("inv:\n", np.linalg.inv(A))
print("solve Ax=b:", x, " check A@x:", A @ x)

eigvals, eigvecs = np.linalg.eig(A)
print("eigenvalues:", eigvals)
print("eigenvectors:\n", eigvecs)

U, s, Vt = np.linalg.svd(A)
print("singular values:", s)
print("norm (Frobenius):", np.linalg.norm(A))
print("norm (L2):", np.linalg.norm(A, 2))
"""))

cells.append(md(r"""
### Saving and loading arrays

`np.save` writes a single array to a `.npy` file; `np.load` reads it back. `np.savez` packs several named arrays into a `.npz` archive, which `np.load` opens like a dictionary. These formats preserve dtype and shape exactly and are much faster than text for large arrays. Here we use a temporary directory so the notebook leaves no files behind.
"""))

cells.append(code(r"""
arr = np.arange(12).reshape(3, 4)
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "arr.npy"
    np.save(p, arr)
    loaded = np.load(p)
    print("npy round-trip equal:", np.array_equal(arr, loaded))

    pz = Path(tmp) / "bundle.npz"
    np.savez(pz, a=arr, b=arr * 2)
    with np.load(pz) as data:
        print("npz keys:", data.files, " b shape:", data["b"].shape)
"""))

cells.append(md(r"""
### Try it yourself #1 — NumPy

1. Create a `6×4` array of random integers from 1 to 100 using a generator seeded with `7`.
2. Replace every value below 40 with `0` using a boolean mask.
3. Compute the mean of each column and the flat index of the overall maximum.
4. Standardise the array — subtract the column mean and divide by the column standard deviation — using broadcasting. Guard against a zero standard deviation.

Write your code in the next cell, then compare it with the worked solution.
"""))

cells.append(md(r"""
### Worked solution #1
"""))

cells.append(code(r"""
rng = np.random.default_rng(7)
values = rng.integers(1, 101, size=(6, 4))
print("original:\n", values)

values[values < 40] = 0
print("masked:\n", values)

column_means = values.mean(axis=0)
max_index = values.argmax()
print("column means:", np.round(column_means, 2))
print("argmax index:", max_index, "value:", values.flat[max_index])

standard_deviations = values.std(axis=0)
safe_std = np.where(standard_deviations == 0, 1, standard_deviations)
standardised = (values - column_means) / safe_std
print("standardised column means:", np.round(standardised.mean(axis=0), 6))
"""))

# ----------------------------------------------------------------------
# SciPy
# ----------------------------------------------------------------------
cells.append(md(r"""
## 2. SciPy: the algorithmic layer on top of NumPy

SciPy is a family of subpackages, each a mature implementation of a family of numerical algorithms. All of them accept and return NumPy arrays, so they compose with the array skills above.

- `scipy.stats` — probability distributions and statistical tests.
- `scipy.optimize` — minimisation, curve fitting, root finding, linear programming.
- `scipy.linalg` — more linear algebra than `np.linalg` (factorisations, matrix functions).
- `scipy.sparse` — memory-efficient matrices for mostly-zero data.
- `scipy.signal` — filtering, convolution, peak finding, spectral analysis.
- `scipy.integrate` — numerical quadrature and ODE solvers.
- `scipy.interpolate` — interpolation and splines.
- `scipy.cluster.hierarchy`, `scipy.spatial.distance` — clustering and distances.
- `scipy.ndimage` — n-dimensional image processing.
"""))

cells.append(code(r"""
import scipy
print("SciPy version:", scipy.__version__)
for module in (stats, optimize, linalg, signal, integrate, interpolate,
               sparse, ndimage, distance, hierarchy):
    doc = (module.__doc__ or "").strip().splitlines()
    first_line = doc[0] if doc else "(no docstring)"
    print(f"{module.__name__:28s} {first_line[:55]}")
"""))

cells.append(md(r"""
### `scipy.stats`: probability distributions

Every distribution object exposes the same methods. For continuous distributions: `pdf(x)` density, `cdf(x)` probability of being at most `x`, `ppf(q)` the inverse CDF (quantile), `rvs(size=..., random_state=...)` random draws, plus `mean`, `var`, `std`, `median` and `stats(moments="mvsk")`. Discrete distributions use `pmf` instead of `pdf`. Pass location with `loc` and scale with `scale`.
"""))

cells.append(code(r"""
normal = stats.norm
x = np.linspace(-4, 4, 7)
print("pdf:", np.round(normal.pdf(x, loc=0, scale=1), 3))
print("cdf:", np.round(normal.cdf(x), 3))
print("ppf(0.975):", normal.ppf(0.975))
print("mean/var/std/median:", normal.mean(), normal.var(), normal.std(), normal.median())
print("moments mvsk:", normal.stats(moments="mvsk"))

samples = normal.rvs(loc=0, scale=1, size=1000, random_state=0)
print("sample mean/std:", round(samples.mean(), 3), round(samples.std(), 3))

print("binomial pmf P(k=3 of 10):", stats.binom.pmf(3, 10, 0.5))
print("binomial cdf P(k<=5 of 10):", stats.binom.cdf(5, 10, 0.5))
"""))

cells.append(md(r"""
### `scipy.stats`: descriptive statistics

Beyond the familiar mean and variance, SciPy offers trimmed and robust summaries: `tmean`, `tvar`, `tstd`, `skew`, `kurtosis`, `sem` (standard error of the mean), `gmean`/`hmean`, and `describe`, which returns a bundle of common statistics in one call. We use the local `iris.csv` for a real example.
"""))

cells.append(code(r"""
iris = pd.read_csv(DATA / "iris.csv")
measure = iris["sepal_length"].to_numpy()

print("n:", len(measure))
print("tmean:", stats.tmean(measure), " median:", np.median(measure))
print("mode:", stats.mode(measure))
print("tvar:", stats.tvar(measure), " tstd:", stats.tstd(measure))
print("skew:", stats.skew(measure), " kurtosis:", stats.kurtosis(measure))
print("min/max:", np.min(measure), np.max(measure))
print("geometric/harmonic mean:", stats.gmean(measure), stats.hmean(measure))
print("25/50/75 percentiles:", np.percentile(measure, [25, 50, 75]))
print("standard error:", stats.sem(measure))
print("describe:", stats.describe(measure))
"""))

cells.append(md(r"""
### Hypothesis tests: t-tests

`ttest_ind` compares the means of two independent groups. By default it assumes equal variances; pass `equal_var=False` for Welch's t-test, which is safer when variances differ. `ttest_1samp` compares a sample mean with a known population mean. Both return a result object with `.statistic` and `.pvalue`.
"""))

cells.append(code(r"""
setosa = iris.loc[iris["species"] == "setosa", "petal_length"]
versicolor = iris.loc[iris["species"] == "versicolor", "petal_length"]

welch = stats.ttest_ind(setosa, versicolor, equal_var=False)
print("Welch t-test:", welch)
print("statistic:", round(welch.statistic, 3), " p-value:", welch.pvalue)

one_sample = stats.ttest_1samp(setosa, popmean=1.5)
print("one-sample t-test:", one_sample)
print("p-value:", one_sample.pvalue)
"""))

cells.append(md(r"""
### Other common hypothesis tests

- `pearsonr` / `spearmanr` — linear / rank correlation with a p-value.
- `chisquare` — one-sample goodness-of-fit; `chi2_contingency` — independence of categorical variables.
- `f_oneway` — one-way ANOVA comparing three or more group means.
- `shapiro` — Shapiro–Wilk normality test.
- `mannwhitneyu` — non-parametric alternative to the two-sample t-test.
"""))

cells.append(code(r"""
# Correlation
r_pearson, p_pearson = stats.pearsonr(iris["sepal_length"], iris["petal_length"])
r_spearman, p_spearman = stats.spearmanr(iris["sepal_length"], iris["petal_length"])
print(f"Pearson r={r_pearson:.3f} (p={p_pearson:.2e})")
print(f"Spearman rho={r_spearman:.3f} (p={p_spearman:.2e})")

# Chi-square goodness of fit vs an expected distribution
observed = np.array([30, 25, 18, 27])
expected = np.full(4, observed.mean())
print("chisquare goodness-of-fit:", stats.chisquare(f_obs=observed, f_exp=expected))

# Chi-square independence on tips.csv
tips = pd.read_csv(DATA / "tips.csv")
contingency = pd.crosstab(tips["sex"], tips["smoker"])
chi2, p_chi, dof, expected_table = stats.chi2_contingency(contingency)
print("contingency table:\n", contingency)
print(f"chi2={chi2:.3f}, p={p_chi:.3f}, dof={dof}")
"""))

cells.append(code(r"""
# One-way ANOVA across three species
groups = [frame["sepal_length"].to_numpy() for _, frame in iris.groupby("species")]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"ANOVA F={f_stat:.3f}, p={p_anova:.2e}")

# Normality and a distribution-free comparison
print("Shapiro-Wilk on setosa:", stats.shapiro(setosa))
virginica = iris.loc[iris["species"] == "virginica", "petal_length"]
print("Mann-Whitney:", stats.mannwhitneyu(setosa, virginica, alternative="two-sided"))
"""))

cells.append(md(r"""
### `scipy.optimize`: minimise, solve roots, fit curves, optimise linearly

`minimize(fun, x0, method=...)` finds a local minimum of a scalar function of several variables. `root(fun, x0)` solves a system of nonlinear equations. `curve_fit(f, xdata, ydata, p0=...)` fits model parameters by least squares. `linprog(c, A_ub, b_ub, bounds, method="highs")` solves a linear program: minimise `c·x` subject to inequality constraints and bounds.
"""))

cells.append(code(r"""
# Minimise the Rosenbrock function (a classic optimisation benchmark)
def rosenbrock(coords):
    x_coord, y_coord = coords
    return (1 - x_coord) ** 2 + 100 * (y_coord - x_coord ** 2) ** 2

result = minimize(rosenbrock, x0=np.array([0.0, 0.0]), method="BFGS")
print("BFGS success:", result.success, " optimum:", np.round(result.x, 4), " fun:", result.fun)

# Solve a nonlinear system: x+y=3 and x^2+y^2=5
def equations(coords):
    x_coord, y_coord = coords
    return [x_coord + y_coord - 3, x_coord ** 2 + y_coord ** 2 - 5]

solution = root(equations, x0=[1.0, 2.0])
print("root:", np.round(solution.x, 4), " residual:", np.round(solution.fun, 8))
"""))

cells.append(code(r"""
# Curve fitting: recover parameters of a decaying exponential
x_data = np.linspace(0, 4, 30)
y_clean = 2.5 * np.exp(-0.7 * x_data) + 0.1
noise = np.random.default_rng(1).normal(0, 0.05, x_data.size)
y_noisy = y_clean + noise

def decay(x_values, amplitude, rate, offset):
    return amplitude * np.exp(-rate * x_values) + offset

popt, pcov = curve_fit(decay, x_data, y_noisy, p0=(2, 1, 0))
print("fitted (amplitude, rate, offset):", np.round(popt, 3))
print("parameter std errors:", np.round(np.sqrt(np.diag(pcov)), 3))

# Linear programming: maximise x+2y  <=>  minimise -x-2y
lp = linprog(c=[-1, -2], A_ub=[[1, 1], [1, -1]], b_ub=[6, 2],
             bounds=[(0, None), (0, None)], method="highs")
print("LP success:", lp.success, " x:", lp.x, " minimum:", lp.fun)
"""))

cells.append(md(r"""
### `scipy.linalg`

`scipy.linalg` offers a richer linear-algebra toolbox than `np.linalg`: `det`, `inv`, `solve`, `eig`/`eigh`, `svd`, `qr`, `cholesky`, `expm` (matrix exponential) and more. It is optimised for LAPACK and offers extra options such as assuming symmetric or positive-definite matrices.
"""))

cells.append(code(r"""
A = np.array([[4, 1, -2], [1, 2, 0], [-2, 0, 3]], dtype=float)
print("det:", linalg.det(A))
print("inv:\n", np.round(linalg.inv(A), 3))
print("solve:", linalg.solve(A, np.array([1, 2, 3], dtype=float)))
print("norm:", linalg.norm(A))
print("eigenvalues:", linalg.eigvals(A))

Q, R = linalg.qr(A)
print("QR: Q.T @ Q ~ I:\n", np.round(Q.T @ Q, 10))
print("matrix exponential (A*0.01):\n", np.round(linalg.expm(A * 0.01), 3))
"""))

cells.append(md(r"""
### `scipy.sparse`: compressed sparse row matrices

When a matrix is mostly zeros, storing only the nonzeros saves a lot of memory. `csr_matrix` (compressed sparse row) is the workhorse for arithmetic and row slicing; `csc_matrix` is better for column slicing. Use `.toarray()` to convert back to dense. `nnz` reports the number of stored nonzeros.
"""))

cells.append(code(r"""
dense = np.array([[0, 0, 3], [4, 0, 0], [0, 5, 0], [0, 0, 6]])
sparse_matrix = csr_matrix(dense)
print("csr representation:\n", sparse_matrix)
print("stored nonzeros:", sparse_matrix.nnz)
print("toarray:\n", sparse_matrix.toarray())
print("row slice 1:3:\n", sparse_matrix[1:3].toarray())
print("matrix-vector product:", (sparse_matrix @ np.array([1, 2, 3])).ravel())
print("dense bytes:", dense.nbytes, " vs sparse data bytes:", sparse_matrix.data.nbytes)
"""))

cells.append(md(r"""
### `scipy.signal`: filtering, convolution and peaks

`signal.convolve` applies a kernel (for example a moving average) to a 1-D signal. `signal.find_peaks` locates local maxima with constraints such as minimum separation and prominence. `signal.butter` designs an IIR filter and `signal.filtfilt` applies it forward and backward so the filtered signal has no phase shift.
"""))

cells.append(code(r"""
time_axis = np.linspace(0, 1, 500, endpoint=False)
signal_values = (np.sin(2 * np.pi * 5 * time_axis)
                 + 0.5 * np.sin(2 * np.pi * 12 * time_axis))

kernel = np.ones(10) / 10
smoothed = signal.convolve(signal_values, kernel, mode="same")
print("smoothed head:", np.round(smoothed[:5], 3))

peaks, properties = signal.find_peaks(signal_values, distance=30, prominence=0.5)
print("peak indices:", peaks)
print("peak prominences:", np.round(properties["prominences"], 3))

b_coeff, a_coeff = signal.butter(4, 0.15)
filtered = signal.filtfilt(b_coeff, a_coeff, signal_values)
print("filtered head:", np.round(filtered[:5], 3))
"""))

cells.append(md(r"""
### `scipy.integrate`: numerical quadrature

`integrate.quad(func, a, b)` computes a definite integral and returns `(value, estimated_error)`. It handles infinite limits with `np.inf`. For ODEs there is `solve_ivp`; for sampled data there are `trapz`/`simpson`.
"""))

cells.append(code(r"""
area, error = integrate.quad(lambda t: np.sin(t), 0, np.pi)
print("integral of sin on [0, pi]:", area, " estimated error:", error)

normal_area, normal_err = integrate.quad(
    lambda t: stats.norm.pdf(t, loc=0, scale=1), -np.inf, np.inf)
print("normal pdf integrates to:", normal_area, " error:", normal_err)

print("integral of x^2 on [0,1]:", integrate.quad(lambda t: t ** 2, 0, 1)[0])
"""))

cells.append(md(r"""
### `scipy.interpolate`

Interpolation estimates values between known data points. NumPy's `np.interp` does simple linear interpolation. SciPy adds `interp1d` for several `kind` values and `CubicSpline` for a smooth piecewise-cubic curve. `PchipInterpolator` avoids overshoot.
"""))

cells.append(code(r"""
x_known = np.linspace(0, 10, 8)
y_known = np.sin(x_known)
x_query = np.linspace(0, 10, 5)

linear_numpy = np.interp(x_query, x_known, y_known)
linear_scipy = interpolate.interp1d(x_known, y_known, kind="linear")(x_query)
cubic = interpolate.CubicSpline(x_known, y_known)(x_query)
pchip = interpolate.PchipInterpolator(x_known, y_known)(x_query)

print("query points:", np.round(x_query, 2))
print("np.interp:", np.round(linear_numpy, 3))
print("interp1d linear:", np.round(linear_scipy, 3))
print("cubic spline:", np.round(cubic, 3))
print("pchip:", np.round(pchip, 3))
"""))

cells.append(md(r"""
### `scipy.spatial.distance` and `scipy.cluster.hierarchy`

`pdist` computes pairwise distances in condensed form; `squareform` turns that into a full matrix. `linkage` builds an agglomerative clustering tree from observations using a method such as `ward` or `average`. `fcluster` cuts the tree into flat clusters by a threshold or a target count. `dendrogram` plots the tree (plotting is covered in the companion notebook).
"""))

cells.append(code(r"""
features = iris[["sepal_length", "petal_length"]].to_numpy()[:20]
condensed = pdist(features, metric="euclidean")
print("first condensed distances:", np.round(condensed[:5], 2))
print("square matrix shape:", squareform(condensed).shape)

tree = linkage(features, method="ward")
print("linkage matrix shape:", tree.shape)
labels = fcluster(tree, t=2, criterion="maxclust")
print("two-cluster labels:", labels)
"""))

cells.append(md(r"""
### `scipy.ndimage`: n-dimensional image processing

`ndimage` generalises image operations to arrays of any dimension. `uniform_filter` is a box blur, `gaussian_filter` a Gaussian blur, `binary_erosion`/`binary_dilation` are morphological operations, and `label` finds connected components. It is commonly used on 2-D images but works on 1-D signals or 3-D volumes too.
"""))

cells.append(code(r"""
image = np.zeros((7, 7))
image[2:5, 2:5] = 1
print("original:\n", image.astype(int))
print("uniform (box) filter:\n", np.round(ndimage.uniform_filter(image, size=3), 2))
print("gaussian filter:\n", np.round(ndimage.gaussian_filter(image, sigma=1), 2))
print("binary erosion:\n", ndimage.binary_erosion(image).astype(int))

cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
labeled, count = ndimage.label(image, structure=cross)
print("connected components:", count)
"""))

cells.append(md(r"""
### Try it yourself #2 — SciPy

Using the local `iris.csv`:

1. Test whether `petal_width` differs between `setosa` and `virginica` with Welch's t-test; report the p-value.
2. Fit a straight line `y = m*x + c` to `sepal_length` (x) versus `petal_length` (y) using `curve_fit`.
3. Compute the 95th percentile of a normal distribution fitted with the sample mean and standard deviation of `sepal_width`.

Compare your code with the worked solution.
"""))

cells.append(md(r"""
### Worked solution #2
"""))

cells.append(code(r"""
setosa_width = iris.loc[iris["species"] == "setosa", "petal_width"]
virginica_width = iris.loc[iris["species"] == "virginica", "petal_width"]
welch_width = stats.ttest_ind(setosa_width, virginica_width, equal_var=False)
print("Welch t-test on petal_width, p-value:", welch_width.pvalue)

x_values = iris["sepal_length"].to_numpy()
y_values = iris["petal_length"].to_numpy()
def line_model(x_values, slope, intercept):
    return slope * x_values + intercept

parameters, covariance = curve_fit(line_model, x_values, y_values, p0=(1, 0))
print("fitted slope/intercept:", np.round(parameters, 3))

mean_sw, std_sw = iris["sepal_width"].mean(), iris["sepal_width"].std()
percentile_95 = stats.norm.ppf(0.95, loc=mean_sw, scale=std_sw)
print(f"95th percentile of sepal_width ~ N({mean_sw:.2f}, {std_sw:.2f}^2): {percentile_95:.3f}")
"""))

# ----------------------------------------------------------------------
# pandas
# ----------------------------------------------------------------------
cells.append(md(r"""
## 3. pandas: labelled, tabular data

pandas adds row and column **labels** to NumPy's fast storage. A `Series` is a labelled 1-D vector; a `DataFrame` is a labelled 2-D table of Series that share an index. Labels make selection, alignment and aggregation readable, and give you missing-value handling, grouping, reshaping and time series support for free.
"""))

cells.append(code(r"""
series = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"], name="values")
print(series)
print("values:", series.values, " index:", series.index.tolist())
print("series['b']:", series["b"], " series.iloc[2]:", series.iloc[2])
print("from dict:", pd.Series({"x": 1, "y": 2}))
"""))

cells.append(code(r"""
frame = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": ["x", "y", "z"]})
print("hand-built DataFrame:\n", frame)

iris = pd.read_csv(DATA / "iris.csv")
print("iris from local CSV:\n", iris.head())
"""))

cells.append(md(r"""
### Inspecting a DataFrame

`head`/`tail` show the first/last rows, `shape` gives `(rows, columns)`, `columns` lists names, `dtypes` shows each column's type, `info()` gives a compact summary including non-null counts and memory, and `describe()` computes numeric summaries. `include="all"` also describes non-numeric columns.
"""))

cells.append(code(r"""
print("shape:", iris.shape)
print("columns:", iris.columns.tolist())
print("head(3):\n", iris.head(3))
print("tail(2):\n", iris.tail(2))
print("dtypes:\n", iris.dtypes)
print("info:"); iris.info()
print("describe():\n", iris.describe())
print("describe(include='all'):\n", iris.describe(include="all"))
"""))

cells.append(md(r"""
### Index and dtypes

The `Index` is the label axis; it can be a simple `RangeIndex`, a string index, a `DatetimeIndex` or a `MultiIndex`. `set_index` promotes a column to the index, `reset_index` moves it back, and `reindex` conforms data to a new label set. Each column has its own dtype; `select_dtypes` picks columns by type and `astype` converts them.
"""))

cells.append(code(r"""
simple = pd.Series([1, 2, 3], index=["r2", "r1", "r3"])
print("index:", simple.index, " type:", type(simple.index).__name__)
print("reindexed:\n", simple.reindex(["r1", "r2", "r3", "r4"]))
print("set_index(head):\n", iris.set_index("species").head(3))
print("reset_index(head):\n", iris.set_index("species").reset_index().head(2))

typed = pd.DataFrame({
    "i": pd.Series([1, 2], dtype="int8"),
    "f": pd.Series([1.5, 2.5], dtype="float32"),
    "b": pd.Series([True, False]),
    "s": pd.Series(["a", "b"], dtype="string"),
})
print("explicit dtypes:\n", typed.dtypes)
print("select_dtypes number:\n", typed.select_dtypes(include="number").dtypes)
"""))

cells.append(md(r"""
### Selection: `loc`, `iloc`, `at`, `iat`

`loc` selects by **label**, `iloc` by **integer position**. Both accept row and column arguments separated by a comma. `at` and `iat` are faster scalar-only versions for a single label or position. Use `loc` for readable, label-based code and `iloc` for positional work.
"""))

cells.append(code(r"""
by_species = iris.set_index("species")
print("loc row (head):\n", by_species.loc["setosa"].head(2))
print("loc rows + columns:\n",
      by_species.loc[["setosa", "versicolor"], ["sepal_length", "petal_length"]].head())
print("iloc [0:3, 0:2]:\n", iris.iloc[0:3, 0:2])
print("column selection:\n", iris[["sepal_length", "species"]].head(3))

scalar_frame = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["p", "q"])
print("at (label -> scalar):", scalar_frame.at["p", "A"])
print("iat (position -> scalar):", scalar_frame.iat[1, 1])
"""))

cells.append(md(r"""
### Boolean filtering, `isin` and `query`

A boolean Series filters rows when placed inside `[]` or `.loc[...]`. Combine conditions with `&`, `|` and `~` in parentheses. `isin` tests membership in a list; `between` gives an inclusive numeric range. `query` expresses the same filter as a readable string and is handy for complex conditions.
"""))

cells.append(code(r"""
print("filtered rows:\n",
      iris[(iris["sepal_length"] > 7) & (iris["species"] == "virginica")].head())
print("isin:\n", iris[iris["species"].isin(["setosa", "virginica"])].head(3))
print("between:\n", iris[iris["sepal_length"].between(5, 6)].head(3))
print("query:\n", iris.query("sepal_length > 7.0 and petal_width < 2.0").head())
print("loc + condition:\n",
      iris.loc[iris["petal_length"] > 6, ["species", "petal_length"]].head())
"""))

cells.append(md(r"""
### Assigning columns and dropping data

Assign a new column with `df["new"] = ...` or functionally with `assign`, which returns a new frame. `np.where` and `np.select` build conditional columns. `drop(columns=..., index=...)` removes columns or rows, and `rename(columns=...)` renames labels.
"""))

cells.append(code(r"""
working = iris.copy()
working["sepal_area"] = working["sepal_length"] * working["sepal_width"]
working["is_setosa"] = np.where(working["species"] == "setosa", "yes", "no")
working = working.drop(columns=["is_setosa"])
working = working.drop(index=[0, 1])
print("new column:\n", working[["sepal_length", "sepal_area"]].head())
print("assign:\n", iris.assign(ratio=iris["petal_length"] / iris["sepal_length"]).head(3))
print("rename:\n", iris.rename(columns={"sepal_length": "SL"}).head(2))
"""))

cells.append(md(r"""
### Missing data: `isna`, `notna`, `dropna`, `fillna`, `interpolate`

`isna`/`notna` produce boolean masks, `dropna` removes incomplete rows, and `fillna` replaces missing values with a constant or a computed value. `ffill`/`bfill` propagate the previous/next value. `interpolate` estimates missing numbers by interpolation. The `subset` and `how` parameters give fine control over which rows are removed.
"""))

cells.append(code(r"""
incomplete = pd.DataFrame({
    "a": [1, np.nan, 3, np.nan, 5],
    "b": [10, 20, np.nan, 40, 50],
    "c": ["x", None, "z", "w", "v"],
})
print("original:\n", incomplete)
print("isna:\n", incomplete.isna())
print("notna counts:\n", incomplete.notna().sum())
print("dropna:\n", incomplete.dropna())
print("dropna(how='all'):\n", incomplete.dropna(how="all"))
print("dropna(subset=['a']):\n", incomplete.dropna(subset=["a"]))
print("fillna(0):\n", incomplete.fillna(0))
print("ffill:\n", incomplete.ffill())
print("interpolate numeric:\n", incomplete[["a", "b"]].interpolate())
"""))

cells.append(md(r"""
### Duplicates

`duplicated()` marks repeated rows as `True` (keeping the first occurrence by default); `drop_duplicates()` removes them. Use `subset` to define identity by certain columns and `keep="first"|"last"|False` to choose which copy survives.
"""))

cells.append(code(r"""
duplicated_frame = pd.DataFrame({
    "key": ["a", "a", "b", "b", "c"],
    "value": [1, 1, 2, 3, 4],
})
print("duplicated mask:\n", duplicated_frame.duplicated())
print("drop_duplicates:\n", duplicated_frame.drop_duplicates())
print("by key, keep last:\n", duplicated_frame.drop_duplicates(subset="key", keep="last"))
"""))

cells.append(md(r"""
### Transforming values: `map`, `apply`, `DataFrame.map`

`Series.map` maps each value through a dict or function. `DataFrame.apply` applies a function along an axis (`axis=0` per column, `axis=1` per row). `DataFrame.map` applies a function elementwise; it was called `applymap` before pandas 2.1 and is the correct current name. These are the pandas equivalents of NumPy's ufunc idea but work with labels.
"""))

cells.append(code(r"""
numbers = pd.Series([1, 2, 3])
print("map with dict:\n", numbers.map({1: "one", 2: "two", 3: "three"}))
print("map with function:\n", numbers.map(lambda value: value ** 2))

table = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
print("apply down columns:\n", table.apply(np.sum))
print("apply across rows:\n", table.apply(np.sum, axis=1))
print("DataFrame.map elementwise:\n", table.map(lambda value: value * 2))
"""))

cells.append(md(r"""
### `groupby`: split, apply, combine

`groupby(key)` splits rows into groups. `.agg` reduces each group to a summary (use named aggregation for readable output), `.transform` returns a result aligned to every original row — ideal for adding group statistics as columns — and `.filter` keeps or drops whole groups based on a predicate. `pivot_table` builds a spreadsheet-style summary with `index`, `columns`, `values` and `aggfunc`.
"""))

cells.append(code(r"""
tips = pd.read_csv(DATA / "tips.csv")
by_day = tips.groupby("day")
print("group sizes:\n", by_day.size())
print("named aggregation:\n",
      by_day.agg(mean_tip=("tip", "mean"), max_bill=("total_bill", "max"), n=("tip", "size")))
print("multiple stats per column:\n", by_day["tip"].agg(["mean", "std", "count"]))
"""))

cells.append(code(r"""
# transform returns one value per original row, aligned by index
tips["tip_z"] = tips.groupby("day")["tip"].transform(
    lambda group: (group - group.mean()) / group.std())
print("z-scored tips:\n", tips[["day", "tip", "tip_z"]].head())

# filter keeps entire groups that satisfy a condition
generous_days = tips.groupby("day").filter(lambda group: group["tip"].mean() > 3)
print("days with mean tip > 3:", generous_days["day"].unique())

# pivot_table: mean total_bill by day and time
print("pivot table:\n",
      tips.pivot_table(values="total_bill", index="day", columns="time", aggfunc="mean"))
"""))

cells.append(md(r"""
### Combining data: `merge`, `join`, `concat`

`merge` joins two DataFrames on a key column with SQL-like `how` modes (`inner`, `left`, `right`, `outer`). `join` is a convenience for joining on the index. `concat` stacks frames vertically (`axis=0`) or horizontally (`axis=1`); `ignore_index=True` builds a fresh index for vertical concatenation.
"""))

cells.append(code(r"""
left = pd.DataFrame({"key": ["a", "b", "c"], "L": [1, 2, 3]})
right = pd.DataFrame({"key": ["b", "c", "d"], "R": [10, 20, 30]})
print("merge inner:\n", left.merge(right, on="key", how="inner"))
print("merge left:\n", left.merge(right, on="key", how="left"))
print("merge outer:\n", left.merge(right, on="key", how="outer"))
print("concat rows:\n", pd.concat([left.head(2), left.tail(2)], ignore_index=True))
print("concat columns:\n", pd.concat([left, right], axis=1))

indexed_a = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
indexed_b = pd.DataFrame({"B": [3, 4]}, index=["y", "z"])
print("join on index:\n", indexed_a.join(indexed_b, how="outer"))
"""))

cells.append(md(r"""
### Reshaping: `melt`, `pivot`, `stack`, `unstack`

`melt` converts wide data (many value columns) into long/tidy data (a `variable` and a `value` column). `pivot` is its inverse. With a MultiIndex, `stack` moves a column level into the index and `unstack` moves an index level out into columns.
"""))

cells.append(code(r"""
wide = pd.DataFrame({"id": [1, 2], "Jan": [100, 200], "Feb": [150, 250]})
long_form = wide.melt(id_vars="id", value_vars=["Jan", "Feb"],
                      var_name="month", value_name="sales")
print("wide:\n", wide)
print("melt to long:\n", long_form)
print("pivot back:\n", long_form.pivot(index="id", columns="month", values="sales"))

stacked = long_form.set_index(["id", "month"])
print("stacked series:\n", stacked["sales"])
print("unstacked:\n", stacked["sales"].unstack())
"""))

cells.append(md(r"""
### MultiIndex (hierarchical index)

A `MultiIndex` has two or more levels per row. Build one with `from_arrays`, `from_tuples` or `from_product`. `xs` extracts a cross-section at one level, `groupby(level=...)` aggregates by level, and `swaplevel` + `sort_index` reorders the hierarchy. MultiIndex is the foundation of grouped reshaping.
"""))

cells.append(code(r"""
levels = [["A", "A", "B", "B"], [1, 2, 1, 2]]
multi_index = pd.MultiIndex.from_arrays(levels, names=["group", "number"])
multi_frame = pd.DataFrame({"value": [10, 20, 30, 40]}, index=multi_index)
print(multi_frame)
print("cross-section group=A:\n", multi_frame.xs("A", level="group"))
print("mean by level 0:\n", multi_frame.groupby(level=0).mean())
print("swapped and sorted:\n", multi_frame.swaplevel().sort_index())
"""))

cells.append(md(r"""
### Sorting, `rank` and `value_counts`

`sort_values` sorts by column values, `sort_index` by labels. `rank` assigns ranks with a chosen tie method (`average`, `min`, `dense`, ...). `value_counts` tabulates category frequencies; with `normalize=True` it returns proportions. `nlargest`/`nsmallest` return the extreme rows efficiently.
"""))

cells.append(code(r"""
print("sort_values:\n", iris.sort_values("petal_length", ascending=False).head(3))
print("sort_index:\n", iris.set_index("species").sort_index().head(3))
print("dense rank:\n", iris["petal_length"].rank(method="dense").head())
print("value_counts:\n", iris["species"].value_counts())
print("value_counts normalized:\n", iris["species"].value_counts(normalize=True))
print("nlargest:\n", iris.nlargest(3, "sepal_length")[["species", "sepal_length"]])
"""))

cells.append(md(r"""
### Binning and categorical dtype

`cut` bins a numeric variable into fixed edges (equal-width or custom), `qcut` into quantile-based bins (roughly equal counts). Both return a `Categorical`. A `pd.Categorical` stores integer codes plus a small `categories` list, which saves memory for repeated strings and lets you declare an `ordered` category with a meaningful sequence.
"""))

cells.append(code(r"""
edges = [0, 4, 5, 6, 8]
bin_names = ["tiny", "small", "medium", "large"]
iris["size_bin"] = pd.cut(iris["sepal_length"], bins=edges, labels=bin_names)
print("cut counts:\n", iris["size_bin"].value_counts().sort_index())
iris["size_quartile"] = pd.qcut(iris["petal_length"], q=4)
print("qcut counts:\n", iris["size_quartile"].value_counts().sort_index())

category = pd.Categorical(["a", "b", "a", "c"], categories=["a", "b"], ordered=True)
print("categorical:", category)
print("codes:", category.codes, " categories:", list(category.categories))
"""))

cells.append(md(r"""
### String methods with `.str`

Pandas vectorises common string operations through the `.str` accessor: `lower`, `upper`, `strip`, `contains`, `startswith`, `split`, `extract`, `replace`, `len`, `find`, `slice` and more. Many accept `na=False` or `regex=False` for predictable behaviour. These work only on string columns.
"""))

cells.append(code(r"""
labels = pd.Series(["  Apple ", "banana", "CHERRY", None, "date"])
print("lower:", labels.str.lower().tolist())
print("strip:", labels.str.strip().tolist())
print("contains 'a':", labels.str.contains("a", case=False, na=False).tolist())
print("startswith 'C':", labels.str.startswith("C", na=False).tolist())
print("first letter:", labels.str.extract(r"(^[A-Za-z])")[0].tolist())
print("split on 'a':", labels.str.split("a").tolist())
print("lengths:", labels.str.len().tolist())
print("replace:", labels.str.replace("a", "@", regex=False).tolist())

diamonds = pd.read_csv(DATA / "diamonds.csv")
print("diamond cut labels:", diamonds["cut"].str.upper().unique()[:5])
"""))

cells.append(md(r"""
### Datetime: `to_datetime`, `.dt`, `resample`, `rolling`

`to_datetime` parses strings into a `DatetimeIndex` or datetime column. The `.dt` accessor extracts `year`, `month`, `day`, `dayofweek`, `month_name()` and more. `resample("MS")` groups time-indexed data by a calendar period. `rolling(window=n)` computes moving-window statistics such as a moving average.
"""))

cells.append(code(r"""
planets = pd.read_csv(DATA / "planets.csv")
print("planets columns:", planets.columns.tolist())

dates = pd.to_datetime(pd.Series(["2021-01-01", "2021-01-03", "2021-02-01", "2021-03-15"]))
print("year:", dates.dt.year.tolist(), " month name:", dates.dt.month_name().tolist())
print("day of week:", dates.dt.dayofweek.tolist())

time_series = pd.Series(
    [10, 20, 30, 40],
    index=pd.to_datetime(["2021-01-01", "2021-01-15", "2021-02-01", "2021-03-01"]),
)
print("resample month-start sum:\n", time_series.resample("MS").sum())
print("rolling 2-period mean:\n", time_series.rolling(window=2).mean())
print("planet discovery year summary:\n", planets["year"].describe())
"""))

cells.append(md(r"""
### Reading and writing files

`read_csv`/`to_csv` handle delimited text, `read_excel`/`to_excel` work with spreadsheets (openpyxl), `read_json`/`to_json` with JSON, and `read_parquet`/`to_parquet` with the columnar parquet format (pyarrow). Parquet preserves dtypes and compresses well; CSV is universal but loses type information. Use `index=False` when you do not want the index written. Here we round-trip files in a temporary directory.
"""))

cells.append(code(r"""
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    iris.to_csv(folder / "iris_copy.csv", index=False)
    tips.to_json(folder / "tips.json", orient="records")
    iris.to_excel(folder / "iris.xlsx", index=False)

    # Arrow cannot store pandas categorical/interval extension dtypes directly.
    parquet_iris = iris.copy()
    for column in parquet_iris.columns:
        if isinstance(parquet_iris[column].dtype, (pd.CategoricalDtype, pd.IntervalDtype)):
            parquet_iris[column] = parquet_iris[column].astype(str)
    parquet_iris.to_parquet(folder / "iris.parquet", index=False)

    print("CSV shape:", pd.read_csv(folder / "iris_copy.csv").shape)
    print("JSON shape:", pd.read_json(folder / "tips.json", orient="records").shape)
    print("Excel shape:", pd.read_excel(folder / "iris.xlsx").shape)
    print("Parquet shape:", pd.read_parquet(folder / "iris.parquet").shape)
    print("parquet dtypes preserved:\n", pd.read_parquet(folder / "iris.parquet").dtypes)
"""))

cells.append(md(r"""
### Memory optimisation and styling

`memory_usage(deep=True)` reports real bytes including strings. Converting repeated strings to `category` and downcasting numeric columns with `to_numeric(..., downcast=...)` can cut memory sharply. A `Styler` (from `.style`) builds HTML for display in Jupyter with methods such as `highlight_max`, `background_gradient`, `bar` and `format`; call `.to_html()` to get the HTML string.
"""))

cells.append(code(r"""
print("diamonds memory (MB):", round(diamonds.memory_usage(deep=True).sum() / 1e6, 2))
compact = diamonds.copy()
for column in compact.columns:
    if pd.api.types.is_string_dtype(compact[column]):
        compact[column] = compact[column].astype("category")
for column in compact.select_dtypes(include=["int64"]).columns:
    compact[column] = pd.to_numeric(compact[column], downcast="integer")
for column in compact.select_dtypes(include=["float64"]).columns:
    compact[column] = pd.to_numeric(compact[column], downcast="float")
print("after optimisation (MB):", round(compact.memory_usage(deep=True).sum() / 1e6, 2))
print("new dtypes:\n", compact.dtypes)

styler = tips.head().style.highlight_max(axis=0, color="lightgreen").format({"tip": "{:.2f}"})
print("Styler type:", type(styler).__name__, "| HTML chars:", len(styler.to_html()))
"""))

cells.append(md(r"""
### Try it yourself #3 — pandas

Using the local `tips.csv`:

1. Add a `tip_pct` column equal to `tip / total_bill * 100`.
2. Use `groupby` + `agg` to get the mean `tip_pct` and the row count per `day`, sorted by mean percentage descending.
3. Convert `day` to an **ordered** categorical with the order `Thur, Fri, Sat, Sun`.
4. Use `.str` on `sex` to create an uppercase version.

Compare your code with the worked solution.
"""))

cells.append(md(r"""
### Worked solution #3
"""))

cells.append(code(r"""
work = pd.read_csv(DATA / "tips.csv")
work["tip_pct"] = work["tip"] / work["total_bill"] * 100

summary = (work.groupby("day")["tip_pct"]
              .agg(mean_pct="mean", n="count")
              .sort_values("mean_pct", ascending=False))
print("mean tip percentage by day:\n", summary)

day_order = ["Thur", "Fri", "Sat", "Sun"]
work["day_ordered"] = pd.Categorical(work["day"], categories=day_order, ordered=True)
print("ordered categories:", work["day_ordered"].cat.categories.tolist())

work["SEX_UPPER"] = work["sex"].str.upper()
print(work[["sex", "SEX_UPPER"]].head())
"""))

# ----------------------------------------------------------------------
# Wrap-up
# ----------------------------------------------------------------------
cells.append(md(r"""
## Cheat sheet / key takeaways

**NumPy**

| Task | Call |
|---|---|
| Create | `np.array`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `default_rng(...).random` |
| Describe | `arr.dtype`, `.shape`, `.ndim`, `.size`, `.itemsize`, `.nbytes` |
| Select | `arr[i, j]`, slices (views), `arr[[i, j]]` (copy), `arr[mask]`, `np.where` |
| Reshape | `.reshape`, `.ravel`, `.T`, `np.vstack`, `np.hstack`, `np.stack`, `np.split` |
| Combine | broadcasting: right-align shapes; sizes equal or 1 |
| Compute | `sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax` + `axis=0/1` |
| Sort | `np.sort`, `np.argsort`, `np.nonzero` |
| Linalg | `np.dot`, `@`, `np.linalg.solve`, `inv`, `det`, `eig`, `svd`, `norm` |
| Persist | `np.save`, `np.load`, `np.savez` |

**SciPy**

| Subpackage | Representative calls |
|---|---|
| `stats` | `norm.pdf/cdf/ppf/rvs`, `ttest_ind`, `ttest_1samp`, `chisquare`, `pearsonr`, `spearmanr`, `f_oneway`, `shapiro`, `mannwhitneyu` |
| `optimize` | `minimize`, `curve_fit`, `root`, `linprog` |
| `linalg` / `sparse` | `expm`, `qr`, `cholesky` / `csr_matrix`, `.toarray()` |
| `signal` / `integrate` / `interpolate` | `convolve`, `find_peaks`, `filtfilt` / `quad` / `interp1d`, `CubicSpline` |
| `spatial` / `cluster` / `ndimage` | `pdist`, `squareform` / `linkage`, `fcluster` / `gaussian_filter`, `label` |

**pandas**

| Task | Call |
|---|---|
| Inspect | `head`, `tail`, `shape`, `dtypes`, `info`, `describe` |
| Select | `loc` (label), `iloc` (position), `at`, `iat`, boolean mask, `isin`, `query` |
| Clean | `isna`, `notna`, `dropna`, `fillna`, `ffill`, `interpolate`, `drop_duplicates` |
| Transform | `map`, `apply`, `DataFrame.map` (ex-`applymap`), `np.where` |
| Aggregate | `groupby(...).agg/transform/filter`, `pivot_table` |
| Combine | `merge`, `join`, `concat` |
| Reshape | `melt`, `pivot`, `stack`, `unstack`, `MultiIndex`, `xs` |
| Bin / text / time | `cut`, `qcut`, `Categorical`, `.str`, `to_datetime`, `.dt`, `resample`, `rolling` |
| I/O / memory / style | `read_csv`/`to_csv`, parquet, `memory_usage`, `.style` |

Key habits: seed generators for reproducibility, prefer `np.linalg.solve` over explicit inverses, distinguish views from copies, use `axis` deliberately, prefer vectorisation over Python loops, and use labels (`loc`) rather than positional indexing when columns may move.
"""))

cells.append(md(r"""
## Further reading

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy ufuncs](https://numpy.org/doc/stable/user/basics.ufuncs.html)
- [NumPy linear algebra routines](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy statistics reference](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [SciPy optimize reference](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [SciPy signal processing reference](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [pandas groupby guide](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [pandas time series guide](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [pandas merging and reshaping guide](https://pandas.pydata.org/docs/user_guide/merging.html)
- [pandas 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
"""))

write_notebook(
    "notebooks/data-science/unit4_numpy_scipy_pandas.ipynb",
    cells,
)
