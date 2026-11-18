"""Build the Unit IV part 2 reference notebook (Matplotlib, Seaborn, Plotly/Bokeh, SymPy, Statsmodels)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nbhelper import md, code, write_notebook

cells = []

# --------------------------------------------------------------------------- 1
cells.append(md(r"""
# Unit IV — Part 2: Matplotlib, Seaborn, Plotly/Bokeh, SymPy and Statsmodels

This is a long-lived reference notebook for the plotting and mathematical-computing half of the Python-libraries syllabus. It moves from the low-level Matplotlib object model to the high-level, data-oriented Seaborn API, then shows the two browser-based interactive libraries (Plotly and Bokeh) with one worked example each. The second half covers SymPy for exact symbolic algebra and calculus, and statsmodels for statistical modelling with an emphasis on reading and interpreting model output. Every code cell is self-contained, uses only the local CSV files or a reproducible synthetic dataset, and needs no internet connection.

**Learning objectives**

- Explain the Matplotlib figure/axes anatomy and compare the stateful `pyplot` API with the explicit object-oriented API.
- Build multi-panel figures with `subplots`, shared axes, layout control, and both `add_axes` and `add_subplot`.
- Produce line, bar, scatter, histogram, pie, boxplot, errorbar, contour, `imshow`, and basic 3D plots; customise titles, labels, legends, grids, ticks, figure size and dpi.
- Apply styles and `rcParams`, add annotations and reference lines, save figures to a buffer or file, and describe how `FuncAnimation` works.
- Use Seaborn's `set_theme`, `set_context`, `set_palette` and colour palettes; explain the data-oriented API.
- Create relational plots (`scatterplot`, `relplot`), categorical plots (`stripplot`, `swarmplot`, `boxplot`, `violinplot`, `barplot`, `pointplot`, `catplot`), distribution plots (`histplot`, `kdeplot`, `ecdfplot`, `displot`, `jointplot`, `pairplot`), correlation heatmaps, `FacetGrid` small multiples, and regression plots (`regplot`, `lmplot`).
- Write one interactive Plotly Express figure and one Bokeh figure with a hover tool, and explain how both render inside Jupyter.
- Use SymPy for symbols, expressions, exact versus floating-point arithmetic, `simplify`/`expand`/`factor`, `solve`/`solveset`, differentiation, integration, limits, series, matrices, substitution, `lambdify` and plotting.
- Fit and interpret an OLS model with the formula and array APIs, read a statsmodels summary, run proportion tests and outlier diagnostics, fit a synthetic ARIMA time series, fit a logistic regression, run ANOVA, and contrast statsmodels with scikit-learn.
"""))

# --------------------------------------------------------------------------- 2
cells.append(md(r"""
## Official documentation & references

- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html) — the official, progressive tutorial series.
- [Matplotlib API reference](https://matplotlib.org/stable/api/index.html) — every class, function and keyword argument.
- [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html) — the official data-visualisation tutorial.
- [Seaborn API reference](https://seaborn.pydata.org/api.html) — full function signatures and parameter details.
- [Plotly Python (Express) documentation](https://plotly.com/python/) — figure constructors, layouts and renderers.
- [Bokeh user guide](https://docs.bokeh.org/en/latest/docs/user_guide.html) — models, plotting interface and interaction tools.
- [SymPy tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html) — official introduction to symbolic Python.
- [SymPy modules documentation](https://docs.sympy.org/latest/modules/index.html) — solve, calculus, matrices, plotting and more.
- [Statsmodels user guide](https://www.statsmodels.org/stable/user-guide.html) — modelling overview and worked examples.
- [Statsmodels API reference](https://www.statsmodels.org/stable/api.html) — formulas, diagnostics, time series and tests.
"""))

# --------------------------------------------------------------------------- 3
cells.append(code(r"""
from pathlib import Path
DATA = Path("../../data")

import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

# Local course data: no download and no network access is performed.
iris = pd.read_csv(DATA / "iris.csv")
tips = pd.read_csv(DATA / "tips.csv")
diamonds = pd.read_csv(DATA / "diamonds.csv")

# Reproducible synthetic data: a noisy straight line, used where a CSV is not appropriate.
rng = np.random.default_rng(42)
synthetic = pd.DataFrame({
    "x": np.linspace(0, 10, 100),
    "noise": rng.normal(0, 0.8, 100),
})
synthetic["y"] = 2.5 * synthetic["x"] - 3 + synthetic["noise"]

print("shapes:", iris.shape, tips.shape, diamonds.shape, synthetic.shape)
print("iris columns:", list(iris.columns))
print("tips columns:", list(tips.columns))
print("diamonds columns:", list(diamonds.columns))
"""))

# --------------------------------------------------------------------------- 4
cells.append(md(r"""
## 1. Matplotlib — figure anatomy and the two APIs

A **Figure** is the whole canvas. Each **Axes** is one plotting region inside it, with its own **Axis** objects (the x scale and the y scale), ticks, labels and artists. A figure may hold many axes, and an axes normally holds one or more data artists such as lines or patches. Keeping this hierarchy straight is the key to writing plotting code that scales.

There are two equally valid styles:

- **Stateful `pyplot`**: `plt.plot(...)`, `plt.title(...)`, `plt.xlabel(...)`. Functions act on a "current figure" and "current axes" held by `pyplot`. Short and convenient for exploration.
- **Object-oriented (OO)**: `fig, ax = plt.subplots(...)`, then `ax.plot(...)`, `ax.set_title(...)`, `ax.set_xlabel(...)`. You hold explicit references. This is clearer, safer with multiple axes, and the recommended style for reusable code.

Both styles produce the same image; the OO form simply makes the target of every call explicit.
"""))

# --------------------------------------------------------------------------- 5
cells.append(code(r"""
# Figure / Axes / Axis anatomy.
fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
ax.plot(iris["petal_length"], iris["petal_width"], "o", markersize=4)
ax.set_title("Figure / Axes anatomy")
fig.canvas.draw()  # apply layout before reading the stored position

print("figure type:", type(fig).__name__)
print("number of axes in figure:", len(fig.axes))
print("axes position (left, bottom, width, height):",
      tuple(round(v, 3) for v in ax.get_position().bounds))
print("x-axis object:", type(ax.xaxis).__name__)
print("lines on this axes:", len(ax.lines))
plt.show()

# Stateful pyplot style: calls target the implicit current axes.
plt.figure(figsize=(5, 3))
plt.plot(synthetic["x"], synthetic["y"])
plt.xlabel("x")
plt.title("pyplot / stateful API")
plt.show()

# Object-oriented style: every call names its target axes explicitly.
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(synthetic["x"], synthetic["y"])
ax2.set_xlabel("x")
ax2.set_title("OO / explicit API")
plt.show()
"""))

# --------------------------------------------------------------------------- 6
cells.append(md(r"""
## 2. Subplots, shared axes and layout

`plt.subplots(nrows, ncols)` returns a figure and an array of axes. The useful keywords are:

| Keyword | Meaning |
|---|---|
| `figsize=(w, h)`, `dpi=n` | canvas size in inches and pixels per inch |
| `sharex`, `sharey` | link scales (and ticks) across the grid; `True`, `"row"` or `"col"` |
| `squeeze=False` | always return a 2-D axes array, even for a single row/column |
| `constrained_layout=True` | automatically reserve space for labels, titles and legends |
| `subplot_kw={...}` | forwarded to each axes, e.g. `{"projection": "3d"}` |

For manual control use `fig.add_axes([left, bottom, width, height])` with fractions of the figure, or `fig.add_subplot(nrows, ncols, index)`. Call `fig.tight_layout()` (or set `constrained_layout=True`) to stop labels overlapping. Shared axes are ideal for small multiples: zooming or panning one panel updates the linked panels.

Twin axes are a different idea: `ax.twinx()` overlays a second y scale on the *same* plotting region, useful when two series have different units.
"""))

# --------------------------------------------------------------------------- 7
cells.append(code(r"""
# A 2x2 grid with linked scales and automatic layout.
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True,
                         constrained_layout=True)
axes[0, 0].scatter(iris["sepal_length"], iris["sepal_width"], s=12)
axes[0, 1].scatter(iris["petal_length"], iris["petal_width"], s=12)
axes[1, 0].hist(tips["total_bill"], bins=15)
axes[1, 1].hist(tips["tip"], bins=15)
for axis in axes.ravel():
    axis.grid(alpha=0.3)
fig.suptitle("2x2 grid with shared x and y")
plt.show()

# Manual placement with add_axes and add_subplot. tight_layout cannot handle
# add_axes, so adjust the margins explicitly instead.
fig = plt.figure(figsize=(7, 3))
fig.add_axes([0.08, 0.15, 0.5, 0.75]).plot(synthetic["x"], synthetic["y"])
fig.add_subplot(1, 2, 2).hist(synthetic["noise"], bins=15)
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.9)
plt.show()

# A second y scale on one axes region.
fig, ax_left = plt.subplots(figsize=(6, 3.5))
ax_left.plot(synthetic["x"], synthetic["y"], color="tab:blue", label="y")
ax_left.set_ylabel("y", color="tab:blue")
ax_right = ax_left.twinx()
ax_right.plot(synthetic["x"], np.cumsum(synthetic["noise"]), color="tab:red", label="cumulative noise")
ax_right.set_ylabel("cumulative noise", color="tab:red")
plt.show()
"""))

# --------------------------------------------------------------------------- 8
cells.append(md(r"""
## 3. Line plots, labels, legends, grids, ticks, styles and rcParams

`Axes.plot` accepts a format string such as `"o--r"` (marker, line style, colour) or keyword arguments `marker`, `linestyle`/`ls`, `color`/`c`, `linewidth`/`lw`, `markersize`/`ms` and `alpha`. Important companion methods:

- `ax.set_title`, `ax.set_xlabel`, `ax.set_ylabel` (or the compact `ax.set(title=..., xlabel=..., ylabel=...)`).
- `ax.legend(loc="best", title=..., frameon=True)` labels the artists that were given a `label=`.
- `ax.grid(True, which="major", ls="--", alpha=0.4)`; `ax.minorticks_on()` enables minor ticks.
- `ax.set_xlim`/`set_ylim` set visible ranges; `ax.set_xticks` and `ax.set_xticklabels` place and name ticks; `ax.tick_params` changes tick and label appearance.

**Styles** are bundled rcParam presets: `plt.style.available` lists them and `plt.style.use(name)` applies one globally, while `with plt.style.context(name):` applies it temporarily. Since Matplotlib 3.6 the Seaborn styles are named `seaborn-v0_8-*`.

**rcParams** are the global defaults stored in `mpl.rcParams` / `plt.rcParams`. Change them with `.update({...})` or, preferably, scope them with `with plt.rc_context({...}):`.
"""))

# --------------------------------------------------------------------------- 9
cells.append(code(r"""
# A fully customised line plot: format strings, aliases, limits, ticks and legend.
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
ax.plot(synthetic["x"], 2.5 * synthetic["x"] - 3, "-b", lw=2, label="true line")
ax.plot(synthetic["x"], synthetic["y"], "o", color="tab:orange", ms=4,
        alpha=0.7, label="noisy samples")
ax.set_title("Line plot with full customization", fontsize=14, fontweight="bold")
ax.set_xlabel("x value")
ax.set_ylabel("y value")
ax.legend(loc="upper left", title="series", frameon=True)
ax.grid(True, which="major", ls="--", alpha=0.4)
ax.minorticks_on()
ax.set_xlim(0, 10)
ax.set_ylim(-8, 32)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(["0", "2", "4", "6", "8", "10"])
ax.tick_params(axis="both", labelsize=9)
fig.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------- 10
cells.append(code(r"""
# Styles and rcParams, both applied with a context manager so nothing leaks.
seaborn_styles = [s for s in plt.style.available if s.startswith("seaborn")]
print("bundled seaborn-style presets (first 6):", seaborn_styles[:6])
style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot"

with plt.style.context(style_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(synthetic["x"], synthetic["y"])
    ax.set_title("Temporary style: " + style_name)
    plt.show()

print("figure.figsize before:", plt.rcParams["figure.figsize"])
with plt.rc_context({"figure.figsize": (5, 3), "axes.titlesize": 11,
                     "xtick.color": "#555555", "lines.linewidth": 2.5}):
    fig, ax = plt.subplots()
    ax.plot(synthetic["x"], synthetic["y"])
    ax.set_title("Temporary rcParams")
    print("figure.figsize inside:", plt.rcParams["figure.figsize"])
    plt.show()
print("figure.figsize after:", plt.rcParams["figure.figsize"])
"""))

# --------------------------------------------------------------------------- 11
cells.append(md(r"""
## 4. Annotations and reference lines

`ax.annotate(text, xy=..., xytext=...)` places text at `xytext` and draws an arrow to `xy` when `arrowprops` is supplied. `xy` is in **data coordinates** by default; `xytext` uses the same coordinate system unless you set `textcoords`. Useful `arrowprops` keys include `arrowstyle` (`"->"`, `"<->"`, `"fancy"`), `color`, `lw`, `connectionstyle` and `shrinkA`/`shrinkB`.

`ax.text(x, y, s, bbox=...)` drops free-standing text, and `bbox` wraps it in a box. `ax.axhline` / `ax.axvline` draw horizontal and vertical reference lines across the whole axes; `ax.axvspan` / `ax.axhspan` shade bands. These tools turn a chart into an explanation.
"""))

# --------------------------------------------------------------------------- 12
cells.append(code(r"""
# Annotation with an arrow, a boxed text label, and reference lines.
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iris["petal_length"], iris["petal_width"], "o", alpha=0.6)
widest = iris.loc[iris["petal_width"].idxmax()]
ax.annotate("widest petal",
            xy=(widest["petal_length"], widest["petal_width"]),
            xytext=(3.0, 2.5),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5,
                            connectionstyle="arc3,rad=-0.15"),
            fontsize=10, color="crimson")
ax.text(1.0, 0.25, "text at data coordinates", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray"))
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.axvline(4.0, color="gray", ls=":", lw=1)
ax.axvspan(4.5, 6.0, color="tab:blue", alpha=0.08)
ax.set(xlabel="petal length (cm)", ylabel="petal width (cm)",
       title="Annotations and reference lines")
plt.show()
"""))

# --------------------------------------------------------------------------- 13
cells.append(md(r"""
## 5. Core chart types: bar, scatter, histogram, pie, boxplot, errorbar

| Method | Typical use | Key keywords |
|---|---|---|
| `ax.bar(x, height)` | categorical comparison | `width`, `color`, `edgecolor`, `bottom`, `label` |
| `ax.scatter(x, y)` | relationship between two numeric variables | `s`, `c`, `cmap`, `alpha`, `edgecolors`, `marker` |
| `ax.hist(x)` | distribution of one numeric variable | `bins`, `density`, `range`, `cumulative`, `edgecolor` |
| `ax.pie(sizes)` | part-to-whole composition | `labels`, `autopct`, `startangle`, `explode` |
| `ax.boxplot(data)` | five-number summary and outliers | `patch_artist`, `vert`, `whis`, `showmeans` |
| `ax.errorbar(x, y, yerr)` | value plus uncertainty | `xerr`, `fmt`, `capsize`, `ecolor`, `label` |

For `boxplot`, set tick names with `ax.set_xticks`/`set_xticklabels` rather than the version-dependent `labels=` argument. Passing a list of arrays draws one box per array. `hist` with `density=True` shows a probability density (area sums to 1), while the default counts observations in each bin.
"""))

# --------------------------------------------------------------------------- 14
cells.append(code(r"""
# Bar, colour-mapped scatter, and histogram in one row.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

by_day = tips.groupby("day")["tip"].mean()
axes[0].bar(by_day.index, by_day.values, color="steelblue", edgecolor="black")
axes[0].set(title="Mean tip by day", xlabel="day", ylabel="mean tip")
axes[0].tick_params(axis="x", rotation=20)

species_code = pd.Categorical(iris["species"]).codes
scatter = axes[1].scatter(iris["petal_length"], iris["petal_width"],
                          c=species_code, cmap="viridis", s=25,
                          alpha=0.8, edgecolors="none")
axes[1].set(title="Scatter with colour map", xlabel="petal length", ylabel="petal width")
fig.colorbar(scatter, ax=axes[1], label="species code")

axes[2].hist(tips["total_bill"], bins=20, density=True,
             color="seagreen", edgecolor="black", alpha=0.75)
axes[2].set(title="Histogram (density)", xlabel="total bill", ylabel="density")
fig.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------- 15
cells.append(code(r"""
# Pie, grouped boxplot, and errorbar in one row.
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

time_counts = tips["time"].value_counts()
axes[0].pie(time_counts.values, labels=time_counts.index.tolist(),
            autopct="%1.1f%%", startangle=90,
            explode=[0.03] * len(time_counts))
axes[0].set_title("Meal time share")

grouped = [tips.loc[tips["sex"] == sex, "total_bill"] for sex in ["Male", "Female"]]
axes[1].boxplot(grouped, patch_artist=True, showmeans=True)
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["Male", "Female"])
axes[1].set_title("Total bill by sex")
axes[1].grid(axis="y", alpha=0.3)

stats = tips.groupby("day")["tip"].agg(["mean", "std"])
positions = np.arange(len(stats))
axes[2].errorbar(positions, stats["mean"], yerr=stats["std"], fmt="o-",
                 capsize=4, color="darkorange", label="mean +/- sd")
axes[2].set_xticks(positions)
axes[2].set_xticklabels(stats.index)
axes[2].set_title("Tip by day with error bars")
axes[2].legend()
fig.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------- 16
cells.append(md(r"""
## 6. Contours, images, 3D, saving, and animation

For gridded data `ax.contour` draws isolines and `ax.contourf` fills between them; both take `X, Y, Z` arrays of equal shape plus `levels` and a `cmap`. `ax.imshow` treats an array as a pixel image; `origin="lower"` puts row zero at the bottom and `extent` maps pixels to data coordinates. Attach a colour scale with `fig.colorbar(mappable, ax=...)`.

Three-dimensional axes come from `fig.add_subplot(111, projection="3d")` after importing `mpl_toolkits.mplot3d` (the import registers the projection). They support `plot_surface(X, Y, Z, cmap=...)`, `scatter(x, y, z)` and `set_xlabel`/`set_ylabel`/`set_zlabel`.

`fig.savefig(path_or_buffer, dpi=..., format=..., bbox_inches="tight", facecolor=...)` writes a PNG, PDF, SVG or other format. Saving to an in-memory `io.BytesIO` avoids creating files.

Animations are built with `matplotlib.animation.FuncAnimation(fig, update, frames, interval)`: `update(frame)` redraws the artists once per frame, and `anim.save("out.gif", writer=PillowWriter(fps=10))` or `anim.to_html5_video()` produces output. Rendering needs a writer (and for HTML5 video, ffmpeg), so the cell below only constructs the object.
"""))

# --------------------------------------------------------------------------- 17
cells.append(code(r"""
# Gridded data for contour and image plots.
xgrid = np.linspace(-3, 3, 120)
ygrid = np.linspace(-3, 3, 120)
Xg, Yg = np.meshgrid(xgrid, ygrid)
Zg = np.sin(Xg) * np.cos(Yg) * np.exp(-0.15 * (Xg**2 + Yg**2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
filled = axes[0].contourf(Xg, Yg, Zg, levels=14, cmap="RdBu_r")
axes[0].contour(Xg, Yg, Zg, levels=8, colors="k", linewidths=0.4)
axes[0].set_title("contourf + contour")
fig.colorbar(filled, ax=axes[0])
image = axes[1].imshow(Zg, origin="lower", extent=[-3, 3, -3, 3],
                       aspect="auto", cmap="magma")
axes[1].set_title("imshow (pixel image)")
fig.colorbar(image, ax=axes[1])
fig.tight_layout()
plt.show()

# Basic 3D surface and point cloud.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

fig = plt.figure(figsize=(6, 5))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot_surface(Xg[::3, ::3], Yg[::3, ::3], Zg[::3, ::3],
                  cmap="viridis", linewidth=0, alpha=0.9)
xs = rng.uniform(-2.5, 2.5, 40)
ys = rng.uniform(-2.5, 2.5, 40)
ax3d.scatter(xs, ys, np.sin(xs) * np.cos(ys), color="crimson", s=18)
ax3d.set(xlabel="x", ylabel="y", zlabel="z", title="3D surface + scatter")
plt.show()

# Saving into memory instead of creating a file.
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="white")
print("PNG bytes:", len(buffer.getvalue()), "| signature:", buffer.getvalue()[:4])

# Animation note: build the object; rendering/saving is done explicitly.
from matplotlib.animation import FuncAnimation, PillowWriter

fig_anim, ax_anim = plt.subplots(figsize=(4, 3))
line, = ax_anim.plot([], [])
ax_anim.set_xlim(0, 5)
ax_anim.set_ylim(0, 5)


def update(frame):
    line.set_data([0, frame], [0, frame])
    return (line,)


animation = FuncAnimation(fig_anim, update, frames=np.linspace(0, 5, 20), interval=100)
print("animation type:", type(animation).__name__)
print("save with: animation.save('out.gif', writer=PillowWriter(fps=10))")
"""))

# --------------------------------------------------------------------------- 18
cells.append(md(r"""
### Try it yourself #1 — Matplotlib

Using only the OO API, build a `2 x 2` figure (`figsize=(10, 8)`, `constrained_layout=True`) with a suptitle "Try-it-yourself #1 — Matplotlib":

1. Top-left: scatter `sepal_length` against `sepal_width`, coloured by species code with a colour map.
2. Top-right: histogram of `sepal_width` with 15 bins.
3. Bottom-left: bar chart of mean `tip` by `day`.
4. Bottom-right: errorbar of mean `total_bill` by `day` with its standard deviation.
5. Add a light grid to every panel, label the axes sensibly, and save the figure to an in-memory `io.BytesIO` buffer.
"""))

# --------------------------------------------------------------------------- 19
cells.append(code(r"""
# Solution #1 — four axes, explicit targets, complete labelling.
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

codes = pd.Categorical(iris["species"]).codes
axes[0, 0].scatter(iris["sepal_length"], iris["sepal_width"], c=codes, cmap="Set1", s=20)
axes[0, 0].set(title="Sepal dimensions", xlabel="length (cm)", ylabel="width (cm)")

axes[0, 1].hist(iris["sepal_width"], bins=15, color="steelblue", edgecolor="black")
axes[0, 1].set(title="Sepal width distribution", xlabel="width (cm)", ylabel="count")

mean_tip = tips.groupby("day")["tip"].mean()
axes[1, 0].bar(mean_tip.index, mean_tip.values, color="seagreen")
axes[1, 0].set(title="Mean tip by day", xlabel="day", ylabel="mean tip")

bill_stats = tips.groupby("day")["total_bill"].agg(["mean", "std"])
positions = np.arange(len(bill_stats))
axes[1, 1].errorbar(positions, bill_stats["mean"], yerr=bill_stats["std"], fmt="o-", capsize=4)
axes[1, 1].set_xticks(positions)
axes[1, 1].set_xticklabels(bill_stats.index)
axes[1, 1].set(title="Total bill by day", xlabel="day", ylabel="mean +/- sd")

for axis in axes.ravel():
    axis.grid(alpha=0.25)
fig.suptitle("Try-it-yourself #1 — Matplotlib")
out = io.BytesIO()
fig.savefig(out, format="png")
print("saved bytes:", len(out.getvalue()))
plt.show()
"""))

# --------------------------------------------------------------------------- 20
cells.append(md(r"""
## 7. Seaborn — themes, palettes and the data-oriented API

Seaborn is built on Matplotlib but works with **whole DataFrames**: you name columns with `x=`, `y=`, `hue=`, `size=`, `style=` and `col=`, and Seaborn handles grouping, colour mapping, aggregation, statistical estimation and legends. There is no need to loop over groups yourself.

Configuration happens through:

- `sns.set_theme(style="whitegrid", context="notebook", palette="deep", font_scale=1.0)` — the single entry point; `sns.set()` is an alias.
- `sns.set_context("paper" | "notebook" | "talk" | "poster", font_scale=1.0)` — scales text and line elements for a target medium.
- `sns.set_palette("muted")` or a `palette=` argument on a single plot — sets the default categorical colours.
- `sns.color_palette("viridis", as_cmap=True)`, `sns.diverging_palette(220, 20, as_cmap=True)`, `sns.light_palette`, `sns.dark_palette`, `sns.cubehelix_palette`, `sns.blend_palette` — build colour lists or continuous colour maps.

Query the current settings with `sns.plotting_context()`, `sns.axes_style()` and `sns.color_palette()`. Colour lists support `.as_hex()` and `.as_rgb()`.
"""))

# --------------------------------------------------------------------------- 21
cells.append(code(r"""
# Themes, contexts and palettes.
sns.set_theme(style="whitegrid", context="notebook", palette="deep", font_scale=1.0)
print("context keys:", sorted(sns.plotting_context().keys())[:6])
print("deep palette as hex:", sns.color_palette("deep").as_hex())

sns.set_context("talk", font_scale=1.05)
print("font.size after set_context('talk'):", sns.plotting_context()["font.size"])

sns.set_palette("muted")
print("first muted colour:", sns.color_palette()[0])

continuous = sns.color_palette("viridis", as_cmap=True)
diverging = sns.diverging_palette(220, 20, as_cmap=True)
print("colour map types:", type(continuous).__name__, type(diverging).__name__)

# Restore a clean baseline for the plots below.
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
"""))

# --------------------------------------------------------------------------- 22
cells.append(md(r"""
## 8. Relational plots: `scatterplot` and `relplot`

`sns.scatterplot` draws one scatter plot and accepts `data`, `x`, `y`, `hue`, `style`, `size`, `sizes`, `palette`, `alpha` and `ax`. Pass `ax=` to draw into an existing Matplotlib axes.

`sns.relplot` is the figure-level wrapper: it creates its own figure and a `FacetGrid`, and `kind="scatter"` (default) or `kind="line"` selects the plot. Use `col=`, `row=` and `col_wrap=` for small multiples, plus `height=` and `aspect=` to size each panel. Because it owns the figure, do not pass `ax=` to `relplot`.
"""))

# --------------------------------------------------------------------------- 23
cells.append(code(r"""
# Axes-level scatterplot: four visual channels from four columns.
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=iris, x="petal_length", y="petal_width",
                hue="species", style="species", size="sepal_length",
                sizes=(25, 180), palette="Set2", alpha=0.85, ax=ax)
ax.set_title("scatterplot: hue + style + size")
plt.show()

# Figure-level relplot with facets: one panel per species.
sns.relplot(data=iris, x="sepal_length", y="sepal_width", hue="species",
            col="species", kind="scatter", palette="Set2",
            height=3.2, aspect=1.1, alpha=0.8)
plt.show()

# A line-oriented relational plot with automatic aggregation and confidence bands.
sns.relplot(data=tips, x="size", y="total_bill", hue="sex",
            kind="line", errorbar=("ci", 95), height=4, aspect=1.3)
plt.show()
"""))

# --------------------------------------------------------------------------- 24
cells.append(md(r"""
## 9. Categorical plots

Categorical functions take a categorical `x` (and usually numeric `y`). The main choices:

| Function | What it shows |
|---|---|
| `stripplot` | every observation as a jittered dot (`jitter`, `dodge`) |
| `swarmplot` | dots repositioned so they do not overlap (`size`, `dodge`) |
| `boxplot` | quartiles and whiskers (`dodge`, `whis`, `gap`) |
| `violinplot` | kernel density of each group (`split`, `inner`, `bw`) |
| `barplot` | estimate (mean by default) plus a confidence interval (`estimator`, `errorbar`) |
| `pointplot` | estimate and interval connected by a line (`dodge`, `linestyles`, `markers`) |
| `catplot` | figure-level wrapper; `kind=` picks any of the above and adds `row`/`col` facets |

Since Seaborn 0.12, confidence intervals use `errorbar=("ci", 95)` (the older `ci=` is deprecated). A `violinplot` with `split=True` requires exactly two levels of `hue`. `catplot` owns its figure, so pass facet controls such as `col=`, `height=` and `aspect=` to it rather than to an axes.
"""))

# --------------------------------------------------------------------------- 25
cells.append(code(r"""
# Six categorical views of the same tips data, plus a faceted catplot.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

sns.stripplot(data=tips, x="day", y="total_bill", hue="sex",
              dodge=True, alpha=0.65, ax=axes[0, 0])
axes[0, 0].set_title("stripplot")

sns.swarmplot(data=tips, x="day", y="total_bill", hue="sex",
              dodge=True, size=3, ax=axes[0, 1])
axes[0, 1].set_title("swarmplot")

sns.boxplot(data=tips, x="day", y="total_bill", hue="sex",
            dodge=True, ax=axes[0, 2])
axes[0, 2].set_title("boxplot")

sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
               split=True, inner="quartile", ax=axes[1, 0])
axes[1, 0].set_title("violinplot (split)")

sns.barplot(data=tips, x="day", y="tip", hue="sex",
            errorbar=("ci", 95), capsize=0.1, ax=axes[1, 1])
axes[1, 1].set_title("barplot")

sns.pointplot(data=tips, x="day", y="tip", hue="sex", dodge=True,
              errorbar=("ci", 95), markers=["o", "s"], linestyles=["-", "--"],
              ax=axes[1, 2])
axes[1, 2].set_title("pointplot")

fig.tight_layout()
plt.show()

# Figure-level categorical plot: one boxplot panel per meal time.
sns.catplot(data=tips, x="day", y="total_bill", hue="sex",
            col="time", kind="box", height=3.8, aspect=1.15)
plt.show()
"""))

# --------------------------------------------------------------------------- 26
cells.append(md(r"""
## 10. Distribution plots

- `sns.histplot(x=...)` is a Matplotlib histogram with DataFrame awareness: `bins`, `stat="count" | "density" | "probability" | "percent"`, `kde=True`, `element="bars" | "step" | "poly"`, `cumulative=True` and `common_norm=False` when splitting by `hue`.
- `sns.kdeplot(x=..., y=...)` draws kernel density estimates; `fill=True`, `levels=`, `thresh=` and `bw_adjust=` control 1-D and 2-D densities.
- `sns.ecdfplot(x=...)` shows the empirical cumulative distribution function; `stat="proportion"` (default) or `"count"`.
- `sns.displot` is the figure-level version with `kind="hist" | "kde" | "ecdf"` plus `row`/`col` facets and `facet_kws`.
- `sns.jointplot(x=..., y=..., kind=...)` combines a bivariate plot with marginal distributions; `sns.pairplot` builds an all-pairs matrix with `hue=`, `diag_kind="kde"`, `corner=True` and `height=`.
"""))

# --------------------------------------------------------------------------- 27
cells.append(code(r"""
# Axes-level distributions: histogram, KDE and ECDF.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(data=tips, x="total_bill", hue="sex", bins=20, kde=True,
             element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("histplot + KDE")
sns.kdeplot(data=iris, x="petal_length", y="petal_width", hue="species",
            fill=True, levels=5, thresh=0.05, alpha=0.5, ax=axes[1])
axes[1].set_title("2-D kdeplot")
sns.ecdfplot(data=tips, x="tip", hue="time", stat="proportion", ax=axes[2])
axes[2].set_title("ecdfplot")
fig.tight_layout()
plt.show()

# Figure-level displot with a facet per meal time and independent y scales.
sns.displot(data=tips, x="total_bill", hue="sex", kind="hist", col="time",
            bins=15, element="step", fill=False,
            facet_kws=dict(sharey=False), height=3.8, aspect=1.15)
plt.show()

# Joint and pairwise views.
sns.jointplot(data=iris, x="petal_length", y="petal_width",
              hue="species", kind="scatter", height=5)
plt.show()

sns.pairplot(iris, hue="species", diag_kind="kde", corner=True, height=2.2)
plt.show()
"""))

# --------------------------------------------------------------------------- 28
cells.append(md(r"""
## 11. Heatmaps, FacetGrid and regression plots

A correlation matrix is the classic use of `sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, cbar_kws=...)`. Keep only numeric columns first with `select_dtypes`.

`sns.FacetGrid(data, row=..., col=..., height=..., aspect=...)` is the lower-level engine behind `relplot`/`displot`/`catplot`. Create it, then call `g.map(func, column_name, **kwargs)` to apply a plotting function to each subset; `g.set_axis_labels`, `g.set_titles`, `g.add_legend` and `g.fig` finish the layout. (`g.map_dataframe` is the older, deprecated alias.)

For fitted trends, `sns.regplot(x=..., y=..., data=...)` adds a linear fit and confidence band to one axes (`scatter_kws`, `line_kws`, `ci`, `order`, `logx`). `sns.lmplot` is the figure-level equivalent with `hue`/`col`/`row` facets.
"""))

# --------------------------------------------------------------------------- 29
cells.append(code(r"""
# Correlation heatmap of the numeric diamond attributes.
numeric_diamonds = diamonds.select_dtypes(include="number")
corr = numeric_diamonds.corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Diamond numeric correlations")
plt.show()

# Manual small multiples with FacetGrid.map.
grid = sns.FacetGrid(tips, col="time", row="sex", height=3.4, aspect=1.2)
grid.map(sns.histplot, "total_bill", bins=15, kde=True)
grid.set_axis_labels("total bill", "count")
grid.set_titles("time={col_name} | sex={row_name}")
plt.show()

# Regression on one axes and faceted regression across groups.
fig, ax = plt.subplots(figsize=(6, 4))
sns.regplot(data=tips, x="total_bill", y="tip", ax=ax,
            scatter_kws={"alpha": 0.5, "s": 20},
            line_kws={"color": "crimson"}, ci=95)
ax.set_title("regplot: OLS trend with 95% CI")
plt.show()

sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex", col="time",
           height=4, aspect=1.1, scatter_kws={"alpha": 0.5},
           line_kws={"lw": 1.5})
plt.show()
"""))

# --------------------------------------------------------------------------- 30
cells.append(md(r"""
### Try it yourself #2 — Seaborn

1. With `sns.set_theme(style="darkgrid", context="talk")`, draw a `relplot` of `sepal_length` versus `petal_length`, coloured by `species`, faceted by `species` into one column (`col_wrap=3`).
2. Draw a `catplot` with `kind="violin"` showing `total_bill` by `day`, split by `sex`, faceted by `time`.
3. Draw a `displot` of `diamonds["price"]` as a KDE with `hue="cut"`, `common_norm=False`, `fill=True`, `height=4`, `aspect=1.3`.
"""))

# --------------------------------------------------------------------------- 31
cells.append(code(r"""
# Solution #2 — three figure-level Seaborn calls.
sns.set_theme(style="darkgrid", context="talk")

sns.relplot(data=iris, x="sepal_length", y="petal_length", hue="species",
            col="species", col_wrap=3, kind="scatter",
            height=3, aspect=1.1, alpha=0.85)
plt.show()

sns.catplot(data=tips, x="day", y="total_bill", hue="sex", col="time",
            kind="violin", split=True, inner="quartile",
            height=4, aspect=1.1)
plt.show()

sns.displot(data=diamonds, x="price", hue="cut", kind="kde",
            common_norm=False, fill=True, height=4, aspect=1.3)
plt.show()

sns.set_theme(style="whitegrid", context="notebook")
"""))

# --------------------------------------------------------------------------- 32
cells.append(md(r"""
## 12. Plotly and Bokeh — interactive figures in Jupyter

Matplotlib and Seaborn produce static images. **Plotly** and **Bokeh** produce browser-based, interactive graphics: hovering reveals values, legend entries toggle series, and toolbars zoom and pan. In Jupyter they render inline as HTML/JavaScript widgets — `fig.show()` for Plotly and `show(fig)` for Bokeh — so no web server or internet connection is needed.

- **Plotly Express** (`plotly.express`, imported as `px`) is a high-level, one-call API. `px.scatter(data_frame, x, y, color, size, hover_data, title, template)` returns a `Figure`. Use `fig.show()` to display it, `fig.write_html("f.html")` to save a standalone page, and `plotly.io.renderers` to inspect the active renderer.
- **Bokeh** (`from bokeh.plotting import figure, show`) is more explicit. Build a `ColumnDataSource`, create a `figure`, add tools such as `HoverTool(tooltips=[...])`, draw with `p.scatter(...)`, then call `show(p)`. `from bokeh.io import output_notebook` enables classic inline output in older notebooks; recent Bokeh versions also detect Jupyter automatically.

Use Plotly Express for rapid, attractive interactive EDA; use Bokeh when you need finer control over interactions, custom JavaScript callbacks or streaming data. Both complement (rather than replace) the static plotting stack.
"""))

# --------------------------------------------------------------------------- 33
cells.append(code(r"""
# Plotly Express interactive scatter: hover, legend toggling and zoom included.
import plotly.express as px

plotly_fig = px.scatter(
    data_frame=iris,
    x="petal_length",
    y="petal_width",
    color="species",
    size="sepal_width",
    hover_data=["sepal_length"],
    title="Plotly Express — iris petals",
    template="plotly_white",
)
print("figure type:", type(plotly_fig).__name__)
plotly_fig.show()
# Equivalent offline export: plotly_fig.write_html("iris.html", include_plotlyjs="cdn")
"""))

# --------------------------------------------------------------------------- 34
cells.append(code(r"""
# Bokeh scatter with a hover tool driven by a ColumnDataSource.
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show

palette = dict(zip(sorted(iris["species"].unique()),
                   ["#1f77b4", "#ff7f0e", "#2ca02c"]))
bokeh_frame = iris.copy()
bokeh_frame["colour"] = bokeh_frame["species"].map(palette)
source = ColumnDataSource(bokeh_frame)

bokeh_fig = figure(
    title="Bokeh — iris sepals",
    x_axis_label="sepal length (cm)",
    y_axis_label="sepal width (cm)",
    width=550, height=380,
    tools="pan,wheel_zoom,box_zoom,reset",
)
bokeh_fig.add_tools(HoverTool(tooltips=[
    ("species", "@species"),
    ("petal length", "@petal_length{0.00}"),
    ("petal width", "@petal_width{0.00}"),
]))
bokeh_fig.scatter(x="sepal_length", y="sepal_width", source=source,
                  size=9, color="colour", alpha=0.75, legend_field="species")
bokeh_fig.legend.title = "species"
output_notebook()
show(bokeh_fig)
"""))

# --------------------------------------------------------------------------- 35
cells.append(md(r"""
## 13. SymPy — symbols, expressions and exact arithmetic

SymPy works with **symbolic expressions**, not just numbers. `sp.Symbol("x")` creates one symbol; `sp.symbols("x y z")` creates several; `sp.symbols("x0:5")` creates indexed names. Assumptions such as `positive=True`, `real=True` or `integer=True` let SymPy simplify more aggressively.

Expressions are built from Python operators and SymPy functions: `x**2`, `sp.sin(x)`, `sp.sqrt(x)`, `sp.exp(x)`, `sp.log(x)`. The numeric atoms `sp.Integer(2)`, `sp.Rational(1, 3)`, `sp.pi`, `sp.E`, `sp.I` and `sp.oo` stay **exact**: `sp.Rational(1, 3) + sp.Rational(1, 6)` gives `1/2`, whereas Python floats give `0.49999999999999994`. Convert to floating point only when you need it, using `expr.evalf(n)` or `sp.N(expr, n)`.

Because SymPy overloads arithmetic operators, `x + x` becomes the expression `2*x`. This is why statements that would raise a `NameError` with plain Python (an undefined variable) work as long as the symbol has been declared.
"""))

# --------------------------------------------------------------------------- 36
cells.append(code(r"""
# Symbols, expression construction and exact versus floating-point values.
x, y = sp.symbols("x y")
positive_t = sp.Symbol("t", positive=True)
indexed = sp.symbols("a0:4")

expression = sp.sin(x) + sp.Rational(1, 3) * x**2 - sp.sqrt(2)
print("expression:", expression)
print("free symbols:", expression.free_symbols)
print("symbol assumptions:", positive_t.is_positive, x.is_real)
print("indexed symbols:", indexed)

print("exact sum:", sp.Rational(1, 3) + sp.Rational(1, 6))
print("plain float sum:", 1 / 3 + 1 / 6)
print("evalf(15):", expression.subs(x, 1).evalf(15))
print("N(pi, 20):", sp.N(sp.pi, 20))
"""))

# --------------------------------------------------------------------------- 37
cells.append(code(r"""
# Rewriting expressions into more useful forms.
raw = (x**2 - 1) / (x - 1)
print("simplify:", sp.simplify(raw))
print("expand:", sp.expand((x + 1)**3))
print("factor:", sp.factor(x**3 - 1))
print("collect:", sp.collect(x**2 * y + x * y**2 + 3 * x**2 - 2 * x, x))
print("cancel:", sp.cancel((x**2 - 1) / (x**3 - 1)))
print("apart:", sp.apart(1 / (x**2 - 1), x))
print("together:", sp.together(1 / x + 1 / y))
print("trigsimp:", sp.trigsimp(sp.sin(x)**2 + sp.cos(x)**2))
print("radsimp:", sp.radsimp(1 / (sp.sqrt(x) + 1)))
"""))

# --------------------------------------------------------------------------- 38
cells.append(code(r"""
# Solving equations and systems exactly.
quadratic = x**2 - 5 * x + 6
print("solve polynomial:", sp.solve(quadratic, x))
print("solveset over Reals:", sp.solveset(quadratic, x, domain=sp.S.Reals))
print("solve equation object:", sp.solve(sp.Eq(x + 2, 5), x))
print("solve system:", sp.solve([x + y - 2, x - y], [x, y]))

# Assumptions prune impossible roots.
print("positive roots:", sp.solve(x**2 - 4, x).copy())  # unrestricted
positive_x = sp.Symbol("u", positive=True)
print("with positive assumption:", sp.solve(positive_x**2 - 4, positive_x))
"""))

# --------------------------------------------------------------------------- 39
cells.append(code(r"""
# Calculus: differentiation, integration, limits and Taylor series.
f = sp.sin(x) * sp.exp(x)
print("first derivative:", sp.diff(f, x))
print("second derivative of x**4:", sp.diff(x**4, x, 2))
print("indefinite integral:", sp.integrate(x**2 * sp.cos(x), x))
print("definite integral 0..1 of x**2:", sp.integrate(x**2, (x, 0, 1)))
print("limit sin(x)/x at 0:", sp.limit(sp.sin(x) / x, x, 0))
print("limit 1/x from the right:", sp.limit(1 / x, x, 0, dir="+"))
print("limit at infinity:", sp.limit((2 * x**2 + 1) / (x**2 + 3), x, sp.oo))

series = sp.series(sp.exp(x), x, 0, 6)
print("series with remainder:", series)
print("series without remainder:", series.removeO())
"""))

# --------------------------------------------------------------------------- 40
cells.append(code(r"""
# Matrices and linear algebra.
matrix = sp.Matrix([[1, 2], [3, 4]])
print("matrix:", matrix)
print("determinant:", matrix.det())
print("inverse:", matrix.inv())
print("transpose:", matrix.T)
print("eigenvalues:", matrix.eigenvals())
print("eigenvectors:", matrix.eigenvects())
print("reduced row echelon:", matrix.rref())
print("identity:", sp.eye(3))
print("zeros:", sp.zeros(2, 3))
print("diagonal:", sp.diag(1, 2, 3))
print("solve M v = b:", sp.linsolve((matrix, sp.Matrix([5, 11])), (x, y)))
"""))

# --------------------------------------------------------------------------- 41
cells.append(code(r"""
# Substitution, numerical evaluation, lambdify and symbolic plotting.
expr = sp.sin(x) + x**2
print("single substitution:", expr.subs(x, 0))
print("multiple substitutions:", (x**2 + y).subs({x: 2, y: 3}))
print("float value:", sp.N(expr.subs(x, 1), 12))

# lambdify compiles a SymPy expression into a NumPy-aware Python function.
fast = sp.lambdify(x, expr, "numpy")
values = fast(np.array([0.0, 1.0, 2.0]))
print("lambdified values:", np.round(values, 5))

# sp.plot uses a Matplotlib backend; sp.plot3d handles surfaces.
sp.plot(sp.sin(x), sp.cos(x), (x, -sp.pi, sp.pi),
        title="SymPy plot", xlabel="x", ylabel="value")
sp.plotting.plot3d(x * y, (x, -2, 2), (y, -2, 2), title="SymPy plot3d")
"""))

# --------------------------------------------------------------------------- 42
cells.append(md(r"""
### Try it yourself #3 — SymPy

1. Declare `t` as a real symbol and build `g = exp(-t) * sin(3*t)`. Print `sp.diff(g, t)` and `sp.integrate(g, (t, 0, sp.oo))`.
2. Compute the exact value of `sp.Rational(7, 12) + sp.Rational(5, 8)` and show its 20-digit decimal form.
3. Build the matrix `A = [[2, 1], [1, 2]]`, print its eigenvalues and eigenvectors, then use `sp.lambdify` to turn the scalar expression `t**2 + 1` into a NumPy function and evaluate it on `np.array([0, 1, 2])`.
"""))

# --------------------------------------------------------------------------- 43
cells.append(code(r"""
# Solution #3 — exact calculus, rationals, eigen-decomposition and lambdify.
t = sp.Symbol("t", real=True)
g = sp.exp(-t) * sp.sin(3 * t)
print("derivative:", sp.diff(g, t))
print("improper integral:", sp.integrate(g, (t, 0, sp.oo)))

exact = sp.Rational(7, 12) + sp.Rational(5, 8)
print("exact:", exact, "| decimal:", sp.N(exact, 20))

A = sp.Matrix([[2, 1], [1, 2]])
print("eigenvalues:", A.eigenvals())
print("eigenvectors:", A.eigenvects())

converted = sp.lambdify(t, t**2 + 1, "numpy")
print("lambdified:", converted(np.array([0, 1, 2])))
"""))

# --------------------------------------------------------------------------- 44
cells.append(md(r"""
## 14. Statsmodels — OLS regression and interpretation

`statsmodels.formula.api` uses R-style formulas handled by patsy:

- `smf.ols("tip ~ total_bill + size", data=tips).fit()` fits an ordinary least-squares model; the intercept is added automatically.
- Categorical predictors use `C(day)`; interactions use `a * b` (main effects plus product) or `a:b` (product only); transformations use `np.log(total_bill)` or `I(total_bill**2)`.
- `.summary()` prints a full report. Key rows: `coef` (estimated effect), `std err`, `t`, `P>|t|` (two-sided p-value) and `[0.025 0.975]` (95% confidence interval). Key header values: `R-squared` (share of variance explained), `Adj. R-squared` (penalised for predictors), `F-statistic` and `Prob (F-statistic)` (joint test that all slopes are zero), `AIC`/`BIC` (model comparison, lower is better), `No. Observations`, `Df Residuals` and `Df Model`.

The fitted result also exposes attributes directly: `.params`, `.bse`, `.tvalues`, `.pvalues`, `.conf_int()`, `.rsquared`, `.rsquared_adj`, `.fvalue`, `.f_pvalue`, `.aic`, `.bic`, `.resid`, `.fittedvalues` and `.nobs`.

To test a specific coefficient or joint hypotheses, use `.t_test("total_bill = 1")` or `.f_test("size = 0")`.
"""))

# --------------------------------------------------------------------------- 45
cells.append(code(r"""
# OLS with the formula API and a guided read of the summary.
ols_model = smf.ols("tip ~ total_bill + size", data=tips).fit()
print(ols_model.summary())

print("\ncoefficients:\n", ols_model.params)
print("p-values:\n", ols_model.pvalues)
print("95% confidence intervals:\n", ols_model.conf_int())
print("R-squared / adjusted:", round(ols_model.rsquared, 4), round(ols_model.rsquared_adj, 4))
print("F statistic / p:", round(ols_model.fvalue, 3), ols_model.f_pvalue)
print("AIC / BIC:", round(ols_model.aic, 2), round(ols_model.bic, 2))
print("t-test that total_bill coefficient equals 1:\n", ols_model.t_test("total_bill = 1"))
"""))

# --------------------------------------------------------------------------- 46
cells.append(code(r"""
# Array API: add a constant column, since sm.OLS does not add an intercept.
X = sm.add_constant(tips[["total_bill", "size"]])
y = tips["tip"]
array_model = sm.OLS(y, X).fit()
print("array-API coefficients:\n", array_model.params.round(4))

# Multiple regression with categorical predictors and an interaction.
multi_model = smf.ols("tip ~ total_bill * sex + C(time) + size", data=tips).fit()
print(multi_model.summary())
print("\nparameter names:", multi_model.params.index.tolist())
print("joint F-test of interaction terms:\n", multi_model.f_test("total_bill:sex[T.Male] = 0"))

# Quick diagnostic plot: residuals versus fitted values and a Q-Q plot.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(ols_model.fittedvalues, ols_model.resid, alpha=0.6)
axes[0].axhline(0, color="gray", ls="--")
axes[0].set(xlabel="fitted", ylabel="residual", title="Residuals vs fitted")
sm.qqplot(ols_model.resid, line="45", ax=axes[1])
axes[1].set_title("Normal Q-Q plot")
fig.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------- 47
cells.append(code(r"""
# statsmodels.stats.api: proportion tests; outliers_influence: diagnostics.
from statsmodels.stats.api import proportion_confint, proportions_ztest
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

smoker_count = int((tips["smoker"] == "Yes").sum())
smoker_nobs = len(tips)
print("Wilson 95% CI for smoker proportion:",
      proportion_confint(smoker_count, smoker_nobs, alpha=0.05, method="wilson"))
print("one-sample z-test against 0.40:",
      proportions_ztest(count=smoker_count, nobs=smoker_nobs, value=0.40))

male_rows = tips[tips["sex"] == "Male"]
female_rows = tips[tips["sex"] == "Female"]
smoker_counts = [int((male_rows["smoker"] == "Yes").sum()),
                 int((female_rows["smoker"] == "Yes").sum())]
print("two-sample z-test (smoker rate: male vs female):",
      proportions_ztest(count=smoker_counts, nobs=[len(male_rows), len(female_rows)]))

# Studentised residuals and Bonferroni-adjusted p-values flag unusual rows.
outlier_table = ols_model.outlier_test()
print("most extreme observations:\n", outlier_table.sort_values("bonf(p)").head())

# Multicollinearity and heteroskedasticity checks.
vif = pd.Series({name: variance_inflation_factor(X.values, idx)
                 for idx, name in enumerate(X.columns)})
print("VIF:\n", vif.round(3))
print("Breusch-Pagan (LM, p, F, F p):",
      tuple(round(v, 4) for v in het_breuschpagan(ols_model.resid, ols_model.model.exog)))
"""))

# --------------------------------------------------------------------------- 48
cells.append(code(r"""
# Time series: build a synthetic AR(1) process, then fit ARIMA(1, 0, 0).
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

rng_ts = np.random.default_rng(7)
n = 200
innovations = rng_ts.normal(0, 1, n)
ar_values = np.zeros(n)
for time_index in range(1, n):
    ar_values[time_index] = 0.6 * ar_values[time_index - 1] + innovations[time_index]
ar_series = pd.Series(ar_values, name="ar1")

adf_statistic, adf_pvalue = adfuller(ar_series, result_object=False)[:2]
print("ADF statistic / p-value:", round(adf_statistic, 4), round(adf_pvalue, 4))
print("ACF lags 0-3:", np.round(acf(ar_series, nlags=3), 3))
print("PACF lags 1-3:", np.round(pacf(ar_series, nlags=3)[1:], 3))

arima_result = ARIMA(ar_series, order=(1, 0, 0), trend="c").fit()
print(arima_result.summary())
print("10-step forecast:\n", arima_result.forecast(steps=10).round(3))
"""))

# --------------------------------------------------------------------------- 49
cells.append(code(r"""
# Logistic regression with the formula API. The response must be numeric, so
# encode the two-level "smoker" string as 0/1 in a copy of the data.
tips_logit = tips.copy()
tips_logit["smoker_flag"] = (tips_logit["smoker"] == "Yes").astype(int)
logit_model = smf.logit("smoker_flag ~ total_bill + size + tip", data=tips_logit).fit(disp=False)
print(logit_model.summary())

# Coefficients are on the log-odds scale; exponentiate for odds ratios.
print("odds ratios:\n", np.exp(logit_model.params).round(4))
print("predicted probabilities for first five rows:\n",
      logit_model.predict(tips_logit.head()).round(3))
print("pseudo R-squared:", round(logit_model.prsquared, 4))
"""))

# --------------------------------------------------------------------------- 50
cells.append(code(r"""
# ANOVA: one-way table and nested model comparison.
from statsmodels.stats.anova import anova_lm

day_model = smf.ols("tip ~ C(day)", data=tips).fit()
print("Type II ANOVA for tip ~ C(day):\n", anova_lm(day_model, typ=2))

full_model = smf.ols("tip ~ C(day) + total_bill", data=tips).fit()
print("Comparison of nested models:\n", anova_lm(day_model, full_model))
print("full-model R-squared:", round(full_model.rsquared, 4))
"""))

# --------------------------------------------------------------------------- 51
cells.append(code(r"""
# Statsmodels versus scikit-learn on the same linear regression.
features = ["total_bill", "size"]
target = tips["tip"]

sm_result = sm.OLS(target, sm.add_constant(tips[features])).fit()
sk_model = LinearRegression().fit(tips[features], target)

sm_coefficients = sm_result.params.to_numpy()
sk_coefficients = np.concatenate([[sk_model.intercept_], sk_model.coef_])
print("statsmodels [intercept, slopes]:", np.round(sm_coefficients, 6))
print("scikit-learn [intercept, slopes]:", np.round(sk_coefficients, 6))
print("coefficients match:", np.allclose(sm_coefficients, sk_coefficients))

print("\nstatsmodels R-squared:", round(sm_result.rsquared, 4))
print("scikit-learn R-squared:", round(sk_model.score(tips[features], target), 4))
print("statsmodels gives p-values / confidence intervals; scikit-learn emphasises prediction pipelines.")
"""))

# --------------------------------------------------------------------------- 52
cells.append(md(r"""
### Try it yourself #4 — Statsmodels

1. Fit `smf.ols("total_bill ~ tip + size + C(time)", data=tips)`, print the summary and report the coefficient, p-value and 95% confidence interval for `tip`.
2. Run a Type II ANOVA on that model with `anova_lm(..., typ=2)` and name the factor with the largest F value.
3. Reproduce the `total_bill` model with `sm.OLS` plus `sm.add_constant` and confirm with `np.allclose` that the coefficients match `LinearRegression`.
"""))

# --------------------------------------------------------------------------- 53
cells.append(code(r"""
# Solution #4 — formula model, ANOVA and cross-library coefficient check.
chosen = smf.ols("total_bill ~ tip + size + C(time)", data=tips).fit()
print(chosen.summary())
print("\ntip coefficient:", round(chosen.params["tip"], 4))
print("tip p-value:", round(chosen.pvalues["tip"], 6))
print("tip 95% CI:", chosen.conf_int().loc["tip"].round(4).tolist())

print("\nType II ANOVA:\n", anova_lm(chosen, typ=2))

predictors = ["tip", "size", "C(time)"]
sm_coefs = chosen.params.to_numpy()
manual = sm.OLS(tips["total_bill"],
                sm.add_constant(pd.get_dummies(
                    tips[["tip", "size", "time"]], columns=["time"],
                    drop_first=True, dtype=float))).fit()
sk_check = LinearRegression().fit(
    pd.get_dummies(tips[["tip", "size", "time"]], columns=["time"],
                   drop_first=True, dtype=float),
    tips["total_bill"])
sk_coefs = np.concatenate([[sk_check.intercept_], sk_check.coef_])
print("\ncoefficients align (statsmodels formula vs scikit-learn):",
      np.allclose(np.sort(sm_coefs), np.sort(sk_coefs), atol=1e-6))
print("manual array API R-squared:", round(manual.rsquared, 4))
"""))

# --------------------------------------------------------------------------- 54
cells.append(md(r"""
## Cheat sheet / key takeaways

**Matplotlib**

| Task | Call |
|---|---|
| Create axes | `fig, ax = plt.subplots(nrows, ncols, figsize=(w,h), sharex=True)` |
| Add axes manually | `fig.add_axes([l,b,w,h])`, `fig.add_subplot(n,m,i)` |
| Plot types | `ax.plot`, `bar`, `scatter`, `hist`, `pie`, `boxplot`, `errorbar`, `contourf`, `imshow` |
| 3D | `fig.add_subplot(111, projection="3d")`, `ax.plot_surface`, `ax.scatter` |
| Decorate | `ax.set(...)`, `legend`, `grid`, `set_xticks`, `annotate`, `text`, `axhline` |
| Layout / style | `fig.tight_layout()`, `constrained_layout=True`, `plt.style.context`, `plt.rc_context` |
| Save | `fig.savefig(path_or_buffer, dpi=..., bbox_inches="tight")` |

**Seaborn**

| Family | Functions |
|---|---|
| Setup | `set_theme`, `set_context`, `set_palette`, `color_palette`, `diverging_palette` |
| Relational | `scatterplot`, `lineplot`, `relplot` |
| Categorical | `stripplot`, `swarmplot`, `boxplot`, `violinplot`, `barplot`, `pointplot`, `catplot` |
| Distributions | `histplot`, `kdeplot`, `ecdfplot`, `displot`, `jointplot`, `pairplot` |
| Other | `heatmap`, `FacetGrid.map`, `regplot`, `lmplot` |
| Figure-level | `relplot`, `displot`, `catplot`, `lmplot`, `jointplot`, `pairplot` own their figure |

**Interactive**: Plotly `px.scatter(...).show()`; Bokeh `figure` + `ColumnDataSource` + `HoverTool` + `show(fig)`. Both render inline in Jupyter.

**SymPy**: `symbols` → build expression → `simplify`/`expand`/`factor`/`collect`; `solve`/`solveset`; `diff`/`integrate`/`limit`/`series`; `Matrix` methods (`det`, `inv`, `eigenvals`, `rref`); `subs`, `evalf`/`N`, `lambdify`, `plot`. Keep values exact until the final step.

**Statsmodels**: `smf.ols("y ~ x1 + C(g)", data).fit()` → `.summary()`, `.params`, `.pvalues`, `.conf_int()`, `.rsquared`; `sm.stats.api` for proportions; `.outlier_test()` for outliers; `ARIMA(y, order=(p,d,q)).fit()` for time series; `smf.logit` for binary outcomes; `anova_lm` for ANOVA. Statsmodels = inference and diagnostics; scikit-learn = prediction and deployment.
"""))

# --------------------------------------------------------------------------- 55
cells.append(md(r"""
## Further reading

- [Matplotlib: Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) — shortest route to a first plot.
- [Matplotlib: Tutorials](https://matplotlib.org/stable/tutorials/index.html) — anatomy, styles, legends, annotations, 3D and animations.
- [Seaborn: Tutorial](https://seaborn.pydata.org/tutorial.html) — relational, categorical, distribution and regression sections.
- [Seaborn: Choosing the right plot](https://seaborn.pydata.org/tutorial/function_overview.html) — decision guide across plot families.
- [Plotly Express reference](https://plotly.com/python-api-reference/plotly.express.html) — every Express function and argument.
- [Bokeh reference guide](https://docs.bokeh.org/en/latest/docs/reference.html) — models, tools and properties.
- [SymPy: Intro tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html) and [SymPy: Solvers](https://docs.sympy.org/latest/modules/solvers/solvers.html).
- [Statsmodels: Regression](https://www.statsmodels.org/stable/regression.html) and [Statsmodels: Time series analysis](https://www.statsmodels.org/stable/tsa.html).
- [Patsy formula syntax](https://patsy.readthedocs.io/en/latest/formulas.html) — the language behind statsmodels formulas.
- [NumPy](https://numpy.org/doc/stable/) and [pandas](https://pandas.pydata.org/docs/) documentation for the array and DataFrame fundamentals used throughout.
"""))

write_notebook(
    "notebooks/data-science/unit4_plotting_sympy_statsmodels.ipynb",
    cells,
)
