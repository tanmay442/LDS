"""Builder for Unit V, Part 1 - scikit-learn and traditional ML reference notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelper import md, code, write_notebook

cells = []

cells.append(md(r"""
# Unit V, Part 1 — scikit-learn and Traditional Machine Learning in Depth

scikit-learn is the default Python library for classical (non-deep) machine learning. It exposes a small, consistent family of objects called **estimators** that learn from data and make predictions. This notebook is a long-lived reference: it walks the full workflow from loading data to evaluating, tuning, interpreting and saving models, and it covers the surrounding ecosystem — imbalanced-learn, XGBoost, LightGBM, CatBoost, mlxtend, Yellowbrick, Optuna and SHAP. Every training run is deliberately tiny so the whole notebook executes in seconds, and all data comes from sklearn's bundled datasets or the local CSV files in `../../data`.

**Learning objectives**
- Explain the scikit-learn API philosophy: `estimator`, `fit`, `predict`, `transform`, `fit_transform`, `get_params`/`set_params`, fitted attributes ending in `_`, and the `Bunch` object.
- Load built-in datasets (`load_iris`, `load_wine`, `load_breast_cancer`, `load_digits`, `load_diabetes`), generate synthetic data with `make_classification`, `make_regression`, `make_blobs`, `make_moons`, and know what `fetch_openml` does.
- Build a complete ML workflow: train/test splits with `stratify`, K-fold cross-validation, preprocessing with scalers/encoders/imputers/`ColumnTransformer`, and pipelines with `set_output`.
- Apply classic classifiers — `LogisticRegression`, `LinearDiscriminantAnalysis`, `KNeighborsClassifier`, `GaussianNB`, `SVC` with different kernels, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`, `AdaBoostClassifier` — plus `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`, and use `decision_function` / `predict_proba`.
- Apply regressors — `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SVR`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor` — and read their metrics.
- Evaluate models with accuracy, precision, recall, F1, confusion matrix, `classification_report`, ROC/AUC, log loss, MAE, MSE, RMSE and R²; use `cross_val_score`, `StratifiedKFold`, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV` and learning curves.
- Use unsupervised methods: `KMeans` (inertia, elbow, silhouette), `DBSCAN`, `AgglomerativeClustering`, `PCA` (explained variance, biplot), a `TSNE`/UMAP note, and apriori association rules with mlxtend.
- Handle imbalanced data with `RandomOverSampler` and `SMOTE`, diagnose models with Yellowbrick, tune with Optuna, select and interpret features with `SelectKBest`, `RFE`, `feature_importances_`, `permutation_importance` and SHAP, and persist models with joblib and pickle.
"""))

cells.append(md(r"""
## Official documentation & references

- [scikit-learn — Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Preprocessing and scaling / encoding](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pipelines and composite estimators (`ColumnTransformer`, `set_output`)](https://scikit-learn.org/stable/modules/compose.html)
- [Model evaluation and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [The `Pipeline` class reference](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [CatBoost documentation](https://catboost.ai/docs/)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [mlxtend documentation](https://rasbt.github.io/mlxtend/)
- [Yellowbrick documentation](https://www.scikit-yb.org/en/latest/)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [SHAP documentation](https://shap.readthedocs.io/)
"""))

cells.append(md(r"""
## 1. The scikit-learn API philosophy

Almost everything in scikit-learn follows three rules:

1. **Estimators share one interface.** Any learning object implements `fit(X, y)` (or just `fit(X)` for unsupervised learning). Supervised estimators add `predict(X)`; transformers add `transform(X)` and `fit_transform(X)`. This uniformity means the same `GridSearchCV`, pipeline or cross-validation loop works with dozens of different models.
2. **Configuration and learned state are separate.** Constructor arguments (`max_iter`, `n_neighbors`, ...) are hyperparameters, readable with `get_params()` and changeable with `set_params()`. Anything *learned from data* is stored in an attribute with a trailing underscore, such as `coef_`, `classes_`, `inertia_` or `feature_importances_`. Fitted attributes do not exist before `fit`.
3. **Data is array-like.** `X` is 2-D (`n_samples, n_features`), `y` is 1-D. pandas DataFrames work and preserve column names through `set_output`.

A **Bunch** is sklearn's dataset container — a dict-like object with `.data`, `.target`, `.feature_names`, `.target_names`, `.DESCR` and (for some datasets) `.images`. You can also unpack it with `load_iris(return_X_y=True)`.

Note the difference between `decision_function` (raw scores, one per class for many linear/SVM models), `predict` (hard class labels) and `predict_proba` (calibrated-ish probabilities, one column per class, only available on estimators that support it).
"""))

cells.append(code(r"""
from pathlib import Path
import pickle
import tempfile
import warnings

warnings.filterwarnings("ignore")  # keep teaching output readable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.stats import loguniform

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes,
    make_classification, make_regression, make_blobs, make_moons, fetch_openml,
)
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV, learning_curve,
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder,
    OrdinalEncoder, LabelEncoder, KBinsDiscretizer,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib

# Heavier / optional libraries are guarded so this notebook still runs if one is absent.
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
# scipy/sklearn set DeprecationWarning to "always" while importing, which
# overrides the filters registered above. Re-apply these before importing
# Yellowbrick so its internal deprecations stay quiet.
warnings.filterwarnings("ignore", message="distutils Version classes are deprecated")
warnings.filterwarnings("ignore", message="Passing 'N' to ListedColormap is deprecated")
try:
    from yellowbrick.classifier import ConfusionMatrix as YellowbrickConfusionMatrix
except ImportError:
    YellowbrickConfusionMatrix = None
try:
    import optuna
except ImportError:
    optuna = None
try:
    import shap
except ImportError:
    shap = None

sns.set_theme(style="whitegrid")

DATA = Path("../../data")

def load_csv(name, loader=None):
    # Read a local CSV when present; otherwise build the same table from a sklearn loader.
    path = DATA / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    if loader is None:
        raise FileNotFoundError(f"Missing {path} and no fallback loader was given")
    bunch = loader()
    frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    frame["target"] = bunch.target
    return frame

print("scikit-learn", sklearn.__version__)
print("numpy", np.__version__, "| pandas", pd.__version__)
"""))

cells.append(code(r"""
# A Bunch and the estimator contract in action.
iris = load_iris()
print("type:", type(iris).__name__)
print("keys:", list(iris.keys()))
print("DESCR (first 240 chars):", iris.DESCR[:240].replace("\n", " "))
X, y = iris.data, iris.target
print("X:", X.shape, "y:", y.shape, "class counts:", np.bincount(y))
print("feature_names:", iris.feature_names)
print("target_names:", iris.target_names)

clf = LogisticRegression(max_iter=200)
print("coef_ before fit?", hasattr(clf, "coef_"))
clf.fit(X, y)
print("coef_ after fit:", clf.coef_.shape, "classes_:", clf.classes_)
print("get_params()['max_iter']:", clf.get_params()["max_iter"])
clf.set_params(max_iter=500)
print("predict:", clf.predict(X[:3]))
print("predict_proba:\n", clf.predict_proba(X[:3]).round(3))
print("decision_function:\n", clf.decision_function(X[:3]).round(3))

scaler = StandardScaler()
print("fit_transform (first row):", scaler.fit_transform(X[:3])[0].round(3))
print("transform (same data):    ", scaler.transform(X[:3])[0].round(3))
"""))

cells.append(md(r"""
## 2. Datasets: loaders, local CSVs and generators

The five bundled loaders return a `Bunch`: `load_iris` (150x4, 3 classes), `load_wine` (178x13, 3 classes), `load_breast_cancer` (569x30, binary), `load_digits` (1797x64, 10 classes) and `load_diabetes` (442x10, continuous target). Each has `return_X_y=True` for a plain `(X, y)` tuple. `load_digits` also carries `.images` for plotting the handwritten digits.

`fetch_openml(name="mnist_784", version=1, as_frame=True)` downloads real datasets from OpenML and needs an internet connection — we only mention it here so you know it exists; nothing in this notebook fetches from the web.

Generators create controlled synthetic data:
- `make_classification(n_samples, n_features, n_informative, n_redundant, n_classes, weights, flip_y, random_state)` — separable classification problems, with `weights` for imbalance.
- `make_regression(n_samples, n_features, noise, coef=True, random_state)` — linear target with Gaussian noise.
- `make_blobs(n_samples, centers, n_features, cluster_std, random_state)` — Gaussian blobs for clustering.
- `make_moons(n_samples, noise, random_state)` — two interleaving half-circles, a classic non-linear benchmark.

Our `load_csv` helper prefers the local CSV and falls back to the loader, so the notebook runs even before the CSVs are copied into `../../data`.
"""))

cells.append(code(r"""
loaders = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": load_diabetes,
}
summary = []
for name, loader in loaders.items():
    bunch = loader()
    target_kind = "continuous" if name == "diabetes" else f"{len(np.unique(bunch.target))} classes"
    summary.append((name, bunch.data.shape, target_kind))
print(pd.DataFrame(summary, columns=["loader", "X shape", "target"]).to_string(index=False))

for name, loader in loaders.items():
    frame = load_csv(name, loader)
    print(f"\n{name}.csv -> shape {frame.shape}, columns: {list(frame.columns)[:6]} ...")
    print(frame.head(2).to_string(index=False))

# Synthetic generators.
Xc, yc = make_classification(n_samples=120, n_features=6, n_informative=3,
                             n_classes=2, weights=[0.85, 0.15], random_state=0)
Xr, yr = make_regression(n_samples=100, n_features=3, noise=8.0, random_state=0)
Xb, yb = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=0.8, random_state=0)
Xmo, ymo = make_moons(n_samples=150, noise=0.12, random_state=0)
for label, arr, tgt in [("make_classification", Xc, yc), ("make_regression", Xr, yr),
                        ("make_blobs", Xb, yb), ("make_moons", Xmo, ymo)]:
    print(f"{label:20s} X={arr.shape} y range=({tgt.min():.1f}, {tgt.max():.1f})")
print("imbalanced class counts:", np.bincount(yc))
"""))

cells.append(md(r"""
## 3. The complete ML workflow and train/test splits

The standard workflow is: **define the question → collect/load data → split → preprocess → train → evaluate → tune → deploy**. The split comes first: hold out a test set that is touched exactly once at the end.

`train_test_split(*arrays, test_size=0.25, random_state=42, shuffle=True, stratify=y)` returns train/test parts in the same order as the inputs. `stratify=y` keeps class proportions identical in both halves — essential for imbalanced classification. `random_state` makes the split reproducible.

The same rule applies inside cross-validation: scaling, imputation, resampling and feature selection must be fitted **only on the training fold**. Putting them in a `Pipeline` guarantees this and removes a common source of data leakage.
"""))

cells.append(code(r"""
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print("train:", X_train.shape, "test:", X_test.shape)
print("train class counts:", np.bincount(y_train))
print("test  class counts:", np.bincount(y_test))

# For a regression target there is nothing to stratify.
Xr_d, yr_d = load_diabetes(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr_d, yr_d, test_size=0.25, random_state=0
)
print("regression train/test:", Xr_train.shape, Xr_test.shape)
"""))

cells.append(md(r"""
## 4. Cross-validation splits

A single split is noisy. Cross-validation repeats it over several folds:

- `KFold(n_splits=5, shuffle=True, random_state=0)` — general split, use for regression.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` — preserves class ratios, use for classification.
- `LeaveOneOut()`, `ShuffleSplit(...)`, `GroupKFold(...)` — special cases (tiny data, repeated random splits, grouped observations).

`cross_val_score(estimator, X, y, cv=5, scoring="accuracy")` clones the estimator, fits each fold and returns one score per fold. It never reuses the held-out test set. `cross_validate` is the richer version: multiple metrics, train scores and timings.
"""))

cells.append(code(r"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"fold {fold}: train={len(train_idx)}, val={len(val_idx)}")

scores = cross_val_score(
    LogisticRegression(max_iter=200), X, y, cv=skf, scoring="accuracy"
)
print("per-fold accuracy:", scores.round(3))
print("mean +/- std:", round(scores.mean(), 3), round(scores.std(), 3))

print("a plain KFold validation slice:", list(KFold(n_splits=3).split(X))[0][1].shape)
"""))

cells.append(md(r"""
## 5. Preprocessing

Most models need features on comparable scales or in numeric form. The transformers all share `fit` / `transform` / `fit_transform`:

- `StandardScaler` — subtract mean, divide by standard deviation (z-scores). Default choice for linear models, SVM and PCA.
- `MinMaxScaler(feature_range=(0, 1))` — squashes each feature into a fixed range; useful for image-like bounded data.
- `RobustScaler` — uses median and interquartile range, so outliers barely move it.
- `Normalizer(norm="l2")` — rescales each **sample** (row) to unit norm, not each feature. Common with text vectors.
- `OneHotEncoder(sparse_output=False, handle_unknown="ignore")` — binary columns for nominal categories; `set_output(transform="pandas")` names them. `handle_unknown="ignore"` prevents crashes on unseen test categories.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` — integer codes for categories with a meaningful order (or for tree models).
- `LabelEncoder` — encodes a 1-D **target** vector `y`, not a feature matrix.
- `SimpleImputer(strategy="mean" | "median" | "most_frequent" | "constant", fill_value=...)` — fills missing values.
- `KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="quantile")` — turns a continuous feature into bins (strategies: `uniform`, `quantile`, `kmeans`).
- `PolynomialFeatures(degree=2, include_bias=False)` — adds interactions and powers; pair it with a linear model for polynomial regression.
- `ColumnTransformer(transformers=[...], remainder="drop" | "passthrough")` applies different transformers to different columns and is the standard way to handle mixed numeric/categorical tables.

`set_output(transform="pandas")` is available on most transformers and on `Pipeline`/`ColumnTransformer` since sklearn 1.2; it makes outputs DataFrames with readable column names.
"""))

cells.append(code(r"""
sample = np.array([[1., 2.], [3., -1.], [5., 4.]])
for name, obj in [("StandardScaler", StandardScaler()),
                  ("MinMaxScaler", MinMaxScaler()),
                  ("RobustScaler", RobustScaler()),
                  ("Normalizer", Normalizer())]:
    print(f"{name:16s}", obj.fit_transform(sample).round(3).tolist())

cats = np.array([["red", "small"], ["blue", "large"],
                 ["red", "large"], ["green", "small"]])
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
print("\nOneHotEncoder:\n", ohe.fit_transform(cats).to_string(index=False))
ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
print("OrdinalEncoder:\n", ordinal.fit_transform(cats))
label = LabelEncoder()
print("LabelEncoder:", label.fit_transform(cats[:, 0]), "classes:", label.classes_)

with_nan = np.array([[1., 2.], [np.nan, 4.], [7., np.nan], [5., 6.]])
print("\nmean impute:\n", SimpleImputer(strategy="mean").fit_transform(with_nan).round(2))
print("median impute:\n", SimpleImputer(strategy="median").fit_transform(with_nan).round(2))
age = np.array([[18.], [25.], [39.], [55.], [70.]])
print("3 quantile bins:\n", KBinsDiscretizer(n_bins=3, encode="onehot-dense",
                                             strategy="quantile").fit_transform(age))
print("degree-2 polynomial:\n",
      PolynomialFeatures(degree=2, include_bias=False).fit_transform(np.array([[1., 2.], [3., 4.]])))
"""))

cells.append(code(r"""
mixed = pd.DataFrame({
    "age": [22, 35, 58, 41, 30],
    "income": [2500, 5500, 9000, np.nan, 4000],
    "city": ["Pune", "Delhi", "Pune", "Mumbai", "Delhi"],
})
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), ["age", "income"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ],
    remainder="drop",
).set_output(transform="pandas")
print(preprocess.fit_transform(mixed).round(2).to_string(index=False))
"""))

cells.append(md(r"""
## 6. Pipelines

`Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])` chains transformers and a final estimator. Calling `fit` fits every step in order; calling `predict` runs every `transform` then `predict`. Benefits:

- no leakage — the scaler sees only the training fold during cross-validation;
- one object to `fit`, `score`, `pickle` and grid-search;
- hyperparameters addressed with `step__parameter`, e.g. `clf__C` or `scale__with_mean`;
- `pipeline[:-1]` can drop the last step, `pipeline.named_steps["clf"]` reaches a step, and `set_output` works on the whole pipeline.

`sklearn.pipeline.make_pipeline(*steps)` auto-generates step names from class names when you do not want to name them.
"""))

cells.append(code(r"""
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200)),
])
pipe.fit(X_train, y_train)
print("pipeline accuracy:", round(pipe.score(X_test, y_test), 3))
print("classifier coefficients shape:", pipe.named_steps["clf"].coef_.shape)
pipe.set_params(clf__C=10.0).fit(X_train, y_train)
print("after clf__C=10:", round(pipe.score(X_test, y_test), 3))
print("searchable params:", [k for k in pipe.get_params() if k.startswith("clf__")])
"""))

cells.append(md(r"""
## 7. Classification models

For a quick baseline, try `LogisticRegression` first. Other linear/discriminant options: `LinearDiscriminantAnalysis` (Gaussian class densities, shared covariance), `KNeighborsClassifier` (`n_neighbors`, `weights="uniform"|"distance"`), `GaussianNB` (naive Bayes, fast, assumes feature independence), and `SVC`.

`SVC` kernels: `kernel="linear"` (flat separating hyperplane), `"rbf"` (default, with `gamma="scale"`), `"poly"` (with `degree`), `"sigmoid"`. `C` controls the margin/penalty trade-off; `decision_function_shape="ovr"` gives one-vs-rest scores for multiclass. By default an SVC has **no** `predict_proba`; set `probability=True` to get it (this adds internal cross-validation and costs time).
"""))

cells.append(code(r"""
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
print("classes:", lr.classes_)
print("coef_:", lr.coef_.shape, "intercept_:", lr.intercept_.round(3))
print("decision_function[0]:\n", lr.decision_function(X_test[:1]).round(3))
print("predict_proba[0]:\n", lr.predict_proba(X_test[:1]).round(3))
print("predict[0]:", lr.predict(X_test[:1]))
"""))

cells.append(code(r"""
classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB(),
    "SVC-linear": SVC(kernel="linear", decision_function_shape="ovr"),
    "SVC-rbf": SVC(kernel="rbf", gamma="scale", probability=True, random_state=0),
    "SVC-poly": SVC(kernel="poly", degree=3, gamma="scale"),
}
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    line = f"{name:12s} accuracy={model.score(X_test, y_test):.3f}"
    if hasattr(model, "predict_proba"):
        line += "  proba[0]=" + str(model.predict_proba(X_test[:1])[0].round(3))
    else:
        line += "  decision[0]=" + str(model.decision_function(X_test[:1])[0].round(3))
    print(line)
"""))

cells.append(md(r"""
## 8. Trees and ensemble methods

A `DecisionTreeClassifier` learns axis-aligned splits; `max_depth`, `min_samples_leaf` and `criterion="gini"|"entropy"` control overfitting. Ensembles combine many trees:

- **Bagging**: `RandomForestClassifier` (bootstrap samples + random feature subsets) and `ExtraTreesClassifier` (also random split thresholds). Key parameters: `n_estimators`, `max_features`, `max_depth`, `n_jobs=-1`.
- **Boosting**: `GradientBoostingClassifier` fits weak learners to residuals (`n_estimators`, `learning_rate`, `max_depth`); `HistGradientBoostingClassifier` bins features into histograms and is much faster on larger data (`max_iter`, `max_leaf_nodes`, `learning_rate`, `l2_regularization`); `AdaBoostClassifier` reweights misclassified samples each round.

Since sklearn 1.6, `AdaBoostClassifier` only implements SAMME and its `algorithm` parameter is deprecated (removed in 1.8), so we simply omit that parameter. Tree-based models expose `feature_importances_`, which sums the impurity decrease each feature produces.
"""))

cells.append(code(r"""
tree_models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4,
                                           random_state=0, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=4,
                                       random_state=0, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                                   random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=50,
                                                           max_leaf_nodes=7, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=0),
}
for name, model in tree_models.items():
    model.fit(X_train, y_train)
    note = ""
    if hasattr(model, "feature_importances_"):
        note = "  top feature index=" + str(int(np.argmax(model.feature_importances_)))
    print(f"{name:22s} accuracy={model.score(X_test, y_test):.3f}{note}")
"""))

cells.append(md(r"""
## 9. Third-party gradient boosters

XGBoost, LightGBM and CatBoost are often stronger than sklearn's boosters on tabular data. They implement the same `fit`/`predict`/`predict_proba` API and can be dropped into pipelines.

- `XGBClassifier(n_estimators, max_depth, learning_rate, eval_metric="logloss", tree_method="hist", random_state)` — `eval_metric` avoids a warning and `tree_method="hist"` is the fast histogram algorithm.
- `LGBMClassifier(n_estimators, num_leaves, max_depth, learning_rate, verbose=-1, random_state)` — leaf-wise growth; `verbose=-1` silences info logs.
- `CatBoostClassifier(iterations, depth, learning_rate, verbose=False, random_state)` — strong handling of categorical features; we pass NumPy arrays so feature names are not an issue here.

They are imported in a `try/except` block at the top so the cell reports gracefully if one is unavailable.
"""))

cells.append(code(r"""
boosters = {}
if XGBClassifier is not None:
    boosters["XGBoost"] = XGBClassifier(
        n_estimators=50, max_depth=3, eval_metric="logloss", tree_method="hist", random_state=0)
if LGBMClassifier is not None:
    boosters["LightGBM"] = LGBMClassifier(
        n_estimators=50, max_depth=3, verbose=-1, random_state=0)
if CatBoostClassifier is not None:
    boosters["CatBoost"] = CatBoostClassifier(
        iterations=50, depth=3, verbose=False, random_state=0)

for name, model in boosters.items():
    model.fit(X_train, y_train)
    print(f"{name:10s} accuracy={model.score(X_test, y_test):.3f} "
          f"proba[0]={model.predict_proba(X_test[:1])[0].round(3)}")
if not boosters:
    print("No optional boosting libraries installed — the sklearn boosters above still work.")
"""))

cells.append(md(r"""
## 10. Regression models

Regression predicts a continuous target. `LinearRegression` fits least-squares weights. Regularized variants shrink coefficients: `Ridge` (L2, `alpha`), `Lasso` (L1, `alpha`, can zero out features), `ElasticNet` (both, with `l1_ratio`). `SVR` is the SVM analogue with `epsilon` (insensitive tube width) and `C`. `KNeighborsRegressor` averages nearby targets. `DecisionTreeRegressor`, `RandomForestRegressor` and `GradientBoostingRegressor` are the tree-based alternatives.

Scale features for ridge/lasso/SVR; the diabetes dataset is used here because it is small and built in. For a regressor, `model.score(X, y)` returns the R² coefficient of determination.
"""))

cells.append(code(r"""
regressors = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=1)": Ridge(alpha=1.0),
    "Lasso(alpha=0.1)": Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "SVR": SVR(kernel="rbf", C=10, epsilon=1.0),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=4, random_state=0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=5,
                                                   random_state=0, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                                           random_state=0),
}
for name, model in regressors.items():
    model.fit(Xr_train, yr_train)
    print(f"{name:26s} R2={model.score(Xr_test, yr_test):.3f}")
"""))

cells.append(md(r"""
## 11. Evaluation metrics

**Classification.** Accuracy = correct / total, but it lies when classes are imbalanced. Precision = TP / (TP + FP) ("of those flagged, how many are right?"), recall = TP / (TP + FN) ("of the real positives, how many did we find?"), and F1 is their harmonic mean. Use `average="macro"` to weight every class equally, `"weighted"` to weight by support, or `"binary"` (default) for the positive class. `confusion_matrix` tabulates actual vs predicted counts; `classification_report` prints precision/recall/F1/support per class. `roc_curve` gives (FPR, TPR, thresholds) from scores or probabilities, and `auc`/`roc_auc_score` integrate that curve. `log_loss` evaluates probability quality (lower is better) and needs `predict_proba`.

**Regression.** MAE is mean absolute error (same units as the target, robust to outliers); MSE squares errors (penalizes large misses); RMSE = √MSE is back in the original units; R² is the fraction of variance explained (1 is perfect, 0 matches a constant predictor).
"""))

cells.append(code(r"""
Xbin, ybin = make_classification(n_samples=300, n_features=6, n_informative=4,
                                 n_classes=2, weights=[0.8, 0.2], random_state=0)
Xbin_tr, Xbin_te, ybin_tr, ybin_te = train_test_split(
    Xbin, ybin, test_size=0.3, stratify=ybin, random_state=42)
bin_clf = LogisticRegression(max_iter=500).fit(Xbin_tr, ybin_tr)
pred = bin_clf.predict(Xbin_te)
proba = bin_clf.predict_proba(Xbin_te)[:, 1]

print("accuracy :", round(accuracy_score(ybin_te, pred), 3))
print("precision:", round(precision_score(ybin_te, pred), 3))
print("recall   :", round(recall_score(ybin_te, pred), 3))
print("F1       :", round(f1_score(ybin_te, pred), 3))
print("confusion matrix:\n", confusion_matrix(ybin_te, pred))
print("classification report:\n", classification_report(ybin_te, pred, target_names=["no", "yes"]))
fpr, tpr, thresholds = roc_curve(ybin_te, proba)
print("AUC via curve:", round(auc(fpr, tpr), 3), "| via roc_auc_score:", round(roc_auc_score(ybin_te, proba), 3))
print("log loss:", round(log_loss(ybin_te, bin_clf.predict_proba(Xbin_te), labels=[0, 1]), 3))

plt.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], "k--", label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
plt.show()

ridge = Ridge(alpha=1.0).fit(Xr_train, yr_train)
r_pred = ridge.predict(Xr_test)
mse = mean_squared_error(yr_test, r_pred)
print("regression -> MAE:", round(mean_absolute_error(yr_test, r_pred), 2),
      "MSE:", round(mse, 2), "RMSE:", round(np.sqrt(mse), 2),
      "R2:", round(r2_score(yr_test, r_pred), 3))
"""))

cells.append(md(r"""
## 12. Model selection and learning curves

`cross_validate(estimator, X, y, cv=..., scoring=["accuracy", "f1_macro"], return_train_score=True)` returns a dict of arrays — handy for comparing several metrics at once.

Hyperparameter search:
- `GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1)` tries every combination exhaustively.
- `RandomizedSearchCV(estimator, param_distributions, n_iter=10, cv=5, random_state=0)` samples from distributions such as `scipy.stats.loguniform` — far cheaper for large spaces.

Both clone the estimator for each fold, expose `best_params_`, `best_score_`, `best_estimator_`, `cv_results_`, and (with `refit=True`, the default) re-fit on the full training set so you can call `.predict` / `.score` directly. Always evaluate the winner on the untouched test set once.

`learning_curve(estimator, X, y, cv=3, train_sizes=...)` returns training sizes plus train and validation scores. If the two curves stay far apart, add data or regularize less; if both are low and close, the model underfits.
"""))

cells.append(code(r"""
res = cross_validate(
    LogisticRegression(max_iter=200), X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    scoring=["accuracy", "f1_macro"], return_train_score=True,
)
print(pd.DataFrame(res)[["train_accuracy", "test_accuracy", "test_f1_macro"]].round(3).to_string(index=False))

grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}
search = GridSearchCV(pipe, grid, cv=3, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
print("grid best:", search.best_params_, "CV accuracy:", round(search.best_score_, 3),
      "| test accuracy:", round(search.score(X_test, y_test), 3))

rand = RandomizedSearchCV(
    pipe, {"clf__C": loguniform(0.01, 100)}, n_iter=6, cv=3, random_state=0, n_jobs=-1)
rand.fit(X_train, y_train)
print("random best:", {k: round(v, 3) for k, v in rand.best_params_.items()},
      "CV accuracy:", round(rand.best_score_, 3))
"""))

cells.append(code(r"""
sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=200), X, y, cv=3,
    train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy")
plt.plot(sizes, train_scores.mean(axis=1), "o-", label="training")
plt.plot(sizes, val_scores.mean(axis=1), "o-", label="validation")
plt.xlabel("Number of training examples"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Learning curve"); plt.show()
"""))

cells.append(md(r"""
## 13. Unsupervised learning — clustering

`KMeans(n_clusters=3, n_init=10, random_state=0)` assigns each point to the nearest of `k` centroids. `.inertia_` is the within-cluster sum of squares; it always falls as `k` grows, so the **elbow plot** looks for where the drop slows. The **silhouette score** (range -1 to 1, higher is better) measures how well-separated the clusters are and can be computed for each `k` with `silhouette_score(X, labels)`. `.labels_` holds the cluster ids.

`DBSCAN(eps=0.5, min_samples=5)` is density-based: it finds arbitrarily shaped clusters and marks outliers as label `-1`, but it needs `eps` tuning and does not scale as well. `AgglomerativeClustering(n_clusters=3, linkage="ward")` builds a hierarchy by repeatedly merging the closest clusters (linkage options: `ward`, `complete`, `average`, `single`); unlike KMeans it needs no `n_init`.
"""))

cells.append(code(r"""
X_blobs, y_blobs = make_blobs(n_samples=200, centers=3, n_features=2,
                              cluster_std=0.7, random_state=0)
inertias, silhouettes = [], []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_blobs)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_blobs, km.labels_))
print(pd.DataFrame({"k": range(2, 7),
                    "inertia": np.round(inertias, 1),
                    "silhouette": np.round(silhouettes, 3)}).to_string(index=False))

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_blobs)
print("KMeans labels:", np.bincount(km.labels_), "inertia:", round(km.inertia_, 2))
db = DBSCAN(eps=0.6, min_samples=5).fit(X_blobs)
print("DBSCAN clusters:", len(set(db.labels_) - {-1}),
      "| noise points:", int((db.labels_ == -1).sum()))
ag = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X_blobs)
print("Agglomerative labels:", np.bincount(ag.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(range(2, 7), inertias, "o-"); ax[0].set(xlabel="k", ylabel="inertia", title="Elbow")
ax[1].plot(range(2, 7), silhouettes, "o-"); ax[1].set(xlabel="k", ylabel="silhouette")
plt.show()
"""))

cells.append(md(r"""
## 14. Dimensionality reduction — PCA and friends

`PCA(n_components=2)` finds orthogonal directions of maximum variance. After `fit_transform`, `.explained_variance_ratio_` tells you how much variance each component captures, `.components_` holds the loadings (original-feature weights), and `.singular_values_` / `.mean_` describe the fit. `whiten=True` rescales components to unit variance. A **biplot** overlays the projected samples and arrows for the original features, showing which features drive each axis.

For non-linear structure, `TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0)` produces a 2-D visualization — it is for display, not for feeding a downstream model (it has no `transform` for new data). **UMAP** (umap-learn) is a faster, structure-preserving alternative; it is not installed here, so we only note it.
"""))

cells.append(code(r"""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("components shape:", pca.components_.shape)
print(pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"],
                   index=iris.feature_names).round(3).to_string())

for target, name in enumerate(iris.target_names):
    mask = y == target
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, s=18)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.legend(); plt.title("Iris projected onto 2 PCs"); plt.show()

# Biplot: draw feature arrows on the same projected space.
loadings = pca.components_.T * np.sqrt(pca.explained_variance_ratio_)
for feature_idx, feature in enumerate(iris.feature_names):
    dx, dy = loadings[feature_idx, 0] * 6, loadings[feature_idx, 1] * 6
    plt.arrow(0, 0, dx, dy, alpha=0.5)
    plt.text(dx * 1.05, dy * 1.05, feature, fontsize=7)
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA biplot"); plt.show()

tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0)
print("t-SNE output shape:", tsne.fit_transform(X).shape)
"""))

cells.append(md(r"""
## 15. Association rules with mlxtend

Apriori finds frequent itemsets in transaction data (baskets of items). `TransactionEncoder().fit_transform(transactions)` produces a boolean one-hot matrix; `apriori(basket, min_support=0.4, use_colnames=True)` returns itemsets with a `support` column; `association_rules(...)` derives rules with metrics such as **support** (how often the whole rule occurs), **confidence** (how often the consequent appears given the antecedent) and **lift** (confidence divided by the consequent's baseline rate; >1 means a real association).

Version note: mlxtend 0.23.x briefly made `num_itemsets` a required argument of `association_rules`. We pass it explicitly and fall back to the older call on a `TypeError`, so the code works across versions.
"""))

cells.append(code(r"""
transactions = [
    ["milk", "bread", "butter"],
    ["milk", "bread"],
    ["milk", "butter", "cheese"],
    ["bread", "butter", "cheese"],
    ["milk", "bread", "butter", "cheese"],
    ["bread", "cheese"],
]
encoder = TransactionEncoder()
basket = pd.DataFrame(encoder.fit_transform(transactions), columns=encoder.columns_)
print(basket.head(3).to_string(index=False))

frequent = apriori(basket, min_support=0.4, use_colnames=True).sort_values("support", ascending=False)
print("\nFrequent itemsets:\n", frequent.head(8).to_string(index=False))
try:
    rules = association_rules(frequent, metric="confidence", min_threshold=0.6,
                              num_itemsets=len(basket))
except TypeError:
    rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
print("\nRules:\n", rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .head(8).to_string(index=False))
"""))

cells.append(md(r"""
## 16. Imbalanced data with imbalanced-learn

With a 95/5 class split, predicting the majority class gives 95% accuracy and zero recall on the class you care about. Two fixes:

- `RandomOverSampler(random_state=0)` duplicates minority rows.
- `SMOTE(random_state=0, k_neighbors=5)` instead **interpolates** new minority points between nearest neighbours — usually better than exact duplicates. `k_neighbors` must be smaller than the minority class count.

Usage is the familiar `fit_resample(X, y)`. Crucially, resample **only the training fold**: use `imblearn.pipeline.Pipeline` with a sampler as a step so cross-validation applies it inside each fold. (Plain sklearn `Pipeline` does not run `fit_resample` steps.)
"""))

cells.append(code(r"""
print("before:", np.bincount(ybin_tr))
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(Xbin_tr, ybin_tr)
print("random oversampling:", np.bincount(y_ros))
smote = SMOTE(random_state=0, k_neighbors=3)
X_sm, y_sm = smote.fit_resample(Xbin_tr, ybin_tr)
print("SMOTE:", np.bincount(y_sm))

imb_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=0, k_neighbors=3)),
    ("clf", LogisticRegression(max_iter=500)),
])
imb_pipe.fit(Xbin_tr, ybin_tr)
print("resampling pipeline test accuracy:", round(imb_pipe.score(Xbin_te, ybin_te), 3))
"""))

cells.append(md(r"""
## 17. Visual diagnostics with Yellowbrick

Yellowbrick visualizers are scikit-learn-style estimators that draw a plot while they fit/score. `ConfusionMatrix(model, force_model=True)` wraps a classifier (the flag works around Yellowbrick 1.5's outdated sklearn tag check): `fit(X_train, y_train)`, `score(X_test, y_test)`, then `show()` renders the heatmap (older docs use the now-deprecated `poof()`). Other useful ones: `ROCAUC`, `PrecisionRecallCurve`, `ClassificationReport`, `ClassBalance` (in `yellowbrick.target`), and `yellowbrick.model_selection.LearningCurve` / `ValidationCurve`. They integrate with pipelines and can be grid-searched like any estimator.
"""))

cells.append(code(r"""
if YellowbrickConfusionMatrix is not None:
    # Yellowbrick 1.5's classifier check predates scikit-learn 1.9's tag API;
    # force_model bypasses that check, the estimator itself is a classifier.
    viz = YellowbrickConfusionMatrix(LogisticRegression(max_iter=200), force_model=True)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    print("visualizer score:", round(viz.score_, 3))
    viz.show()
else:
    print("yellowbrick is not installed")
"""))

cells.append(md(r"""
## 18. Hyperparameter optimization with Optuna

Optuna is a modern tuning library built around a **study** that runs many **trials**. Each trial calls an objective function and uses `trial.suggest_int`, `suggest_float`, `suggest_categorical`, `suggest_loguniform` etc. to propose hyperparameters. `study.optimize(objective, n_trials=20)` explores the space; `study.best_params` and `study.best_value` give the winner. Optuna supports pruning unpromising trials and much richer search spaces than a plain grid. Here we keep it to a few trials with cross-validated accuracy as the objective.
"""))

cells.append(code(r"""
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 15)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        return cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    print("best parameters:", study.best_params)
    print("best CV accuracy:", round(study.best_value, 3))
else:
    print("optuna is not installed")
"""))

cells.append(md(r"""
## 19. Feature selection and model interpretation

- **Filter**: `SelectKBest(score_func=f_classif, k=2)` ranks each feature by a univariate statistical test; `.scores_` and `.get_support()` expose the ranking. (Regression counterparts: `f_regression`, `mutual_info_regression`.)
- **Wrapper**: `RFE(estimator=..., n_features_to_select=2)` recursively drops the weakest features; `.ranking_` and `.support_` show the result. `RFECV` chooses the number of features by cross-validation.
- **Embedded**: tree ensembles expose `.feature_importances_`; linear models expose `.coef_`.
- **Model-agnostic**: `permutation_importance(model, X_test, y_test, n_repeats=5, scoring="accuracy", random_state=0)` shuffles one feature at a time on held-out data and measures the score drop; `.importances_mean` and `.importances_std` summarize it.
- **SHAP** assigns each feature a contribution to each individual prediction using cooperative game theory. `shap.TreeExplainer(model).shap_values(X)` works well with tree models, and `shap.summary_plot` shows a global importance/effect plot. Be aware that the SHAP API has changed across versions; the call below is guarded.
"""))

cells.append(code(r"""
selector = SelectKBest(score_func=f_classif, k=2).fit(X, y)
print("SelectKBest support:", selector.get_support(), "scores:", selector.scores_.round(1))
print("reduced shape:", selector.transform(X).shape)

rfe = RFE(estimator=LogisticRegression(max_iter=500), n_features_to_select=2).fit(X, y)
print("RFE support:", rfe.support_, "ranking:", rfe.ranking_)

rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
print("feature_importances_:", rf.feature_importances_.round(3))

perm = permutation_importance(rf, X_test, y_test, n_repeats=5,
                              random_state=0, scoring="accuracy")
print("permutation mean:", perm.importances_mean.round(3))
print("permutation std :", perm.importances_std.round(3))
"""))

cells.append(code(r"""
if shap is not None:
    X_shap, y_shap = make_classification(n_samples=150, n_features=5,
                                         n_informative=3, random_state=0)
    shap_model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_shap, y_shap)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap[:40])
    shap.summary_plot(shap_values, X_shap[:40], plot_type="bar", show=False)
    plt.show()
else:
    print("shap is not installed")
"""))

cells.append(md(r"""
## 20. Model persistence

`joblib.dump(model, path)` / `joblib.load(path)` is the recommended way to save sklearn models — it stores large NumPy arrays much more efficiently than plain pickle. `pickle.dump` / `pickle.load` from the standard library also works. Save the **whole pipeline** (preprocessing + model) so the deployed object applies the exact same transformations. Never load a pickle from an untrusted source (pickle executes arbitrary code), and expect breakage if library versions change between save and load.
"""))

cells.append(code(r"""
with tempfile.TemporaryDirectory() as tmp:
    joblib_path = Path(tmp) / "model.joblib"
    pickle_path = Path(tmp) / "model.pkl"
    joblib.dump(pipe, joblib_path)
    with open(pickle_path, "wb") as fh:
        pickle.dump(pipe, fh)

    from_joblib = joblib.load(joblib_path)
    with open(pickle_path, "rb") as fh:
        from_pickle = pickle.load(fh)

    expected = pipe.predict(X_test)
    print("joblib round-trip matches:", np.array_equal(from_joblib.predict(X_test), expected))
    print("pickle round-trip matches:", np.array_equal(from_pickle.predict(X_test), expected))
    print("saved file sizes:", joblib_path.stat().st_size, pickle_path.stat().st_size, "bytes")
"""))

cells.append(md(r"""
## Try it yourself #1

Build a `Pipeline` that standardizes features and then fits an `SVC(kernel="rbf")` on the **wine** dataset (use `load_wine(return_X_y=True)`). Evaluate it with `cross_val_score` using `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` and `scoring="accuracy"`. Print the five fold scores and their mean. Then fit the pipeline on the whole dataset and confirm it has a `predict_proba` method only if you set `probability=True` — check both cases.
"""))

cells.append(code(r"""
# Solution
X_wine, y_wine = load_wine(return_X_y=True)
wine_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, random_state=0)),
])
wine_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
wine_scores = cross_val_score(wine_pipe, X_wine, y_wine, cv=wine_cv, scoring="accuracy")
print("fold scores:", wine_scores.round(3), "mean:", round(wine_scores.mean(), 3))
wine_pipe.fit(X_wine, y_wine)
print("has predict_proba:", hasattr(wine_pipe, "predict_proba"))
print("probability row:", wine_pipe.predict_proba(X_wine[:1]).round(3))
print("plain SVC has predict_proba:", hasattr(SVC(kernel="rbf"), "predict_proba"))
"""))

cells.append(md(r"""
## Try it yourself #2

Using the **digits** dataset (`load_digits(return_X_y=True)`), split the data with `stratify=y` and `random_state=0`. Run a `GridSearchCV` over a pipeline with `StandardScaler` + `KNeighborsClassifier`, searching `n_neighbors` in `[3, 5, 7]` and `weights` in `["uniform", "distance"]` with `cv=3` and `scoring="accuracy"`. Report `best_params_`, `best_score_` and the held-out test accuracy.
"""))

cells.append(code(r"""
# Solution
X_dig, y_dig = load_digits(return_X_y=True)
dig_tr, dig_te, ydig_tr, ydig_te = train_test_split(
    X_dig, y_dig, test_size=0.25, stratify=y_dig, random_state=0)
dig_pipe = Pipeline([("scale", StandardScaler()),
                     ("knn", KNeighborsClassifier())])
dig_grid = GridSearchCV(
    dig_pipe,
    {"knn__n_neighbors": [3, 5, 7], "knn__weights": ["uniform", "distance"]},
    cv=3, scoring="accuracy", n_jobs=-1,
)
dig_grid.fit(dig_tr, ydig_tr)
print("best params:", dig_grid.best_params_)
print("best CV accuracy:", round(dig_grid.best_score_, 3))
print("test accuracy:", round(dig_grid.score(dig_te, ydig_te), 3))
"""))

cells.append(md(r"""
## Try it yourself #3

Reduce the **digits** data to 2 components with `PCA`, then cluster those 2-D points with `KMeans(n_clusters=10, n_init=10, random_state=0)`. Compute the `silhouette_score` of the clustering and show a `pd.crosstab` of true digit labels versus cluster ids. Remember: PCA and KMeans are both unsupervised, so do **not** pass `y` when fitting them — `y` is only used for the final evaluation.
"""))

cells.append(code(r"""
# Solution
X_2d = PCA(n_components=2, random_state=0).fit_transform(X_dig)
digits_km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X_2d)
cluster_labels = digits_km.labels_
print("2-D shape:", X_2d.shape, "cluster counts:", np.bincount(cluster_labels))
print("silhouette:", round(silhouette_score(X_2d, cluster_labels), 3))
print(pd.crosstab(y_dig, cluster_labels).to_string())
"""))

cells.append(md(r"""
## Cheat sheet / key takeaways

| Task | Idiomatic scikit-learn call |
|---|---|
| Load bundled data | `load_iris(return_X_y=True)`, `load_wine()`, `load_breast_cancer()`, `load_digits()`, `load_diabetes()` |
| Make synthetic data | `make_classification`, `make_regression`, `make_blobs`, `make_moons` (always set `random_state`) |
| Split data | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` |
| Cross-validate | `cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))` |
| Scale / encode | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder(sparse_output=False)`, `OrdinalEncoder`, `LabelEncoder` (for `y` only) |
| Clean / reshape | `SimpleImputer`, `KBinsDiscretizer`, `PolynomialFeatures`, `ColumnTransformer(...).set_output(transform="pandas")` |
| Avoid leakage | wrap every learned transform in `Pipeline([...])` |
| Classify | `LogisticRegression`, `LDA`, `KNeighborsClassifier`, `GaussianNB`, `SVC(kernel=..., probability=True)`, trees/forests/boosters |
| Inspect classifier | `decision_function` (scores), `predict_proba` (probabilities), `predict` (labels) |
| Regress | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SVR`, `KNeighborsRegressor`, tree ensembles |
| Class metrics | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`, `roc_auc_score`, `log_loss` |
| Regression metrics | `mean_absolute_error`, `mean_squared_error`, `np.sqrt(mse)`, `r2_score` |
| Tune | `GridSearchCV`, `RandomizedSearchCV`, `optuna.create_study()` |
| Cluster | `KMeans` (inertia/elbow, `silhouette_score`), `DBSCAN`, `AgglomerativeClustering` |
| Reduce dimensions | `PCA(...).explained_variance_ratio_`, `TSNE(...)` for display only |
| Imbalanced data | `RandomOverSampler`, `SMOTE` inside `imblearn.pipeline.Pipeline` |
| Select / explain | `SelectKBest`, `RFE`, `feature_importances_`, `permutation_importance`, `shap.TreeExplainer` |
| Visual diagnostics | Yellowbrick `ConfusionMatrix`, `ROCAUC`, `LearningCurve` |
| Association rules | mlxtend `TransactionEncoder` → `apriori` → `association_rules` |
| Save / load | `joblib.dump(pipe, path)` / `joblib.load(path)` (prefer over `pickle`) |

**Rules of thumb:** fit preprocessors only on training data (use pipelines); stratify classification splits; pick metrics that match the problem (not plain accuracy for imbalance); tune on CV, report once on the test set; `random_state` everywhere for reproducibility.
"""))

cells.append(md(r"""
## Further reading

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [API reference (all estimators and functions)](https://scikit-learn.org/stable/modules/classes.html)
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Cache memory for pipelines (`Memory`)](https://scikit-learn.org/stable/modules/generated/sklearn.memory.Memory.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [CatBoost Python reference](https://catboost.ai/en/docs/references/python-api-reference)
- [imbalanced-learn pipeline documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
- [mlxtend association rules user guide](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- [Yellowbrick gallery](https://www.scikit-yb.org/en/latest/gallery.html)
- [Optuna tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [SHAP explanation types](https://shap.readthedocs.io/en/latest/explanation_types.html)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [matplotlib documentation](https://matplotlib.org/stable/contents.html)
"""))

write_notebook("notebooks/data-science/unit5_scikit_learn.ipynb", cells)
