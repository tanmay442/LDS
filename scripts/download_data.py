"""Download every sample dataset used by this course into data/.

Usage:
    .venv/bin/python scripts/download_data.py            # all CSV datasets
    .venv/bin/python scripts/download_data.py --nltk     # also download NLTK corpora

Sources:
    - Seaborn example data repository: https://github.com/mwaskom/seaborn-data
    - scikit-learn toy datasets and generators: https://scikit-learn.org/stable/datasets.html
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import ROOT, apply

apply()
DATA = ROOT / "data"

SEABORN_DATASETS = [
    "anagrams", "anscombe", "attention", "brain_networks", "car_crashes",
    "diamonds", "dots", "dowjones", "exercise", "flights", "fmri", "geyser",
    "glue", "healthexp", "iris", "mpg", "penguins", "planets", "seaice",
    "taxis", "tips", "titanic",
]

SEABORN_URLS = [
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/main/{name}.csv",
]

SEABORN_DESCRIPTIONS = {
    "anagrams": "Anagram task reaction times (psychology experiment).",
    "anscombe": "Anscombe's quartet: four datasets with identical statistics, different shapes.",
    "attention": "Attention task data (psych252).",
    "brain_networks": "Functional brain connectivity networks.",
    "car_crashes": "US state car crash fatalities and related factors.",
    "diamonds": "54,000 diamond prices with carat, cut, color, clarity.",
    "dots": "Perceptual decision-making dot-motion data.",
    "dowjones": "Dow Jones index daily closing values.",
    "exercise": "Cardiovascular exercise and diet experiment.",
    "flights": "Monthly airline passenger counts, 1949-1960.",
    "fmri": "fMRI signal timeseries across brain regions and events.",
    "geyser": "Old Faithful geyser eruption duration and waiting time.",
    "glue": "Glue benchmark NLP scores.",
    "healthexp": "Life expectancy vs health expenditure by country.",
    "iris": "Fisher's iris flowers: 150 rows, 3 species, 4 measurements.",
    "mpg": "Fuel economy and car attributes.",
    "penguins": "Palmer Station penguin body measurements and species.",
    "planets": "NASA exoplanet candidate discoveries and properties.",
    "seaice": "Arctic sea ice extent by month and year.",
    "taxis": "NYC taxi trip records.",
    "tips": "Restaurant tips by bill, sex, smoker, day, time, party size.",
    "titanic": "Titanic passenger survival data.",
}


def download_seaborn(name: str) -> bool:
    target = DATA / f"{name}.csv"
    if target.exists() and target.stat().st_size > 0:
        print(f"  [skip] seaborn/{name} (already present)")
        return True
    for url_template in SEABORN_URLS:
        url = url_template.format(name=name)
        try:
            urllib.request.urlretrieve(url, target)
            print(f"  [ok]   seaborn/{name} -> {target.name} ({target.stat().st_size:,} bytes)")
            return True
        except Exception as error:
            last_error = error
    print(f"  [FAIL] seaborn/{name}: {last_error}")
    return False


def save_bunch(name: str, bunch, description: str) -> None:
    import pandas as pd

    target = DATA / f"{name}.csv"
    frame = pd.DataFrame(bunch.data, columns=[str(c) for c in bunch.feature_names])
    targets = pd.DataFrame(bunch.target)
    if targets.shape[1] == 1:
        frame["target"] = targets.iloc[:, 0]
        if getattr(bunch, "target_names", None) is not None:
            frame["target_name"] = [bunch.target_names[int(value)] for value in bunch.target]
    else:
        target_names = getattr(bunch, "target_names", None)
        for column in range(targets.shape[1]):
            label = f"target_{target_names[column]}" if target_names else f"target_{column}"
            frame[label] = targets.iloc[:, column]
    frame.to_csv(target, index=False)
    print(f"  [ok]   sklearn/{name} -> {target.name} ({len(frame)} rows) | {description}")


def save_generated(name: str, generator, description: str, **kwargs) -> None:
    import pandas as pd

    target = DATA / f"{name}.csv"
    features, labels = generator(**kwargs)
    columns = [f"feature_{index}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    frame["target"] = labels
    frame.to_csv(target, index=False)
    print(f"  [ok]   sklearn/{name} -> {target.name} ({len(frame)} rows) | {description}")


def install_sklearn_datasets() -> None:
    from sklearn import datasets

    print("\nscikit-learn toy datasets:")
    save_bunch("sk_iris", datasets.load_iris(), "150 flowers, 3 classes (classification)")
    save_bunch("sk_wine", datasets.load_wine(), "178 wines, 3 cultivars (classification)")
    save_bunch("sk_breast_cancer", datasets.load_breast_cancer(), "569 tumors, 2 classes (classification)")
    save_bunch("sk_diabetes", datasets.load_diabetes(), "442 patients, regression target")
    save_bunch("digits", datasets.load_digits(), "1797 handwritten digit images (64 pixels)")
    save_bunch("linnerud", datasets.load_linnerud(), "20 exercise/physique records (multi-output)")

    print("\nscikit-learn synthetic generators:")
    save_generated("blobs", datasets.make_blobs, "3 Gaussian blobs (clustering)", n_samples=600, centers=3, random_state=42)
    save_generated("moons", datasets.make_moons, "Two interleaving half circles", n_samples=600, noise=0.06, random_state=42)
    save_generated("circles", datasets.make_circles, "Concentric circles", n_samples=600, noise=0.05, factor=0.5, random_state=42)
    save_generated("classification", datasets.make_classification, "Informative classification task", n_samples=1000, n_features=20, n_informative=8, random_state=42)
    save_generated("regression", datasets.make_regression, "Linear regression task", n_samples=1000, n_features=10, noise=12.0, random_state=42)


def install_nltk_data() -> None:
    print("\nNLTK corpora:")
    try:
        import nltk
    except ImportError:
        print("  [skip] nltk is not installed")
        return
    packages = [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "vader_lexicon", "brown", "movie_reviews", "reuters", "gutenberg",
        "maxent_ne_chunker", "maxent_ne_chunker_tab", "universal_tagset", "words",
    ]
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"  [ok]   {package}")
        except Exception as error:
            print(f"  [FAIL] {package}: {error}")


def write_index() -> None:
    import pandas as pd

    rows = []
    for path in sorted(DATA.glob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=5)
            full = pd.read_csv(path, usecols=[0])
            rows.append({
                "file": path.name,
                "rows": len(full),
                "columns": ", ".join(frame.columns),
                "source": "seaborn-data" if path.stem in SEABORN_DATASETS else "scikit-learn",
                "description": SEABORN_DESCRIPTIONS.get(path.stem, "Synthetic or bundled dataset"),
            })
        except Exception as error:
            print(f"  [warn] could not index {path.name}: {error}")
    index = pd.DataFrame(rows)
    index.to_csv(DATA / "dataset_index.csv", index=False)
    print(f"\nWrote {DATA / 'dataset_index.csv'} with {len(index)} datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nltk", action="store_true", help="also download NLTK corpora")
    parser.add_argument("--only-seaborn", action="store_true", help="skip sklearn datasets and the index")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA}")
    print("\nSeaborn example datasets:")
    results = [download_seaborn(name) for name in SEABORN_DATASETS]
    print(f"\nSeaborn: {sum(results)}/{len(results)} available")

    if not args.only_seaborn:
        install_sklearn_datasets()
        write_index()

    if args.nltk:
        install_nltk_data()

    print("\nDone. Start Jupyter with: uv run jupyter lab --notebook-dir .")


if __name__ == "__main__":
    main()
