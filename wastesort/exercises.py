"""Independent regression exercise using scikit-learn's bundled diabetes data."""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .common import ROOT, SEED, save_json


def main():
    data = load_diabetes(as_frame=True)
    indices = list(range(len(data.target)))
    train, test = train_test_split(indices, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(data.data.iloc[train], data.target.iloc[train])
    predicted = model.predict(data.data.iloc[test])
    save_json(ROOT / "reports/regression_exercise.json", {
        "purpose": "Separate numerical regression learning exercise; not part of WasteSort classification.",
        "dataset": "sklearn.datasets.load_diabetes, bundled numerical benchmark",
        "train_rows": len(train), "test_rows": len(test), "random_seed": SEED,
        "mse": mean_squared_error(data.target.iloc[test], predicted),
        "r2": r2_score(data.target.iloc[test], predicted),
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "limitation": "Educational predictions only; no clinical use or medical interpretation.",
    })
    pd.DataFrame({"source_row": test, "actual": data.target.iloc[test].to_numpy(),
                  "predicted": predicted}).to_csv(ROOT / "reports/regression_predictions.csv", index=False)


if __name__ == "__main__":
    main()
