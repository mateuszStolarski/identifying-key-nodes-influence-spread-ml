from pathlib import Path

import pandas as pd

COLUMNS = ["max_si", "max_iter", "max_recovered"]


def main():
    root = Path("data/processed")

    for graph in root.glob("*"):
        for treshold in graph.glob("*"):
            if not treshold.is_dir():
                continue

            treshold_files = []

            for index, file in enumerate(treshold.glob("*.csv")):
                df = pd.read_csv(file)
                df.columns = [column + f"_{index}" for column in df.columns]
                treshold_files.append(df)

            if not treshold_files:
                continue

            tresh_df = pd.concat(treshold_files, axis=1)
            for col in COLUMNS:
                tresh_df[f"{col}_mean"] = tresh_df.filter(regex=(f"{col}_*")).mean(
                    axis=1
                )

            tresh_df = tresh_df[
                [f"{col}_mean" for col in COLUMNS]
            ]  # leave only averaged columns
            tresh_df.to_csv(graph / f"{graph.stem}_{treshold.name}.csv", index=False)


if __name__ == "__main__":
    main()
