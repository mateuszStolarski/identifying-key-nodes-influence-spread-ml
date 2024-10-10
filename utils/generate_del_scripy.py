import sys

import pandas as pd


def main():
    if len(sys.argv) == 1:
        filename = "kolejka.txt"
    else:
        filename = sys.argv[1]

    df = pd.read_csv(filename, delim_whitespace=True, header=None)

    df["cmd"] = "qdel"

    df = df[["cmd", 0]]
    df.to_csv("del_jobs.sh", header=False, index=False, sep=" ")


if __name__ == "__main__":
    main()
