import csv
import os

#! python2 compatible


def to_csv(filename, data):
    with open(filename, "w") as handle:
        csvwriter = csv.writer(handle)
        csvwriter.writerows(data)


def python3_main():
    from pathlib import Path

    root = Path("data/processed")
    result = [["graph", "treshold", "count"]]

    for graph in root.glob("*"):
        for treshold in graph.glob("*"):
            count = len(list(treshold.glob("*")))
            result.append([graph.stem, treshold.name, count])

    to_csv("report.csv", result)


def main():
    root = "data/processed"
    result = [["graph", "treshold", "count"]]

    for graph in os.listdir(root):
        graph_dir = os.path.join(root, graph)
        if not os.path.isdir(graph_dir):
            continue

        for treshold in os.listdir(os.path.join(root, graph)):
            treshold_dir = os.path.join(root, os.path.join(graph, treshold))
            if not os.path.isdir(treshold_dir):
                continue

            count = len(os.listdir(treshold_dir))
            result.append([graph, treshold, count])

    to_csv("report.csv", result)


if __name__ == "__main__":
    main()
