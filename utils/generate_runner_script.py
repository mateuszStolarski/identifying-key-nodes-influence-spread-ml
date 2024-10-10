import random

#! python2 compatible


def main():
    # "Citeseer", "Facebook", "Github", "Pubmed"
    datasets = ["Citeseer"]
    tresholds = [0.3, 0.4]

    result = []

    for dataset in datasets:
        for treshold in tresholds:
            for _ in range(100):
                command = "qsub -v treshold=%s scripts/run_%s.sh \n" % (
                    treshold,
                    dataset,
                )
                result.append(command)

    random.shuffle(result)

    with open("scripts/unleash.sh", "w") as handle:
        handle.writelines(result)


if __name__ == "__main__":
    main()
