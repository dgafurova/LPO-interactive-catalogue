import csv


def trajectory_csv_to_json(file_name):
    reader = csv.DictReader(open(file_name))
    v = list()
    for row in reader:
        v.append(row)

    return v


if __name__ == "__main__":
    vectors = trajectory_csv_to_json('/Users/nikitashvyrev/Downloads/orb2.csv')
    print(vectors)
