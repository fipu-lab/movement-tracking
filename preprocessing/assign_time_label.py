import pandas as pd
import datetime


def assign_time(csv_path, fps, start_time):
    start_time = datetime.datetime.fromisoformat(start_time)

    step = 1/fps

    csv = pd.read_csv(csv_path, index_col=0)

    time_labels = [start_time + datetime.timedelta(seconds=step*i) for i in range(len(csv) + 60)][60:] # csv pocinje sa 60 frame-ova

    csv["time"] = time_labels

    csv.to_csv(csv_path)


def main():
    dir = "/Volumes/GoogleDrive/Shared drives/Uniri projekt FIPU/3_Formula/"
    csvs = [
        ["active_cam1_formula.csv", "2020-01-30T08:42:41"],
        ["active_cam2_formula.csv", "2020-01-30T08:32:34"],
        ["active_cam3_formula.csv", "2020-01-30T08:26:24"],
        ["inactive_cam1_formula.csv", "2020-02-05T08:31:15"],
        ["inactive_cam2_formula.csv", "2020-02-05T08:21:56"],
        ["inactive_cam3_formula.csv", "2020-02-05T08:35:03"]
    ]

    for p, t in csvs:
        print(p, t)
        assign_time(dir + p, 25, t)
