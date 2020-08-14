import argparse
import os
import pandas as pd
from geneactiv_reader import GENEActiv


def process_bins(file_list, outfile):

    agg = pd.DataFrame()

    # TODO: align by starting time

    for file in file_list:

        ga = GENEActiv(file)

        df = ga.aggregate("1s")

        # Samo prvih 20 min
        df = df[:20 * 60]

        if agg.index.empty:
            agg["Time"] = df.index.values
            agg = agg.set_index("Time")

        agg[os.path.basename(file).replace(" ", "_") + "_SVM"] = df.values.flatten()

    agg["Total_SVM"] = agg.sum(axis=1)
    agg.to_csv(outfile)
    return agg


if __name__ == "__main__":
    # Example: python preprocessing/accelerometer.py --dir ./directory_of_bin_files

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Directory of .bin files")
    ap.add_argument("-o", "--outfile", required=False, help=".csv output file", default=None)
    args = ap.parse_args()

    args.dir = os.path.normpath(args.dir)
    if args.outfile is None:
        args.outfile = args.dir + os.path.sep + "all_accelerometers.csv"
    else:
        assert args.outfile.endswith(".csv"), "output file must be .csv"

    file_list = [args.dir + os.path.sep + x for x in os.listdir(args.dir) if x.endswith(".bin")]
    process_bins(file_list, args.outfile)
