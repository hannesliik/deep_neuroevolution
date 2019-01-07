import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("-c", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
args = parser.parse_args()

dirs = os.walk(args.path)
fig = plt.figure(dpi=200)
if not args.c:
    for dir in list(dirs)[1:]:
        print(dir[0])
        df = pd.read_csv(os.path.join(dir[0], "plot.csv"))
        sns.lineplot(data=df, x="frames", y="score", ci="sd")
    plt.show()
else:
    df = None
    for dir in list(dirs)[1:]:
        print(dir[0])
        if df is None:
            df = pd.read_csv(os.path.join(dir[0], "plot.csv"))
        else:
            df = pd.concat((df, pd.read_csv(os.path.join(dir[0], "plot.csv"))))
    sns.lineplot(data=df, x="frames", y="score", ci="sd")
    if args.save:
        plt.imsave("plot.png")
    else:
        plt.show()
