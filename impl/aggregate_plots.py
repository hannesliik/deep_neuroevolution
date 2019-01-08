import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("-c", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-x", type=str, choices=["frames", "time", "generation"], default="generation")
args = parser.parse_args()

dirs = os.walk(args.path)
fig = plt.figure(dpi=200)
if not args.c:
    for dir in list(dirs)[1:]:
        print(dir[0])
        df = pd.read_csv(os.path.join(dir[0], "plot.csv"))
        with open(os.path.join(dir[0], "params.json"), "r") as fp:
            exp_data = json.load(fp)
        sns.lineplot(data=df, x=args.x, y="score", ci="sd", label=exp_data["exp_name"])
else:
    df = None
    for dir in list(dirs)[1:]:
        print(dir[0])
        if df is None:
            df = pd.read_csv(os.path.join(dir[0], "plot.csv"))
        else:
            df = pd.concat((df, pd.read_csv(os.path.join(dir[0], "plot.csv"))))
    sns.lineplot(data=df, x=args.x, y="score", ci="sd", label="placeholder")
plt.ylabel("Mean score")
plt.legend()
if args.save:
    plt.imsave("plot.png")
else:
    plt.show()
