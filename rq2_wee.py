from bisect import bisect_left, bisect_right

import pandas as pd
import pingouin as pg

from constants import WEE_PATH


def interpret_A12(A12):
    assert (0 <= A12 <= 1)

    magnitude = ["N", "S", "M", "L"]
    if A12 >= 0.5:
        levels = [0.56, 0.64, 0.71]
        i = bisect_right(levels, A12)
    else:
        levels = [0.29, 0.34, 0.44]
        magnitude.reverse()
        i = bisect_left(levels, A12)

    return magnitude[i]


wee = pd.read_csv(WEE_PATH)
baseline = "SOP"
models = ["FCN", "OSCNN", "Rocket"]

res = pd.DataFrame(columns=["clf", "p", "A12", "r", "interpretation"]).set_index("clf")

wee = wee.pivot(index=["benchmark", "no_fork"], columns="technique", values="wee")

for model in models:
    wee_ = wee[[model, baseline]].dropna()
    impr = (wee_[model]< wee_[baseline]).sum()
    det = (wee_[model]> wee_[baseline]).sum()

    stats = pg.wilcoxon(wee_[model], wee_[baseline], alternative="less")
    p, A12, r = stats.loc["Wilcoxon", ["p-val", "CLES", "RBC"]]
    interpretation = interpret_A12(A12)
    res.loc[model, "p"] = p
    res.loc[model, "A12"] = A12
    res.loc[model, "r"] = -r
    res.loc[model, "interpretation"] = interpretation

res.to_csv("./results/rq2_wee.csv", index=True)

# drop interpretation column for paper table
res.drop(columns="interpretation", inplace=True)

# reset index
res.reset_index(inplace=True)

# format p-values < 0.001
res['p'] = res['p'].apply(lambda x: x if x >= 0.001 else "$<$0.001")

# format model column
res["clf"] += " \emph{vs.} SOP"

#format columns
labels = { "p": "$p$-value", "A12": "A12", "r": "$r$", "clf": "\\textbf{Model \\emph{vs.} SOP}"}
#index = [("", "\\textbf{Model}")] + [("\\textbf{Model \\emph{vs.} SOP}", labels[c]) for c in res.columns if c != "clf"]
#res.columns =pd.MultiIndex.from_tuples(index)
res.rename(columns=labels, inplace=True)


res.to_latex("./tables/rq2_wee.tex", float_format="%.3f", index=False, column_format="lccc")






