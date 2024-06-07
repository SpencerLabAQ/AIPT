import pandas as pd

from constants import TC_PATH, PD_PATH

time_cost = pd.read_csv(TC_PATH)
time_cost = time_cost.pivot(index="benchmark", columns="technique", values="time_cost")


performance_deviation = pd.read_csv(PD_PATH)
performance_deviation['not_different'] = (~((performance_deviation['lower_bound'] > 1.0 ) | (performance_deviation['upper_bound'] < 1.0))).astype(int)
performance_deviation = performance_deviation.pivot(index="benchmark", columns="technique", values="not_different")

baseline = "SOP"
models = ["FCN", "OSCNN", "Rocket"]

index = [('', 'clf'),
         ('impr', 'qual',),
         ('impr', 'time'),
         ('impr', 'tot'),
         ('regr', 'qual'),
         ('regr', 'time'),
         ('regr', 'tot')]

res = pd.DataFrame(columns=pd.MultiIndex.from_tuples(index)).set_index(('', 'clf'))

for model in models:
    technique = f"{model}_{baseline}"
    quality_improvements = ((performance_deviation[baseline] == 0) & (performance_deviation[technique] == 1)).sum()
    cost_reduction = (( time_cost[technique] < time_cost[baseline]) & (performance_deviation[baseline] == 1) & (performance_deviation[technique] == 1)).sum()
    cost_increase = (( time_cost[technique] > time_cost[baseline]) & (performance_deviation[baseline] == 1) & (performance_deviation[technique] == 1)).sum()
    quality_deterioration = ((performance_deviation[baseline] == 1) & (performance_deviation[technique] == 0)).sum()
 
    res.loc[model, ('impr', 'qual')] = quality_improvements/len(performance_deviation)
    res.loc[model, ('impr', 'time')] = cost_reduction/len(performance_deviation)
    res.loc[model, ('impr', 'tot')] = (quality_improvements + cost_reduction)/len(performance_deviation)
    res.loc[model, ('regr', 'qual')] = quality_deterioration/len(performance_deviation)
    res.loc[model, ('regr', 'time')] = cost_increase/len(performance_deviation)
    res.loc[model, ('regr', 'tot')] = (quality_deterioration + cost_increase)/len(performance_deviation)


#transformto percentage
res = res*100

#net improvementS
res[("net_impr", "tot_diff")] = res[("impr", "tot")] - res[("regr", "tot")]

#Store csv results
res.to_csv("./results/rq2_impr.csv", index=True)


# Format Total  columns
tot_formatter = lambda x: "\\textbf{\\textit{" + "{:,.1f}".format(x) + "}}"
diff_formatter = lambda x: "\\textbf{\\textit{" + "{:+,.1f}".format(x) + "}}"
res[("impr", "tot")] = res[("impr", "tot")].map(tot_formatter)
res[("regr", "tot")] = res[("regr", "tot")].map(tot_formatter)
res[("net_impr", "tot_diff")] = res[("net_impr", "tot_diff")].map(diff_formatter)

#format columns
res.reset_index(inplace=True)
res[("", "clf")] += "  \emph{vs.} SOP"
labels = { "clf": "\\textbf{Model \\emph{vs.} SOP}",
           "impr": "\\textbf{Improvement (\\%)}",
            "regr": "\\textbf{Regression (\\%)}",
            "qual": "Meas. Quality", "time": "Time Cost",
            "tot": "\\textbf{\\textit{Total}}",
            "net_impr": "\\textbf{Net Improvement (\\%)}",
            "tot_diff": "(\\textbf{\\textit{Tot. Impr. - Tot. Regr.}})"}
res.rename(columns=labels, inplace=True)

# Store latex table
res.to_latex("./tables/rq2_impr.tex", float_format="%.1f",  multicolumn_format="c", index=False, column_format="l|ccc|ccc|c")






