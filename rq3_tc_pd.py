import functools

import numpy as np
import pandas as pd

from constants import TC_PATH, PD_PATH

q1 = functools.partial(np.percentile, q=25)
q1.__name__ = 'q1'

q3 = functools.partial(np.percentile, q=75)
q3.__name__ = 'q3'

rows_order = ["FCN", "OSCNN", "Rocket", "SOP", "CV", "RCIW", "KLD"]


time_cost = pd.read_csv(TC_PATH)
time_cost.technique = time_cost.technique.str.split('_').str[0]
df_tc = time_cost.groupby('technique').agg({'time_cost': [q1, 'median', q3]})

df_tc = df_tc.loc[rows_order]

df_tc

df_tc['Median (IQR)'] = df_tc['time_cost']['median'].apply(lambda x: f"{x:.0f}").str.cat(df_tc['time_cost'][['q1', 'q3']].apply(lambda x: f" ({x[0]:.0f}-{x[1]:.0f})", axis=1))


performance_deviation = pd.read_csv(PD_PATH)
performance_deviation.technique = performance_deviation.technique.str.split('_').str[0]
performance_deviation['not_different'] = (~((performance_deviation['lower_bound'] > 1.0 ) | (performance_deviation['upper_bound'] < 1.0))).astype(int)
performance_deviation['center'] = (performance_deviation['upper_bound'] - performance_deviation['lower_bound'])/2 + performance_deviation['lower_bound']
performance_deviation['deviation'] = abs(performance_deviation['center'] - 1)
performance_deviation= performance_deviation[performance_deviation['not_different'] == 0]

df_pd = performance_deviation.groupby('technique').agg({'deviation': [q1, 'median', q3]})
df_pd = df_pd * 100
df_pd = df_pd.loc[rows_order]

df_pd['Median (IQR)'] = df_pd['deviation']['median'].apply(lambda x: f"{x:.1f}").str.cat(df_pd['deviation'][['q1', 'q3']].apply(lambda x: f" ({x[0]:.1f}-{x[1]:.1f})", axis=1))
df_pd = df_pd.loc[rows_order]


df = pd.concat([df_pd['Median (IQR)'], df_tc['Median (IQR)']], axis=1)

df.columns = ["Rel. Meas. Dev. (\%)", "Time Cost (sec)"]


df.to_latex(f"./tables/rq23_tc_pd.tex", float_format="%.0f",  multicolumn_format="c", index=True, column_format="lcc")




