import pandas as pd

from constants import EVAL_MODELS_METRICS_PATH

if __name__ == '__main__':
    df = pd.read_csv(EVAL_MODELS_METRICS_PATH)
    labels = {"clf": "Model", "bal_acc": "Bal. Acc.", "f1": "F1", "prec": "Prec.", "rec": "Rec."}

    df = df.groupby("clf").mean().reset_index()
    # rename columns
    df.rename(columns=labels, inplace=True)
    df = df[["Model", "Prec.", "Rec.", "F1", "Bal. Acc."]]   
    
    df.to_csv("./tables/rq1.csv", index=False)

    df.rename(columns=lambda c:"\\textbf{" + c + "}" , inplace=True)
    df.to_latex("./tables/rq1.tex", float_format="%.3f", column_format="lcccc",  index=False)

