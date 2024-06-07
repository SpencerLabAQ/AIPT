import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import WEE_PATH


def evaluate_estimate(x):
    if  x['estimated_warmup_time'] > x['actual_warmup_time']:
        return "overestimation"
    if x['estimated_warmup_time'] < x['actual_warmup_time']:
        return "underestimation"
    
    return "correct"




def plot(g, file_name):
    sns.set_theme(style="whitegrid")

    order = ["FCN", "OSCNN", "Rocket", "SOP", "CV", "RCIW", "KLD"]
    hue_order = ["overestimation", "underestimation", "correct"]
    palette = {"overestimation":"gainsboro", "underestimation":"gray", "correct":"whitesmoke"}

    _, ax = plt.subplots(figsize=(12, 3))
    sns.barplot(g, ax=ax, x="technique", y="proportion", hue="evaluation", order=order, hue_order=hue_order, errorbar=None, width=0.8, linewidth=0.5, edgecolor="black", palette=palette)
    ax.bar_label(ax.containers[0], fmt='%.1f')
    ax.bar_label(ax.containers[1], fmt='%.1f')
    ax.bar_label(ax.containers[2], fmt='%.1f')

    # customize legend title
    legend = ax.get_legend()
    legend.set_title('')
    legend.texts[0].set_text(hue_order[0].capitalize())
    legend.texts[1].set_text(hue_order[1].capitalize())
    legend.texts[2].set_text("Exact match")
    # set legend position to upper left
    legend._loc = 2 


    # replace y label
    ax.set_ylabel("Forks (%)")

    # delete x label
    ax.set_xlabel("")

    ax.set_ylim(0, 100)

    plt.tight_layout()


    plt.savefig(file_name)


if __name__ == "__main__":
    df = pd.read_csv(WEE_PATH)
    df["evaluation"] = df.apply(evaluate_estimate, axis=1)

    g = df.groupby('technique')['evaluation'].value_counts(normalize=True).to_frame() * 100
    g.loc[("RCIW", "correct"), :] = 0

    g.reset_index(inplace=True)

    plot(g, "./figures/rq23_estimates.pdf")



