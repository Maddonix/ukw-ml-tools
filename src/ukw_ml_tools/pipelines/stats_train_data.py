import plotly.express as px


def train_data_plots(stats_df):
    plots = {
        "interventions": [],
        "images": []
    }
    _df = stats_df[stats_df.is_val != "all"]
    plots["images"].append(px.bar(_df, "is_val", "count", title="Image Count"))
    plots["interventions"].append(px.bar(_df, "is_val", "unique", title="Intervention Count"))

    s1 = stats_df["origins"] == "all"
    s2 = stats_df["labels"] != "all"
    _df = stats_df[s1 & s2]
    plots["images"].append(px.bar(_df, "labels", "count", title="Image Labels by Origin"))
    plots["interventions"].append(px.bar(_df, "labels", "unique", title="Labeled Interventions by Origin"))

    s1 = stats_df["origins"] != "all"
    s2 = stats_df["labels"] != "all"
    _df = stats_df[s1 & s2]
    plots["images"].append(px.bar(_df, "origins", "count", color="labels", title="Image Labels by Origin"))
    plots["interventions"].append(px.bar(_df, "origins", "unique", color="labels", title="Labeled Interventions by Origin"))

    return plots
