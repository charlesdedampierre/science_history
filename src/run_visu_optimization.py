import pandas as pd
import plotly.express as px
import sqlite3
from functions.env import DB_SCIENCE_PATH, IMAGES_PATH


conn = sqlite3.connect(DB_SCIENCE_PATH)


if __name__ == "__main__":
    df = pd.read_sql("SELECT * FROM optimization", conn)

    df["category"] = df.apply(
        lambda x: str(x["edge_rule"]) + " + resolution " + str(x["resolution"]),
        axis=1,
    )

    df = df.sort_values("mean", ascending=False).reset_index(drop=True)
    df_fig = df[["n_neighbours", "mean", "category"]].copy()
    df_fig = df_fig.sort_values(["category", "n_neighbours"])

    fig = px.line(
        df_fig, x="n_neighbours", y="mean", color="category", template="simple_white"
    )
    fig.write_image(IMAGES_PATH + "/optimization.png", scale=5)
    fig.show()
