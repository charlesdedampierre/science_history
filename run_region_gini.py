import numpy as np
import plotly.express as px
import sqlite3
import pandas as pd
import os
from src.feat_utils import gini

from dotenv import load_dotenv

load_dotenv()

conn_full_db = sqlite3.connect(os.getenv("FULL_DB_PATH"))
conn = sqlite3.connect("database.db")


if __name__ == "__main__":
    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_baseline = df = pd.read_sql("SELECT * FROM optimal_partition", conn)
    df_baseline = df_baseline.rename(columns={"node": "meta_occupation"})
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df = pd.merge(df_baseline, df_occupation, on="meta_occupation")

    cat_name = {2: "abstract", 0: "nature", 1: "humans"}
    df["interest"] = df["community"].apply(lambda x: cat_name.get(x))

    df = pd.merge(df, df_ind_regions, on="wikidata_id")
    df = df[
        ["meta_occupation", "interest", "region_code", "wikidata_id"]
    ].drop_duplicates()
    df = (
        df.groupby(["region_code", "interest"])["wikidata_id"]
        .count()
        .rename("count_interest")
        .reset_index()
    )

    all_regions = list(set(df["region_code"]))

    gini_list = []
    for region in all_regions:
        df_region = df[df["region_code"] == region]

        values = np.array(list(df_region["count_interest"]))
        gini_list.append({"region": region, "gini": gini(values)})

    df_gini = pd.DataFrame(gini_list)
    df_gini = df_gini.sort_values("gini")
    df_gini.to_sql("region_gini_interets", conn, if_exists="replace", index=False)

    fig = px.bar(
        df_gini,
        x="gini",
        y="region",
        template="simple_white",
        height=600,
        width=500,
        title="Gini Index",
    )

    fig.write_image(f"images/gini_region.png", scale=5)
