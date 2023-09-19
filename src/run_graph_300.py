import sqlite3
import pandas as pd
import polars as pl
from functions.feat_network import get_edge_node_table, filter_edge_table
from functions.feat_visualization import sygma_graph
from functions.datamodel import OptimumParameter

from functions.env import GRAPH_RESULTS, DB_SCIENCE_PATH, FULL_DB_PATH

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)

optimal_parameters = pd.read_sql("SELECT * FROM optimization", conn)
optimal_parameters = optimal_parameters.sort_values("mean", ascending=False)

dict_op = optimal_parameters.iloc[0].to_dict()
dict_op = OptimumParameter(**dict_op)

columns_to_keep = [
    "re_arabic_world",
    "re_central_europe",
    "re_chinese_world",
    "re_eastern_europe",
    "re_france",
    "re_german_world",
    "re_greek_world",
    "re_indian_world",
    "re_italy",
    "re_japan",
    "re_low_countries",
    "re_nordic_countries",
    "re_persian_world",
    "re_slav_world",
    "re_spain",
    "re_united_kingdom",
]


if __name__ == "__main__":
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)

    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_final = pd.merge(df_occupation, df_ind_regions, on="wikidata_id")

    df_regions = df_final[["wikidata_id", "region_code"]].drop_duplicates()
    df_regions = df_regions[df_regions["region_code"].isin(columns_to_keep)]

    final = []

    for region in columns_to_keep:
        df_sample = df_regions[df_regions["region_code"] == region]

        if len(df_sample) > 100:
            res = df_sample.sample(100, random_state=42)
        else:
            res = df_sample.copy()

        final.append(res)

    df_fin = pd.concat([x for x in final])

    individuals_filters = list(set(df_fin["wikidata_id"]))

    df = df_occupation[df_occupation["wikidata_id"].isin(individuals_filters)]
    df.columns = ["source", "target"]
    df["weight"] = 1

    df = pl.from_pandas(df)
    df_edge, df_nodes = get_edge_node_table(df)

    df_edge_filter = filter_edge_table(
        df_edge,
        edge_rule=dict_op.edge_rule,
        top_directed_neighbours=dict_op.n_neighbours,
        normalize_on_top=False,
        min_count_link=0,
    )

    df_partition = sygma_graph(
        df_edge_filter,
        df_nodes,
        edge_bins=10,
        node_bins=10,
        resolution=dict_op.resolution,
        filepath=GRAPH_RESULTS + "/region/300.html",
    )
