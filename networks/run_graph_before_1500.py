import sys

sys.path.append("../")

import sqlite3

import pandas as pd
import polars as pl

from functions.datamodel import OptimumParameter
from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH, GRAPH_RESULTS
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_visualization import sygma_graph

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)

optimal_parameters = pd.read_sql("SELECT * FROM optimization_europe", conn)
optimal_parameters = optimal_parameters.sort_values("mean", ascending=False)

dict_op = optimal_parameters.iloc[0].to_dict()
dict_op = OptimumParameter(**dict_op)

from region_filters import columns_eu

if __name__ == "__main__":
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)

    df_temporal = pd.read_sql("SELECT * FROM temporal_data", conn)
    df_temporal = df_temporal[df_temporal["region_code"].isin(columns_eu)]
    df_temporal = df_temporal[["wikidata_id", "birthyear"]]
    df_temporal = df_temporal[df_temporal["birthyear"] <= 1500]
    print(len(set(df_temporal.wikidata_id)))

    df = pd.merge(df_occupation, df_temporal, on="wikidata_id")
    df = df.drop("birthyear", axis=1)
    df = df.drop_duplicates()

    df.columns = ["source", "target"]
    df["weight"] = 1

    df = pl.from_pandas(df)
    df_edge, df_nodes = get_edge_node_table(df)

    df_edge_filter = filter_edge_table(
        df_edge,
        edge_rule=dict_op.edge_rule,
        top_directed_neighbours=dict_op.n_neighbours,
        normalize_on_top=False,
        min_count_link=dict_op.min_count_link,
    )

    df_partition, g = sygma_graph(
        df_edge_filter,
        df_nodes,
        edge_bins=10,
        node_bins=10,
        resolution=dict_op.resolution,
        filepath=GRAPH_RESULTS + "/before_1500.html",
    )

    df_partition.to_sql("partition_before_1500", conn, if_exists="replace", index=False)
    print(df_partition)
