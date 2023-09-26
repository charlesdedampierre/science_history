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


if __name__ == "__main__":
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)

    df_temporal = pd.read_sql("SELECT * FROM temporal_data", conn)
    df_temporal = df_temporal[["wikidata_id", "birthyear"]]
    df_temporal = df_temporal[df_temporal["birthyear"] <= 1600]

    df = pd.merge(df_occupation, df_temporal, on="wikidata_id")
    df = df.drop("birthyear", axis=1)

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
        filepath=GRAPH_RESULTS + "/region/before_1600.html",
    )
