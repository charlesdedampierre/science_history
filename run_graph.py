import sqlite3
import os
import pandas as pd
import polars as pl

from src.feat_network import get_edge_node_table, filter_edge_table
from src.feat_visualization import sygma_graph
from src.datamodel import OptimumParameter

directory = "graph"
if not os.path.exists(directory):
    os.makedirs(directory)

conn = sqlite3.connect("database.db")


optimal_parameters = pd.read_sql("SELECT * FROM optimization", conn)
optimal_parameters = optimal_parameters.sort_values("mean", ascending=False)

dict_op = optimal_parameters.iloc[0].to_dict()
dict_op = OptimumParameter(**dict_op)


if __name__ == "__main__":
    conn = sqlite3.connect("database.db")
    df = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df.columns = ["source", "target"]
    df["weight"] = 1

    # Draw the graph
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
        filepath="graph/optimal_graph.html",
    )

    df_partition.to_sql("optimal_partition", conn, if_exists="replace", index=False)
