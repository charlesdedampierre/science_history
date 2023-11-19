import sys

sys.path.append("../")

import sqlite3
import pandas as pd
import polars as pl
import numpy as np

from functions.datamodel import OptimumParameter
from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH, GRAPH_RESULTS
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_visualization import sygma_graph, sygma_graph_leiden

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)


from optimal_clustering import optimal_clustering

dict_op = optimal_clustering
dict_op = OptimumParameter(**dict_op)

from region_filters import columns_eu

if __name__ == "__main__":
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df_temporal = pd.read_sql("SELECT * FROM temporal_data", conn)
    df_temporal = df_temporal[df_temporal["region_code"].isin(columns_eu)]
    df_temporal = df_temporal[["wikidata_id", "birthyear"]]

    centuries = np.arange(800, 2000, 100)
    print(centuries)

    for century in centuries:
        df_temporal_filtered = df_temporal[df_temporal["birthyear"] <= century]

        df = pd.merge(df_occupation, df_temporal_filtered, on="wikidata_id")
        df = df.drop("birthyear", axis=1)
        df = df.drop_duplicates()

        df.columns = ["source", "target"]
        df["weight"] = 1
        df = df.drop_duplicates()

        print(len(set(df.source)))

        df = pl.from_pandas(df)
        df_edge, df_nodes = get_edge_node_table(df)

        century_str = str(century)

        df_edge_filter = df_edge[df_edge["weight"] >= dict_op.min_count_link]
        df_edge_filter = df_edge_filter[
            df_edge_filter["source"] != df_edge_filter["target"]
        ]
        df_edge_filter = df_edge_filter[
            df_edge_filter["rank_count"] <= dict_op.n_neighbours
        ]

        df_partition, g = sygma_graph_leiden(
            df_edge_filter,
            df_nodes,
            edge_bins=10,
            node_bins=10,
            filepath=GRAPH_RESULTS + f"/before_europe_{century_str}.html",
        )

        df_partition.to_sql(
            f"partition_europe_before_{century_str}",
            conn,
            if_exists="replace",
            index=False,
        )
