import sys

sys.path.append("../")

import sqlite3

import pandas as pd
import polars as pl
from tqdm import tqdm

from functions.datamodel import OptimumParameter
from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_visualization import sygma_graph, sygma_graph_leiden

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)

from optimal_clustering import optimal_clustering

dict_op = optimal_clustering
dict_op = OptimumParameter(**dict_op)

if __name__ == "__main__":
    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df_occupation.columns = ["source", "target"]
    df_occupation["weight"] = 1

    all_regions = list(set(df_ind_regions["region_code"]))
    print(all_regions)

    final_clustering = []
    for region_code in tqdm(all_regions):
        try:
            df_region = df_ind_regions[df_ind_regions["region_code"] == region_code]
            wiki_ids = list(set(df_region["wikidata_id"]))
            df = df_occupation[df_occupation["source"].isin(wiki_ids)]
            df = pl.from_pandas(df)
            df_edge, df_nodes = get_edge_node_table(df)

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
                filepath=f"../graph/region/{region_code}.html",
            )

            df_partition["region_code"] = region_code
            final_clustering.append(df_partition)

        except Exception as e:
            print(f"Not enought data for {region_code}")

    # Compare the profiles of the different regions
    df_res = pd.concat([x for x in final_clustering])
    df_res.to_sql("region_optimized_partition", conn, if_exists="replace", index=False)
