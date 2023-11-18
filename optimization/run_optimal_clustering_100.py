import sys

sys.path.append("../")
import sqlite3

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_optimization import (
    get_mean,
    get_rand_index_keep_identical,
)
from functions.feat_visualization import sygma_graph

from networks.region_filters import columns_eu_unique, columns_non_eu_unique

conn = sqlite3.connect(DB_SCIENCE_PATH)
conn_full_db = sqlite3.connect(FULL_DB_PATH)

if __name__ == "__main__":
    data_occupations = pd.read_sql(
        "SELECT * FROM individual_id_cleaned_occupations", conn
    )

    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_regions = df_ind_regions[["wikidata_id", "region_code"]].drop_duplicates()
    df_regions = df_regions[
        df_regions["region_code"].isin(columns_eu_unique + columns_non_eu_unique)
    ]

    df_regions = pd.merge(df_regions, data_occupations, on="wikidata_id")
    df_regions = df_regions.drop_duplicates()

    sample_size = 100
    data = df_regions.groupby("region_code").sample(
        sample_size, random_state=42, replace=True
    )

    data = data.drop("region_code", axis=1)

    data.columns = ["source", "target"]
    data["weight"] = 1

    n_sample_individual = int(len(set(data.source)) / 10)
    batch_number = 10

    final_dict = []
    for n_neighbours in tqdm(np.arange(1, 4)):
        for resolution in [1]:
            final_partition = []
            for seed in np.arange(batch_number):
                data_id = data[["source"]].drop_duplicates()
                data_id_sample = list(
                    data_id.sample(n_sample_individual, random_state=seed)["source"]
                )
                df = data[data["source"].isin(data_id_sample)]

                df = pl.from_pandas(df)
                df_edge, df_nodes = get_edge_node_table(df)

                df_edge_filter = filter_edge_table(
                    df_edge,
                    edge_rule="count",
                    top_directed_neighbours=n_neighbours,
                    normalize_on_top=False,
                    min_count_link=1,
                )

                df_partition, g = sygma_graph(
                    df_edge_filter,
                    df_nodes,
                    edge_bins=5,
                    node_bins=10,
                    resolution=resolution,
                    filepath="../graph/cached_graph.html",
                )

                df_partition = df_partition.rename(
                    columns={"community": f"community_{seed}"}
                )
                final_partition.append(df_partition)
            # Merge the different clustering of the different samples together
            merged_df = final_partition[0]
            for df in final_partition[1:]:
                merged_df = pd.merge(merged_df, df, on="node", how="outer")
            merged_df = merged_df.set_index("node")

            final_list = []
            for col in merged_df.columns:
                list_community = list(merged_df[col])
                final_list.append(list_community)

            similarity_matrix = get_rand_index_keep_identical(final_list)
            mean = get_mean(similarity_matrix)

            new_dict = {
                "n_neighbours": n_neighbours,
                "mean": mean,
                "edge_rule": "count",
                "resolution": resolution,
                "min_count_link": 1,
            }

            final_dict.append(new_dict)

    df_final = pd.DataFrame(final_dict)
    df_final["n_neighbours"] = df_final["n_neighbours"].astype(int)
    df_final = df_final.sort_values("mean", ascending=False)
    df_final.to_sql("optimization_100", conn, if_exists="replace", index=False)
    print(df_final)
