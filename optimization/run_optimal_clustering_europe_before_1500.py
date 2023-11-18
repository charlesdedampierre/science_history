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
    get_rand_index,
    get_rand_index_keep_identical,
)
from functions.feat_visualization import sygma_graph

from region_filters import columns_eu

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)

if __name__ == "__main__":
    data = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)

    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_ind_regions = df_ind_regions[df_ind_regions["region_code"].isin(columns_eu)]
    df_ind_regions = df_ind_regions[["wikidata_id", "region_code"]]

    df_temporal = pd.read_sql("SELECT * FROM temporal_data", conn)
    df_temporal = df_temporal[df_temporal["region_code"].isin(columns_eu)]
    df_temporal = df_temporal[["wikidata_id", "birthyear"]]
    df_temporal = df_temporal[df_temporal["birthyear"] <= 1500]

    df = pd.merge(df_ind_regions, df_temporal, on="wikidata_id")
    df = df.drop("birthyear", axis=1)
    df = df.drop_duplicates()

    list_individuals = list(df["wikidata_id"])

    data = data[data["wikidata_id"].isin(list_individuals)]
    data = data.drop_duplicates()
    print(len(data))

    data.columns = ["source", "target"]
    data["weight"] = 1

    final_dict = []
    n_sample_individual = int(round(len(data) / 10, 0))
    batch_number = 10

    for n_neighbours in tqdm(np.arange(2, 15)):
        for edge_rule in ["count"]:
            for resolution in [1]:
                for min_count_link in [1]:
                    final_partition = []
                    for seed in np.arange(batch_number):
                        data_id = data[["source"]].drop_duplicates()

                        if len(data_id) < n_sample_individual:
                            df = data.copy()

                        else:
                            data_id_sample = list(
                                data_id.sample(n_sample_individual, random_state=seed)[
                                    "source"
                                ]
                            )
                            df = data[data["source"].isin(data_id_sample)]

                        df = pl.from_pandas(df)
                        df_edge, df_nodes = get_edge_node_table(df)

                        df_edge_filter = filter_edge_table(
                            df_edge,
                            edge_rule=edge_rule,
                            top_directed_neighbours=n_neighbours,
                            normalize_on_top=False,
                            min_count_link=2,
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

                    # node | community

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
                    "edge_rule": edge_rule,
                    "resolution": resolution,
                    "min_count_link": min_count_link,
                }

                final_dict.append(new_dict)

    df_final = pd.DataFrame(final_dict)
    df_final["n_neighbours"] = df_final["n_neighbours"].astype(int)
    df_final = df_final.sort_values("mean", ascending=False)
    df_final.to_sql(
        "optimization_europe_before_1500", conn, if_exists="replace", index=False
    )
    print(df_final)
