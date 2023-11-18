import sqlite3

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from functions.env import DB_SCIENCE_PATH
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_optimization import get_mean, get_rand_index
from functions.feat_visualization import sygma_graph

if __name__ == "__main__":
    conn = sqlite3.connect(DB_SCIENCE_PATH)
    data = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    data.columns = ["source", "target"]
    data["weight"] = 1

    final_dict = []
    n_sample_individual = 5000
    batch_number = 10

    for n_neighbours in tqdm(np.arange(3, 15)):
        for edge_rule in ["specificity", "count"]:
            for resolution in [1, 2]:
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
                        edge_rule=edge_rule,
                        top_directed_neighbours=n_neighbours,
                        normalize_on_top=False,
                    )

                    df_partition, g = sygma_graph(
                        df_edge_filter,
                        df_nodes,
                        edge_bins=10,
                        node_bins=10,
                        resolution=resolution,
                        filepath="graph/clean_graph.html",
                    )

                    df_partition = df_partition.rename(
                        columns={"community": f"community_{seed}"}
                    )
                    final_partition.append(df_partition)

                    # node | community

                # Merge the different clustering of the different samples together
                merged_df = final_partition[0]
                for df in final_partition[1:]:
                    merged_df = pd.merge(merged_df, df, on="node")
                merged_df = merged_df.set_index("node")

                final_list = []
                for col in merged_df.columns:
                    list_community = list(merged_df[col])
                    final_list.append(list_community)

                similarity_matrix = get_rand_index(final_list)
                mean = get_mean(similarity_matrix)

                new_dict = {
                    "n_neighbours": n_neighbours,
                    "mean": mean,
                    "edge_rule": edge_rule,
                    "resolution": resolution,
                }

                final_dict.append(new_dict)

    df_final = pd.DataFrame(final_dict)
    df_final["n_neighbours"] = df_final["n_neighbours"].astype(int)
    df_final = df_final.sort_values("mean", ascending=False)
    df_final.to_sql("optimization", conn, if_exists="replace", index=False)
