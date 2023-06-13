import sqlite3
import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import seaborn as sns
import matplotlib.pyplot as plt

directory = "images"
if not os.path.exists(directory):
    os.makedirs(directory)

conn = sqlite3.connect("database.db")

if __name__ == "__main__":
    df = pd.read_sql("SELECT * FROM region_optimized_partition", conn)
    df_res = df.pivot(index="node", columns="region_code", values="community")

    columns_to_compare = [
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
        "re_latin",
        "re_low_countries",
        "re_muslim_world",
        "re_nordic_countries",
        "re_persian_world",
        "re_slav_world",
        "re_spain",
        "re_united_kingdom",
        "re_western_europe",
    ]

    ari_values = {}
    final = []

    # Iterate through each pair of columns
    for i in range(len(columns_to_compare)):
        for j in range(i + 1, len(columns_to_compare)):
            column1 = columns_to_compare[i]
            column2 = columns_to_compare[j]

            df = df_res[[column1, column2]].dropna()
            ari = adjusted_rand_score(df[column1], df[column2])

            dict_row = {"region_1": column1, "region_2": column2, "ari": ari}
            dict_row_2 = {"region_2": column1, "region_1": column2, "ari": ari}
            final.append(dict_row)
            final.append(dict_row_2)

    df_final = pd.DataFrame(final)
    df_final = df_final.pivot(index="region_1", columns="region_2", values="ari")
    df_final = df_final.fillna(1)

    fig = sns.clustermap(df_final, cmap="Blues")
    plt.savefig("images/region_ari_comparison.png", dpi=300)
