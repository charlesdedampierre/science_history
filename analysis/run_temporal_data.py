import sqlite3

import pandas as pd

from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)


if __name__ == "__main__":
    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_birthyear = pd.read_sql_query(
        "SELECT * FROM individuals_main_information", conn_full_db
    )
    df_birthyear = df_birthyear[["individual_wikidata_id", "birthyear"]]
    df_birthyear = df_birthyear.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    dict_interest = {0: "nature", 1: "humans", 2: "abstract"}

    df_baseline = pd.read_sql("SELECT * FROM optimal_partition", conn)
    df_baseline = df_baseline.rename(
        columns={"community": "community_baseline", "node": "meta_occupation"}
    )
    df_baseline["interest"] = df_baseline["community_baseline"].apply(
        lambda x: dict_interest.get(x)
    )

    df_region_partition = pd.read_sql("SELECT * FROM region_optimized_partition", conn)
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df_occupation = pd.merge(df_occupation, df_baseline, on="meta_occupation")

    df_final = pd.merge(df_ind_regions, df_birthyear, on="wikidata_id")
    df_final = pd.merge(df_final, df_occupation, on="wikidata_id")
    df_final = df_final[["wikidata_id", "region_code", "birthyear", "interest"]]
    df_final = df_final.drop_duplicates()
    df_final["decade"] = df_final["birthyear"].apply(lambda x: round(x / 10) * 10)
    df_final["fifty"] = df_final["birthyear"].apply(lambda x: round(x / 50) * 50)
    df_final["century"] = df_final["birthyear"].apply(lambda x: round(x / 100) * 100)

    df_final.to_sql("temporal_data", conn, if_exists="replace", index=False)
