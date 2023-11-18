import sqlite3
import sys
from functions import get_wiki_abstract
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


sys.path.append("../")

import os

from dotenv import load_dotenv


# Load the environment variables from the .env file
dotenv_path = os.path.join("../", ".env")
load_dotenv(dotenv_path)


conn = sqlite3.connect(os.getenv("DB_SCIENCE_PATH"))
conn_full_db = sqlite3.connect(os.getenv("FULL_DB_PATH"))


if __name__ == "__main__":
    df = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df_name = pd.read_sql("SELECT * FROM individuals_main_information", conn_full_db)
    df_name = df_name[["individual_wikidata_id", "individual_name"]]
    df_name = df_name.rename(columns={"individual_wikidata_id": "wikidata_id"})

    df_final = pd.merge(df, df_name, on="wikidata_id")
    df_final = (
        df_final[["wikidata_id", "individual_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    names = list(df_final.sample(10, random_state=42).individual_name)
    names = list(df_final.individual_name)

    with Pool(8) as p:
        res = list(tqdm(p.imap(get_wiki_abstract, names), total=len(names)))

    df_res = pd.DataFrame({"individual_name": names, "abstract": res})
    df_res = pd.merge(df_res, df_final, on="individual_name")

    df_res.to_sql("wiki_abtract", conn, if_exists="replace", index=False)

    """
    res = []
    for name in tqdm(names):
        try:
            wiki_abstract = get_wiki_abstract(title=name, wiki="en")
            res.append({"individual_name": name, "abstract": wiki_abstract})
        except:
            pass
            
    """

    print(df_res)
