import pandas as pd
import sqlite3
from functions.env import DATA_PATH, DB_SCIENCE_PATH

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH + "/df_cleaned_occupations.csv", index_col=[0])
    conn = sqlite3.connect(DB_SCIENCE_PATH)
    data.to_sql(
        "individual_id_cleaned_occupations", conn, if_exists="replace", index=False
    )
