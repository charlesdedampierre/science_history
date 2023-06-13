import pandas as pd
import sqlite3

if __name__ == "__main__":
    data = pd.read_csv("data/df_cleaned_occupations.csv", index_col=[0])
    conn = sqlite3.connect("database.db")
    data.to_sql(
        "individual_id_cleaned_occupations", conn, if_exists="replace", index=False
    )
