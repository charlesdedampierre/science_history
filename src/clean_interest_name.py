from functions.env import DB_SCIENCE_PATH
import sqlite3
import pandas as pd

conn = sqlite3.connect(DB_SCIENCE_PATH)
df = pd.read_sql("SELECT * FROM temporal_data", conn)

dict = {
    "nature": "Natural Domain",
    "humans": "Human Domain",
    "abstract": "Abstract Domain",
}

df["interest"] = df["interest"].apply(lambda x: dict.get(x))
df.to_sql("temporal_data_clean", conn, if_exists="replace", index=False)
