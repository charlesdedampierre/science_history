import sys

sys.path.append("../")

import sqlite3

import os
import pandas as pd

from dotenv import load_dotenv

# Load the environment variables from the .env file
dotenv_path = os.path.join("../", ".env")
load_dotenv(dotenv_path)


conn = sqlite3.connect(os.getenv("DB_SCIENCE_PATH"))

from nationalities import NATIONALITIES_list

df = pd.read_sql("SELECT * FROM wiki_abtract", conn)
df = df[~df["abstract"].isna()]

import re


def suppress_dates(text):
    try:
        # Regular expression to match individual dates in the format "dd Month yyyy"
        date_pattern = r"\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}"

        # Regular expression to match date ranges in the format "(yyyy-yyyy)"
        date_range_pattern = r"\(\d{4}–\d{4}\)"

        date_range_pattern2 = r"\d{4} – \d{4}"

        # Use re.sub to replace matched dates and date ranges with an empty string
        cleaned_text = re.sub(date_pattern, "", text)
        cleaned_text = re.sub(date_range_pattern, "", cleaned_text)
        cleaned_text = re.sub(date_range_pattern2, "", cleaned_text)

    except:
        cleaned_text = None

    return cleaned_text


import re


def clean_abstract_with_country(abstract, country_names):
    try:
        # Create a case-insensitive regular expression pattern for the country name
        pattern = re.compile(re.escape(country_name), re.IGNORECASE)

        # Replace all occurrences of the country name with an empty string
        cleaned_abstract = pattern.sub("", abstract)
    except:
        cleaned_abstract = None

    return cleaned_abstract


import re
import pycountry

country_names = [country.name for country in pycountry.countries]


def remove_country_names(abstract, country_names):
    try:
        for country_name in country_names:
            # Create a case-insensitive regular expression pattern for the country name
            pattern = re.compile(re.escape(country_name), re.IGNORECASE)

            # Replace all occurrences of the country name with an empty string
            abstract = pattern.sub("", abstract)
    except:
        abstract = None
    return abstract


df["clean_abstract"] = df["abstract"].apply(lambda x: suppress_dates(x))
df["clean_abstract"] = df["clean_abstract"].apply(
    lambda x: remove_country_names(x, country_names)
)
df["clean_abstract"] = df["clean_abstract"].apply(
    lambda x: remove_country_names(x, NATIONALITIES_list)
)
df.to_sql("wiki_abtract", conn, if_exists="replace", index=False)
