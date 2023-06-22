import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
DB_SCIENCE_PATH = os.getenv("DB_SCIENCE_PATH")
GRAPH_RESULTS = os.getenv("GRAPH_RESULTS")
FULL_DB_PATH = os.getenv("FULL_DB_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")


if __name__ == "__main_":
    directory = GRAPH_RESULTS + "/region"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = GRAPH_RESULTS
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = GRAPH_RESULTS
    if not os.path.exists(directory):
        os.makedirs(directory)
