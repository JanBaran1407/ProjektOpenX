from consts import DATASET_PATH
import pandas

def load_dataset():
    dataset_df = pandas.read_csv(DATASET_PATH, header=None)
    return dataset_df.head(1000)