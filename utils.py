from consts import DATASET_PATH, RANDOM_STATE,TEST_SIZE,MODELS_SAVE_PATH
import pandas
from pickle import load
from sklearn.model_selection import train_test_split
import os

class ModelNotFoundException(Exception):
    pass

def load_dataset():
    dataset_df = pandas.read_csv(DATASET_PATH, header=None)
    return dataset_df.head(1000)

def load_model(file_name):
    path = f"{MODELS_SAVE_PATH}/{file_name}.pkl"
    if not os.path.exists(path):
        raise ModelNotFoundException

    return load(open(path, 'rb'))

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
    }