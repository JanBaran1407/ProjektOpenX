from utils import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from consts import RANDOM_STATE, TEST_SIZE, MODELS_SAVE_PATH
from pickle import dump
from utils import load_model
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from utils import split


def train_sk_model(clf, clf_name, X, y):
    print(f"Training {clf_name}")
    train_set = split(X, y)['train']
    clf.fit(train_set[0], train_set[1])
    dump(clf, open( f"{MODELS_SAVE_PATH}/{clf_name}.pkl", 'wb'))

def train_sklearn_models():
    print("Loading dataset")
    dataset = load_dataset()
    y = dataset.pop(54)

    neighbors_clf = KNeighborsClassifier(n_neighbors= 5)
    train_sk_model(neighbors_clf, 'knn', dataset, y)

    neighbors_clf = RandomForestClassifier(n_estimators= 100)
    train_sk_model(neighbors_clf, 'rf', dataset, y)

def get_model(name = 'all'):
    if name == 'all':
        neighbors_clf = load_model('knn')
        forest_clf = load_model('rf')
        tf_network = load_model('tf_model')

        return neighbors_clf, forest_clf, tf_network
    else:
        return  load_model(f'{name}')


def create_model(units=128, dropout_rate=0.2, optimizer='adam'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units, activation='relu', input_shape=(54,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units // 2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_tf_model():
    print("Loading dataset")
    dataset = load_dataset()
    y = dataset.pop(54)

    print("Creaing model")
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=0)

    param_grid = {
        'units': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4],
        'optimizer': ['adam', 'rmsprop']
    }

    print("Performing grid search")
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
    train_set = split(dataset, y)['train']
    grid_result = grid.fit(train_set[0], train_set[1])
    
    best_model = grid_result.best_estimator_
    dump(best_model, open( f"{MODELS_SAVE_PATH}/tf_model.pkl", 'wb'))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    grid_iteration_dict = [f"U: {grid_iteration['units']}| DR: {grid_iteration['dropout_rate']}|O: {grid_iteration['optimizer']}" for grid_iteration in params]
    plt.errorbar(grid_iteration_dict, means, yerr=stds)
    plt.title('Grid search results')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('grid_search_results.png')
    plt.show()

def evaluate_all_models():
    knn, rf, dnn  = get_model(name = 'all')
    dataset = load_dataset()
    y = dataset.pop(54)

    test_set = split(dataset, y)['test']
    scores = {
        "K_nearest_neighbours": knn.score(test_set[0], test_set[1]),
        "Random forest": rf.score(test_set[0], test_set[1]),
        "Deep neural network": dnn.score(test_set[0], test_set[1]),
    }

    info = []
    for score in sorted(scores.items(), key=lambda x:x[1], reverse=True):
        info.append(f'{score[0]} got {int(score[1] * 100)}% accuracy.')

    return info