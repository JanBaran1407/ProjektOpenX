from utils import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pickle import dump


def train_sk_model(model,x_train,x_test,y_train,y_test,file_name):
    model.fit(x_train,y_train)
    dump(model,open(f"{file_name}.pkl","wb"))
    score=model.score(x_test,y_test)
    return score

def train_sk_models():
    dataset=load_data()
    y=dataset.pop(54)
    x_train,x_test,y_train,y_test=train_test_split(dataset,y,test_size=0.3,random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    print(train_sk_model(knn,x_train,x_test,y_train,y_test,"knn"))

    rf = RandomForestClassifier(n_estimators=100)
    print(train_sk_model(rf,x_train,x_test,y_train,y_test,"rf"))
