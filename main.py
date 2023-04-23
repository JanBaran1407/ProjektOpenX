from models import train_sklearn_models, train_tf_model, evaluate_all_models, get_model
from consts import VALUE_TO_NAME_MAP
from fastapi import FastAPI, Request
from utils import ModelNotFoundException
app = FastAPI()

@app.get("/train")
def train():
    train_sklearn_models()
    train_tf_model()
    return "Trained all models."

@app.get("/evaluate")
def evaluate():
    info = evaluate_all_models()
    return info

@app.post("/predict")
async def get_body(request: Request):
    body = await request.json()
    clf_name = body.get('clf', None)
    data = body.get('data', None)

    if not clf_name or not body:
        return "Please provide valide classifier name and values"
    
    try:
        values = [int(value) for value in data.split(',')]
        model = get_model(name = clf_name)
        result = model.predict([values])
        return VALUE_TO_NAME_MAP[result[0]]
    except ModelNotFoundException:
        return "Sorry, provided model does not exist."
