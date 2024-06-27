from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import RedirectResponse
from function import InputData, predict, training, model

app = FastAPI(
    title="StreamGuesser - AI",
    description="Analysez la musique comme jamais auparavant avec StreamGuesser's AI",
    version="1.0.0"
)


@app.post(path="/training",
          tags=["Model"],
          summary="Entraîner le modèle",
          description="Enrichir le model avec de nouvelles donnees.")
async def training_route(file: UploadFile = File(...)):
    return training(file)


@app.post(path="/predict",
          tags=["Model"],
          summary="Prédiction des streams",
          description="Faire une prediction sur le nombre de stream potentiel sur une musique.")
async def predict_route(input_data: InputData):
    prediction = predict(input_data)
    return prediction


@app.get(path="/model",
         tags=["HuggingFace"],
         summary="Parler avec un GPT-2",
         description="Parler avec une IA capable de comprendre et de générer du texte.")
async def model_route(message: str):
    return model(message)


@app.middleware("http")
async def handle_docs_redirect(request: Request, call_next):
    path = request.url.path
    if path != "/" and not path.startswith("/docs") and not path.startswith("/openapi.json") and not path.startswith(
            "/training") and not path.startswith("/predict") and not path.startswith("/model"):
        return RedirectResponse(url="/docs", status_code=302)
    return await call_next(request)
