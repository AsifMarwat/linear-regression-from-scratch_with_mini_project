from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import numpy as np
import os
import sys

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Load the trained model
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
sys.path.append(ROOT_DIR)
import os
#  Load model globally — use relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "icecream_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f) 


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request, temp: float = Form(...)):
    # Reshape input and predict
    input_data = np.array([[temp]])
    prediction = model.predict(input_data)[0][0]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": round(prediction, 2)
    })

# Optional: For dev run
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
