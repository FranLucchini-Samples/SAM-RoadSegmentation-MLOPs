from statistics import mean 
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/healthcheck")
def healthcheck():
    # TODO: return status 200
    return {"Hello": "World"}


@app.post("/predict")
def predict(body: dict):
    print(body)
    inferences = body["instances"]
    mean_list = []
    for inf in inferences:
        mean_list.append(mean(inf['values']))
    
    parameters = body["parameters"]
    return {
        "predictions": [{
            "mean": mean_list, 
            "config": parameters['param1']
        }]
        }


if __name__ == "__main__":
    uvicorn.run("FastAPI-Test-inference_server:app", host="0.0.0.0", port=8000, reload=True,)