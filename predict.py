import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer

import pickle

# def load(filename):
#     with open(filename, 'rb') as f_in:
#         return pickle.load(f_in)
dv = DictVectorizer(sparse=False)

# # dv = load('dv.bin')
# model = load('pipeline.bin')
model_file = './pipeline.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI()

class UserProfile(BaseModel):
    relative_compactness: float
    surface_area: float
    wall_area: float
    roof_area: float
    overall_height: float
    orientation: int
    glazing_area: float
    glazing_area_distribution: int

@app.get('/')
async def Index():
    return {"Predict energy efficiency of buildings, go to /docs"}


@app.post('/predict', status_code=200)
async def predict(user_profile: UserProfile):
    application_data = user_profile.dict()
    prediction = pipeline.predict(application_data)
    

    result = prediction[0]
    print(result)
    print(result[0]+result[1])
    if (result[0]+result[1]) > 65:
        return {"total_load" : "high"}
    elif (result[0]+result[1]) > 29:
        return {"total_load" : "average"}
    else:
        return {"total_load" : "low"}
    
    return {"[heating_load, cooling_load]" : result}


