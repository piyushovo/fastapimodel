from fastapi import FastAPI
from fastapi.responses import JSONResponse

import pickle
import pandas as pd
from user_input import UserInput
# import the ml model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()



#human readable
@app.get('/')
@app.get("/home")
def read_root():
    return {'message': 'INSURANCE PREMIUM PREDICTION API MODEL RANDOM FOREST welcomes you'}

#health check machine readable by aws /kubernetes
@app.get('/health')
def health_check():
    return {'status': 'OK ',
    'version': '1.0.0',
    'model': 'Random Forest Classifier'
    }





@app.post('/predict')#_________creating the route endpoint for fastapi_________________________________________________________

def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    pred_class = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]  # if your model supports probabilities
    class_labels = model.classes_.tolist()

    # Build response
    response = {
        "predicted_category": pred_class,
        "confidence": float(max(proba)),
        "class_probabilities": dict(zip(class_labels, proba))
    }

    return JSONResponse(status_code=200, content={"response": response})