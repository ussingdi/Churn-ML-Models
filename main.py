from fastapi import FastAPI
import pickle
import  pandas as pd
app = FastAPI()

with open('xgb_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)



def preprocess_data(customer_data: dict):
    input_dict = {
        'CreditScore': customer_data['CreditScore'],
        'Age': customer_data['Age'],
        'Tenure': customer_data['Tenure'],
        'Balance': customer_data['Balance'],
        'NumOfProducts': customer_data['NumOfProducts'],
        'HasCrCard': customer_data['HasCrCard'],
        'IsActiveMember': customer_data['IsActiveMember'],
        'EstimatedSalary': customer_data['EstimatedSalary'],
        'Gender_Male': 1 if customer_data['Gender'] == 'Male' else 0,
        'Gender_Female': 1 if customer_data['Gender'] == 'Female' else 0,
        'Geography_France': 1 if customer_data['Geography'] == 'France' else 0,
        'Geography_Germany': 1 if customer_data['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if customer_data['Geography'] == 'Spain' else 0,
    }
    return pd.DataFrame([input_dict])

@app.post("/predict")
def predict_churn(customer_data: dict):
    processed_data = preprocess_data(customer_data)
    prediction = loaded_model.predict(processed_data)
    probability = loaded_model.predict_proba(processed_data)[:, 1]
    return prediction, probability




async def predict_churn_endpoint(customer_data: dict):
    prediction, probability = predict_churn(customer_data)
    return {"prediction": prediction.tolist(), "probability": probability.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

