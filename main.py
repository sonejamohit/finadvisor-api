from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:4200",  # Add the URL of your Angular application
]

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict this to specific HTTP methods
    allow_headers=["*"],
)

# Load your trained model for profit prediction
with open('profit_pred.pkl', 'rb') as file:
    profit_model = joblib.load(file)

# Load your trained model for bankruptcy prediction
with open('logreg_model.pkl', 'rb') as model_file:
    logreg_model = joblib.load(model_file)

class InputData(BaseModel):
    r_and_d_spend: float
    marketing_spend: float

class PredictionResult(BaseModel):
    prediction: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare input data for profit prediction
        input_data = [[data.r_and_d_spend, data.marketing_spend]]

        # Make predictions
        prediction = profit_model.predict(input_data)
        print(prediction)
        # Return the result
        return PredictionResult(prediction=float(prediction))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FeatureInput(BaseModel):
    net_value_growth_rate: float
    borrowing_dependency: float
    net_income_to_stockholders_equity: float
    persistent_eps_last_four_seasons: float
    net_profit_before_tax_paid_in_capital: float
    per_share_net_profit_before_tax: float
    interest_bearing_debt_interest_rate: float
    degree_of_financial_leverage: float
    net_worth_assets: float
    net_value_per_share: float
    roa_c_before_interest: float
    cash_current_liability: float
    cash_total_assets: float
    continuous_interest_rate_after_tax: float
    debt_ratio_percent: float
    working_capital_equity: float
    total_debt_total_net_worth: float
    interest_coverage_ratio: float
    interest_expense_ratio: float
    net_income_to_total_assets: float


@app.post("/predict-bankruptcy")
async def predict_bankruptcy(features: FeatureInput):
    # Convert input features to a numpy array
    feature_values = np.array([val for val in features.dict().values()]).reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = logreg_model.predict(feature_values)

    # Return the prediction result
    if prediction[0] == 1:
        return {"prediction": "The model predicts bankruptcy."}
    else:
        return {"prediction": "The model predicts no bankruptcy."}
