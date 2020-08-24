from typing import List
from fastapi import FastAPI,Body
from joblib import load

MLAsAPI_app = FastAPI()


@MLCalAsAPI_app.post("/predict/")
def predictS(a:int = Body(...),b:int = Body(...),c:int = Body(...)):

    model_file = "/home/kali/Downloads/fast_api/MLCalAsAPI/model.pkl"

    predict = load(model_file) # Load "model.pkl"
    print ('Model loaded')

    if predict:
        X_TEST = [[ a, b, c ]]
        # Predict the result of X_TEST which holds testing data 
        outcome = predict.predict(X = X_TEST) 

        # Predict the coefficients 
        coefficients = predict.coef_ 

        # Print the result obtained for the test data 
        # print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients)) 


        return {"outcome": str(outcome)}
	