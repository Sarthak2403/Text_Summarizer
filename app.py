# from fastapi import FastAPI
# import uvicorn
# import sys
# import os
# from fastapi.templating import Jinja2Templates
# from starlette.responses import RedirectResponse
# from fastapi.responses import Response
# from text_Summarizer.pipeline.prediction import PredictionPipeline


# text:str = "What is Text Summarization?"

# app = FastAPI()

# @app.get("/", tags=["authentication"])
# async def index():
#     return RedirectResponse(url="/docs")



# @app.get("/train")
# async def training():
#     try:
#         os.system("python main.py")
#         return Response("Training successful !!")

#     except Exception as e:
#         return Response(f"Error Occurred! {e}")
    



# @app.post("/predict")
# async def predict_route(text):
#     try:

#         obj = PredictionPipeline()
#         text = obj.predict(text)
#         return text
#     except Exception as e:
#         raise e
    

# if __name__=="__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from text_Summarizer.pipeline.prediction import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)

# Define the homepage route
@app.route('/')
def index():
    return redirect(url_for('home'))

# Route to display the homepage with input form
@app.route('/home', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form["text_input"]
        if request.form.get("train_button") == "Train Model":
            # Trigger the model training
            os.system("python main.py")
            return render_template("home.html", result="Model training started!")
        elif request.form.get("predict_button") == "Test Prediction":
            # Trigger the prediction
            obj = PredictionPipeline()
            prediction_result = obj.predict(text_input)
            return render_template("home.html", result=f"Prediction: {prediction_result}")
    return render_template("home.html", result="")

# Define a custom error page for handling exceptions
@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html", error_message=str(error)), 500

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
