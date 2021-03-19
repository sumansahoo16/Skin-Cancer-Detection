
import torch
import requests
from utils import SkinCancerModel, model_predict
from flask import Flask, render_template, request




model = SkinCancerModel()
model.load_state_dict(torch.load('abc_24.pth', map_location=torch.device('cpu')))

app = Flask(__name__)

@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
        # Make prediction
            preds = model_predict(img, model)
            return preds
    return 'OK'


@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content, model)
            return preds
    return 'OK'

if __name__ == '__main__':
    app.run()
