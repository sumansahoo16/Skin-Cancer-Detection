
import os
from pathlib import Path
from io import BytesIO
import base64
import requests

#import timm 



import numpy as np


import torch
import torch.nn as nn


# Flask utils
from flask import Flask, redirect, url_for, render_template, request
import PIL
from PIL import Image 

# Define a flask app
app = Flask(__name__)

NAME_OF_FILE = 'model23.pth' # Name of your exported file
PATH_TO_MODELS_DIR = Path('') # by default just use /models in root dir

classes = ['bgn', 'nv', 'mel',
           'bkl', 'othr', 'bcc', 'akiec', 'vasc', 'df']


class SkinCancerModel(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=False)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 9)

    def forward(self, x):
        x = self.model(x)
        return x

def open_image(fn, convert_mode:str='RGB'):

    x = PIL.Image.open(fn).convert(convert_mode)
    x = x.resize((224,224)) 
    x =  np.asarray(x)
    x = x / 255
    x = x.transpose(2, 0, 1)
    x = torch.tensor(x)
    x = x.unsqueeze(0)
    return x.float()


def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    learn = SkinCancerModel()
    learn.load_state_dict(torch.load(learner_name_to_load, map_location=torch.device('cpu')))
    return learn

learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

def image2np(image):
    "Convert from torch style `image` to numpy/matplotlib style"
    image = image.squeeze(0)
    res = image.cpu().permute(1,2,0).numpy()
    return res[...,0] if res.shape[2]==1 else res

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
	
def model_predict(img):
    img = open_image(BytesIO(img))

    outputs = learn(img).detach()
    outputs = outputs.softmax(1)[0].numpy()


    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in outputs]]

    pred_probs = sorted(
            zip(classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
	
    img_data = encode(img)
    result = {"class":pred_probs[0], "probs":pred_probs, "image":img_data}
    return render_template('result.html', result=result)
   

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
            preds = model_predict(img)
            return preds
    return 'OK'
	
@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'
    

if __name__ == '__main__':
    app.run()
