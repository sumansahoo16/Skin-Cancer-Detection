
import torch
import PIL, io, timm
import numpy as np, base64
from flask import render_template

classes = ['Benign Cancer',
            'Melanocytic Nevi',
            'Melanoma ',
            'Benign Keratosis',
            'Irrelevant',
            'Basal Cell Carcinoma ', 
            'Actinic keratoses',
            'Vascular Lesions',
            'Dermatofibroma ']

def preprocess_img(img):
    img = img.resize((224,224))

    img =  np.asarray(img)
    img = img / 255
    img = img.transpose(2, 0, 1)

    X = torch.tensor(img)
    X = X.unsqueeze(0)
    return X.float()

def encode_img(img):
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def model_predict(img, model):

    img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
    X = preprocess_img(img)

    out = model(X).detach()
    out = out.softmax(1)[0].numpy()
    
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in out]]

    pred_probs = sorted(zip(classes, map(str, formatted_outputs)),
                        key=lambda p: p[1], reverse=True)
	
    result = {"class":pred_probs[0][0], "probs":pred_probs, "image":encode_img(img)}
    return render_template('result.html', result=result)

class SkinCancerModel(torch.nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=False)
        n_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(n_features, 9)

    def forward(self, x):
        x = self.model(x)
        return x