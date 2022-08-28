from flask import Flask, render_template, request
import yaml
import numpy as np
import keras
from keras.preprocessing import image
import json
from json import JSONEncoder
from urllib.request import urlretrieve
import requests


app = Flask(__name__)

urlretrieve('https://storage.googleapis.com/bangkit2022-akrab.appspot.com/model/alphabet_model.h5','/tmp/model.h5')
model1 = keras.models.load_model('/tmp/model.h5')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def predict_image(path):
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model1.predict(images, batch_size=50)
    return classes


@app.route("/predict/huruf", methods=['POST'])
def output():
    
    if request.method == 'POST':
        url = request.args.get('url')
        # Download the image
        urlretrieve(url, '/tmp/Cracked-1.jpg')

        img_path = '/tmp/Cracked-1.jpg'

        p = predict_image(img_path)
        # print(p)
        i = np.argmax(p[0])
        # print(i)
        index =i
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 
               'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S',  
               'T', 'U', 'V', 'W', 'X', 'Y']
        result = classes[index]
        return {
            "code_status": 200,
            "message": "Success",
            "predict": result
        }
    return result

@app.route('/', methods=['GET'])
def index():
    return 'Akrab, Comunicate Better'


if __name__ == '__main__':
    app.run(debug=True)