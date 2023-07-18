
from flask import Flask, render_template
from flask_socketio import SocketIO

from io import BytesIO
import io
import cv2
from cv2 import dnn_superres
from PIL import Image
import PIL
import base64
import numpy as np


# initialize super resolution object
sr = dnn_superres.DnnSuperResImpl_create()

# read the model
path = 'LapSRN_x2.pb'
sr.readModel(path)

# set the model and scale
sr.setModel('lapsrn', 2)



app = Flask(__name__)
app.config['SECRET_KEY'] = "bruh"
socket = SocketIO(app)


@app.route('/')
def main():
    return render_template("index.html")

@socket.on('message')
def handlemsg(msg):
    im_bytes = base64.b64decode(msg)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    

    # upsample the image
    upscaled = sr.upsample(img)
    
    _, im_arr = cv2.imencode('.png', upscaled)  # im_arr: image in Numpy one-dim array format.
    encoded_img_data = base64.b64encode(im_arr)
    #print(type(encoded_img_data))

    socket.send(encoded_img_data.decode('utf-8'))


if __name__ == "__main__":
    socket.run(app)
