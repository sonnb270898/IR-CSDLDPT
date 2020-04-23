from model import *
from flask import Flask,request,render_template
import os
from region_segmentation import extractobj
import cv2
app = Flask(__name__,
            static_url_path='', 
            static_folder='data',
            template_folder='templates')
app.config["IMAGE_UPLOADS"] = "/home/son/Desktop/DPT/public"

@app.route("/")
def index():
    return render_template('hello.html')

@app.route("/upload-image", methods=["POST"])
def upload_image():

    if request.method == "POST":

        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            # extractobj('/home/son/Desktop/DPT/public/' + image.filename)
            # originName = image.filename.split(".")[len(image.filename.split(".")) - 2]
            # extensionName = image.filename.split(".")[-1]
            # print('out/'+originName+'_cut.'+extensionName)
            # print('#'*50)
            # im = cv2.imread('out/'+originName+'_cut.'+extensionName)
            im = cv2.imread('public/'+image.filename)
            return predict_image(im)
        return "2"

if __name__ == '__main__':
    app.run(threaded=False)
