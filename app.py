from cProfile import label
from crypt import methods
from cv2 import CascadeClassifier, imwrite, waitKey
from flask import Flask,render_template,request,Response
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from PIL import Image,ImageOps
import numpy as np
from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import mediapipe as mp
import time
# Load the model


app= Flask(__name__)

model = load_model('keras_model.h5')
mpbody = mp.solutions.hands
body = mpbody.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

f = open('labels.txt', 'r')
classNames = f.read().split('\n')

@app.route('/',methods=['GET'])
def main():
    return render_template('main.html')

@app.route('/upload', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def detection():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    x=imagefile.filename
    

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1,224, 224,3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(image_path)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction1 = model.predict(data)
    classID = np.argmax(prediction1)
    className = classNames[classID]
    
    return render_template('index.html',prediction=className)
    

    # return render_template('index.html',prediction=prediction1)



def generate_frames():
    camera=cv2.VideoCapture(0)
    while True:
        time.sleep(5)
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            
            ret,buffer=cv2.imencode('.jpg',frame)
            imwrite("./static/buffer.jpg",frame)
            frame=buffer.tobytes()
            
            

        # yield(b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        return ('buffer.jpg')
        





        

@app.route('/camera',methods=['GET'])
def video_feed():
    generate_frames()
    return render_template('camera.html', user_image='buffer.jpg')

@app.route('/cameraans')
def videoans():
    data = np.ndarray(shape=(1,224, 224,3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open('./static/buffer.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction1 = model.predict(data)
    classID = np.argmax(prediction1)
    className = classNames[classID]
    
    return render_template('ans.html',prediction=className)
    

if __name__ == '__main__':
    app.run(debug=True,port=3000)