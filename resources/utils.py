import os.path
import numpy as np
import keras
from keras.models import load_model

from skimage.io import imread
from skimage.transform import resize

# Path to saved classifier
my_path = os.path.abspath(os.path.dirname(__file__))
BrainTumorClassifier = os.path.join(my_path, "../static/BrainTumorClassifier/BrainTumorClassifier.h5")


# ALLOWED_EXTENSIONS = set(['jpeg','jpg'])
# def allowed_file(filename):
#     """Only .jpg files allowed"""
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = {'jpeg', 'jpg','JPEG','JPG'}

def allowed_file(filename):
    """Only .jpg files allowed"""
    file_extension = filename.rsplit('.', 1)[1]
    print(f"File extension: {file_extension}")
    return '.' in filename and file_extension in ALLOWED_EXTENSIONS



def image_classification(image):
    """Apply image classifier"""
    # clear Tensor session to avoid error
    keras.backend.clear_session()
    # load saved model
    image_classifier = load_model(BrainTumorClassifier)
    # prepare labels
    class_labels = {0: 'No', 1: 'Yes'}
    # read photo & transform it into array
    img = imread(image)
    print(f"Image shape: {img.shape}")
    img = resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img / 255.0
    # Predict class
    prediction = image_classifier.predict(img)
    percent_values = prediction.tolist()
    print(f"Raw prediction values: {prediction}")

    # Display labels
    guess = class_labels.get(int(prediction[0][0]), 'Unknown Label')
    print(f"Predicted label: {guess}")


    
    # for website display
    guess = class_labels.get(int(prediction[0][0]), 'Unknown Label')
    return guess


