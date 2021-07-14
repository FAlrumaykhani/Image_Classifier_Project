# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:05:35 2021

@author: Firas Alrumaykhani
"""
import argparse
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings("ignore")


def process_image(image):
    image_size = 224
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
        
    return image



def predict(image_path, model_path, json_file = None ,top_k = 5):
    
    im = Image.open(image_path)
    im_np = np.asarray(im)

    preprocessed_im = process_image(im_np)
    preprocessed_im = np.expand_dims(preprocessed_im, axis=0)

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    probab = model.predict(preprocessed_im)

    probab, img_lab = tf.math.top_k(input = probab, k = int(top_k))
    
    if type(json_file) != type(None):
        with open(json_file, 'r') as f:
            class_names = json.load(f)
    
        # Note the +1 since the labeling start from 1 in json
        img_lab = [class_names[str(x+1)] for x in img_lab.numpy().tolist()[0]]
    
    else:
        img_lab = img_lab.numpy().tolist()[0]
        
    probab_approx = [round(x,6) for x in probab.numpy().tolist()[0]]
    
    return probab_approx, img_lab



parser = argparse.ArgumentParser(description = "Flowers Prediction app")
parser.add_argument("image_path", default="")
parser.add_argument("saved_model", default="")
parser.add_argument("--top_k", default=5, required= False )
parser.add_argument("--category_names", default="label_map.json", required= False )

args = parser.parse_args()

prob, label = predict(image_path = args.image_path, model_path = args.saved_model,
        json_file = args.category_names, top_k= args.top_k)

print("Image possible labels are {} with the following probabilities {}".format(label, prob))