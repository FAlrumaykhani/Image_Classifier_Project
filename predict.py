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


def process_image(image):
    image_size = 224
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
        
    return image

def predict(image_path, model):
    
    im = Image.open(image_path)
    im_np = np.asarray(im)

    preprocessed_im = process_image(im_np)
    preprocessed_im = np.expand_dims(preprocessed_im, axis=0)

    probab = model.predict(preprocessed_im)

    probab, img_lab = tf.math.top_k(input = probab, k= 5)
    
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    # Note the +1 since the labeling start from 1 in json
    img_lab_txt = [class_names[str(x+1)] for x in img_lab.numpy().tolist()[0]]
    probab_approx = [round(x,6) for x in probab.numpy().tolist()[0]]
    
    return probab_approx, img_lab_txt



parser = argparse.ArgumentParser(description = "Flowers Prediction app")
parser.add_argument("image_path")
parser.add_argument("saved_model")

args = parser.parse_args()

predict(args.image_path, args.saved_model)








