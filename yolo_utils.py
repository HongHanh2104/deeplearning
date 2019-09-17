import colorsys
import imghdr
import os
import random
from keras import backend as K
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def read_classes(classes_path):
    try:
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    except FileNotFoundError:
        return -1

def read_anchors(anchors_path):
    try:
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
    except FileNotFoundError:
        return -1

def convert_color_space(image, type_color):
    """
    This function in order to convert a BGR image into other color spaces.
    
    Arguments:
    ----------
    image -- an array 
        OpenCV's format, default: BGR color space.
    type_color -- str
        other color spaces. 

    Returns:
    --------
    returned_image -- an array
        converted into another color space.
    """
    if(type_color == "RGB"):
        returned_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif(type_color == "gray"):
        returned_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif(type_color == "HSV"):
        returned_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return returned_image

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(image, model_image_size):
    temp = Image.open(image)
    #print("temp: ", temp)
    resized_image_temp = temp.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    #resized_image = cv2.resize(image, (608, 608), interpolation = cv2.INTER_CUBIC)
    image_data = np.array(resized_image_temp, dtype = 'float32')
    #print("resized_image: \n \n \n \n", image_data)
    image_data /= 255.
    image_data = (image_data, 0)  # Add batch dimension.
    return temp, image_data

def preprocess_image_cv2(image, model_image_size):
    """
    This function in order to resize input image into (608, 608), and add batch dimensions.
    
    Arguments:
    image -- an array - OpenCV's format.
    model_image_size -- (608, 608)

    Returns:
    image -- the input image
    image_data -- the preprecessed image
    """
    
    #image = Image.fromarray(image)
    #resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    resized_image = cv2.resize(image, (608, 608))
    image_data = np.array(resized_image, dtype = 'float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    #font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf', size = np.floor(3e-2 * image.shape[1] + 0.5).astype('int32'))
    #thickness = (image.size[0] + image.size[1]) // 300
    thickness = (image.shape[0] + image.shape[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {} {:.2f}'.format(i, predicted_class, score)
        '''
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline = colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill = colors[c])
        draw.text(text_origin, label, fill = (0, 0, 0), font = font)
        del draw
        '''
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        #print(i, (left, top), (right, bottom))
        if top - label_size[1] >= 0:
            #text_origin = np.array([left, top - label_size[1]])
            text_origin = (left, top - label_size[1])
        else:
            #text_origin = np.array([left, top + 1])
            text_origin = (left, top + 1)
       
        for i in range(thickness):
            cv2.rectangle(image, (left + i, top + i), (right - i, bottom - i), colors[c])
        cv2.rectangle(image, text_origin, (text_origin[0] + label_size[0], text_origin[1] + label_size[1]), colors[c], -1)
        cv2.putText(image, label, (text_origin[0], text_origin[1] + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


