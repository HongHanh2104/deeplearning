import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imread
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import cv2
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import *
from keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        box_confidence -- tensor of shape (19, 19, 5, 1)
        boxes -- tensor of shape (19, 19, 5, 4)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Compute box scores
    box_scores = box_confidence * box_class_probs
    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1)
    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    prediction_mask = box_class_scores >= threshold

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    boxes = tf.boolean_mask(boxes, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    return scores, boxes, classes

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Get the coordinates of the intersection of two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    inter_area = (x2 - x1)*(y2 - y1)
    
    # Calculate the Union area: Union(A, B) = A + b - Inter(A, B)
    box1_area = (box1[3] - box1[1])*(box1[2] - box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2] - box2[0])
    union_area = (box1_area + box2_area) - inter_area
    
    # Compute IOU
    iou = inter_area / union_area
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, 
                iou_threshold = 0.5):
    
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,)
        output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4)
        output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,)
        output of yolo_filter_boxes()
    max_boxes -- integer
        maximum number of predicted boxes you'd like
    iou_threshold -- real value
        "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None)
        predicted score for each box
    boxes -- tensor of shape (4, None)
        predicted box coordinates
    classes -- tensor of shape (, None)
        predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less 
        than max_boxes. Note also that this function will transpose 
        the shapes of scores, boxes, classes. This is made for convenience.
    """
    max_boxes_tensor = K.variable(max_boxes, dtype = 'int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    # get the list of indices corresponding to boxes
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    # select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes = 10, score_threshold = .6, iou_threshold = .5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes 
    along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    # Rerform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    # Perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

def predict(sess, image_file, yolo_model, class_names, scores, boxes, classes):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- the tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess the image
    image, image_data = preprocess_image_cv2(image_file, model_image_size = (608, 608))
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input: image_data, K.learning_phase(): 0})
    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    #image.save(os.path.join("out", image_file), quality=90)
    
    # Display the results in the notebook
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    
    #imshow(output_image)
    #plt.show()
    return image, out_scores, out_boxes, out_classes

def detect_image(sess, image_input, yolo_model, class_names, anchors):
    """ Detect an image

    Parameters
    ----------
    sess: tensor
        the tensorflow/Keras session containing the YOLO graph.
    image_input: a array
        OpenCV's format in RGB color space.
    yolo_model:
        the model returned by load_model function.
    class_name: str
        the name lists.
    anchors: str
        the anchor lists.

    Returns
    -------
    output_image: str
        still the direction of image.
    class_names: str
        the name lists read from file.
    anchors: str
        the anchor lists read from file.
    """
    image_shape = float(image_input.shape[0]), float(image_input.shape[1])
        
    
    # Convert output of the model to usable bounding box tensors
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
 
    # Filtering boxes
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    output_image, out_scores, out_boxes, out_classes = predict(sess, image_input, yolo_model, class_names, scores, boxes, classes)
    
    return output_image, out_scores, out_boxes, out_classes 
    

def detect_video(sess, video_input, yolo_model, class_names, anchors):
    """
    This function in order to detect video.
    
    Arguments:
    ----------
    sess: tensor
        the tensorflow/Keras session containing the YOLO graph.
    video_input: str
        a path to video.
    yolo_model:
        the model returned by load_model function.
    class_name: str
        the name lists.
    anchors: str
        the anchor lists.

    Returns:
    --------
    returned_image -- an array
        converted into another color space.
    """
    cap = cv2.VideoCapture(video_input)
    if (cap.isOpened() == False):
        print("The direction of video is wrong.")
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert_color_space(frame, "RGB")
        out_image, out_scores, out_boxes, out_classes = detect_image(sess, frame, yolo_model, class_names, anchors)
        opencvImage = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', opencvImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_frame(sess, folder_input, folder_output, yolo_model, class_names, anchors):
    input_files = os.listdir(folder_input)
    
    for files in input_files:
        filename, extension = os.path.splitext(files)
        full_input_filename = folder_input + files
        img = cv2.imread(full_input_filename)
        img = convert_color_space(img, "RGB")
        out_image, out_scores, out_boxes, out_classes = detect_image(sess, img, yolo_model, class_names, anchors)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        full_output_filename = folder_output + "detected" + filename + extension
        cv2.imwrite(full_output_filename, out_image)
        #full_file_name = os.path.join(src, file_name)
        #if os.path.isfile(full_file_name):


def pre_model(class_dir, anchor_dir):
    """ Read necessary files: image file, class_name and anchor.

    Parameters
    ----------
    class_dir: str
        the path of class_name file.
    anchor_dir: str
        the path of anchor file.
    
    Returns
    -------
    class_names: str
        the name lists read from file.
    anchors: str
        the anchor lists read from file.
    """
    class_names = read_classes(class_dir)
    anchors = read_anchors(anchor_dir)
    return class_names, anchors

def showImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', default = '/home/honghanh/Documents/experiment/input/test.jpg', type = str, 
                        help = 'path to image \n' )
    parser.add_argument('-video_dir', default = 'input/cam3.mp4', type = str, 
                        help = 'path to video \n' )
    parser.add_argument('-folder_dir', default = 'input/frame/', type = str, 
                        help = 'path to video \n' )   
    parser.add_argument('-output_dir', default = '/home/honghanh/Documents/experiment/output/', type = str, 
                        help='Output anchor directory \n' )                                       
    parser.add_argument('-model_dir', default = '/home/honghanh/Documents/experiment/model_data/full_yolo.h5', type = str,
                        help = 'path to model \n')                      
    parser.add_argument('-class_dir', default = '/home/honghanh/Documents/experiment/model_data/coco_classes.txt', type = str,
                        help = 'path to class name \n')
    parser.add_argument('-anchor_dir', default = '/home/honghanh/Documents/experiment/model_data/anchors5.txt', type = str, 
                        help = 'path to anchors \n')
    parser.add_argument('-type', default = 'image', type = str, 
                        help = 'detect image or detect video \n')
    
    
    args = parser.parse_args()


    sess = K.get_session()

    class_names, anchors = pre_model(args.class_dir, args.anchor_dir)
        
    # Load a pretrained model
    yolo_model = load_model(args.model_dir)

    # BGR image 
    image_input = cv2.imread(args.img_dir)
    image_input = convert_color_space(image_input, "RGB")
    
    # Run the graph on an image
    if(args.type == "image"):
        out_image, out_scores, out_boxes, out_classes = detect_image(sess, image_input, yolo_model, class_names, anchors)
        showImage(out_image)
    elif (args.type == "video"):
        detect_video(sess, args.video_dir, yolo_model, class_names, anchors)
    elif(args.type == "folder"):
        detect_frame(sess, args.folder_dir, args.output_dir, yolo_model, class_names, anchors)
        
if __name__=="__main__":
    main(sys.argv)