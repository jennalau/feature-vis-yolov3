import os
import errno
from predict import read_fits, normalize_image, get_files, get_satnet_images_dir_list
import numpy as np
import csv
import json

from models.keras_yolo3 import Yolo
from visualize.vis_utils.helpers import *

import matplotlib.pyplot as plt
import collections
import cv2

from deep_vis_keras.guided_backprop import GuidedBackprop
from deep_vis_keras.saliency import GradientSaliency

from deep_vis_keras.visual_backprop import VisualBackprop
from deep_vis_keras.integrated_gradients import IntegratedGradients


def load_fits_input_images(input_path_list):
        """
        Arguments:
                input_path_list: list of paths to .fits files
        Returns:
                img_results: list of (1,512,512,1) images
        """

        img_results = []

        for image_path in input_path_list:
                image = read_fits(image_path)
                shape = image.shape

                # Reshape image to 4D (1,512,512,1)
                if len(shape) < 4:
                        image = np.expand_dims(image, axis=0)
                        image = np.expand_dims(image, axis=-1)

                # Normalize image
                image_n = normalize_image(image) 

                img_results.append(image_n)
                      
        return img_results



def convert_to_3_channel(image):
        """
        Convert 1 channel, grayscale channel ==> 3 channel, rgb image

        Arguments:
                image: (512,512,1) grayscale image
        Returns:
                image: (512,512,3) rgb image
        """
        # Make this an RGB on the appropriate scale/dtype
        if image.shape[-1] == 1:
                image = np.stack([image[:, :, -1], image[:, :, -1], image[:, :, -1]],
                                axis=-1)
        return image


def prettify_image(image):
        """
        Normalize image (mask generated after gradient adjustments)
        """
        if len(image.shape) == 3:
                if image.shape[-1] == 1:
                        image = convert_to_3_channel(image)

        image = np.sum(np.abs(image), axis=2)

        image_max = np.percentile(image, 99)
        image_min = np.min(image)

        image = (image - image_min) / (image_max - image_min)
        image = (255.0 * image).astype(np.uint8)

        return image


def draw_pred_boxes(image, pred_boxes, confidence_threshold, label_str="Satellite"):
        """
        Draw prediction boxes in green

        Arguments:
                image: (512,512,3) mask image
                pred_boxes: 2D np array of predictions,
                        each element = [ymin, xmin, ymax, xmax, object confidence, satellite confidence]
                confidence_threshold: confidence threshold for prediction boxes to draw
                label_str: class label

        Returns:
                image: image with prediction boxes drawn in green
        """
        h, w = image.shape[0], image.shape[1]
        cyan_color = (255,255,0) #BGR

        # Draw the predicted boxes
        for j in range(pred_boxes.shape[0]):
                box = pred_boxes[j, :]

                # Only draw the boxes that have a sufficiently high score
                if box[5] > confidence_threshold:
                        pt1 = (int(w * box[1]), int(h * box[0]))
                        pt2 = (int(w * box[3]), int(h * box[2]))
                        cv2.rectangle(image, pt1, pt2, cyan_color, 1)
                        cv2.putText(image, 
                                label_str + ' ' + "{:.3f}".format(box[5]), 
                                (int(w * box[1]), int(h * box[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.30, 
                                cyan_color, 1, cv2.LINE_AA)
        return image



def draw_gt_boxes(image, gt_boxes):
    """
    Draw ground truth boxes in white

    Arguments:
        image = (512,512,3) mask image
        gt_boxes = 2D np array of ground truth boxes, 
                   each element = [ymin, xmin, ymax, xmax]
    Returns:
        image: image with ground truth boxes drawn in white
    """
    h, w = image.shape[0], image.shape[1]
    green = (0,255,0)

    # Draw the ground truth boxes
    for j in range(gt_boxes.shape[0]):
        box = gt_boxes[j, :]
        pt1 = (int(w * box[1]), int(h * box[0]))
        pt2 = (int(w * box[3]), int(h * box[2]))
        cv2.rectangle(image, pt1, pt2, green, 1)
    
    return image



def mark_images(image, pred_boxes, gt_boxes, confidence=0.0, draw_pred=True, draw_gt=True, hist_eq=False):
        """
        Mark up images with prediction and ground truth boxes

        Arguments:
                image: (512,512,1) normalized image
                pred_boxes: 2D np array of predictions
                gt_boxes: 2D np array of ground truth box 
                          coords/objectness score/confidence score
                draw_pred: Bool to draw prediction boxes in blue
                draw_gt: Bool to draw ground truth boxes in green
                hist_eq: Bool to peform histogram equalization on normalized image
                
        Returns:
                image: marked up image with prediction & ground truth boxes
        """
        # Adjust image
        if hist_eq:
                image = image.astype(np.uint16)
                image = histogram_equalize(image, False) # returns a 3 channel image
                image = (image/256).astype(np.uint8)
        else:
                image = prettify_image(image) # returns 2D image

        # Reshape 
        if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

        if len(image.shape) == 3:
                if image.shape[-1] == 1:
                        image = convert_to_3_channel(image)

        # Draw prediction and/or ground truth boxes
        if draw_pred:
                image = draw_pred_boxes(image, 
                                        pred_boxes, 
                                        confidence,
                                        label_str="Satellite")
        if draw_gt:
                image = draw_gt_boxes(image, gt_boxes)
        
        return image

def save_to_png(image, dir_path, sat_img_name, grad_name, ext='.png', *args):
        """
        Save image (512,512,1) to filename.ext in highest quality

        Arguments:      
                mask_image: (512,512,1) mask image with adjusted gradients
                dir_path: path to Visualizations folder
                sat_img_name: name of satellite
                grad_name: name of backprop method
                *args: additional arguments to extend filename
        """
        filename = ''
        grad_name = grad_name.lower().replace(' ', '_')

        if not args:
                filename = sat_img_name + "_" + grad_name
        else:
                extended_filename = [a for a in args]
                extended_filename.insert(0, filename)
                filename = '_'.join(extended_filename)
        
        save_path = os.path.join(dir_path, filename + ext)
        print("Visualization resulting image saved to: ", save_path)

        image = (image).astype('uint8')
        png_compression = 0 # lowest compression, highest quality, largest file size
        cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        return save_path


def histogram_equalize(data, eightBits = True):
        """
        Histogram equalization for adjusting/enhancing image pixel intensities
        """
        sizes = np.shape(data)
        height = sizes[0]
        width = sizes[1]

        img = np.zeros([height,width,3])
        img = data

        if eightBits == True:
                depth = 256
        else:
                depth = 65536

        img_gray = img

        if eightBits == True:
                lut = histogram_clip(img_gray,0.1,0.999,depth,31,248,True)
        else:
                lut = histogram_clip(img_gray,0.2, 0.99,depth,31*256,248*256,False)

        img_equal_gray = apply_lut(img_gray,lut)

        img_equal_rgb = cv2.cvtColor(img_equal_gray, cv2.COLOR_GRAY2RGB)

        return img_equal_rgb
   

def histogram_clip(img, clip_lo, clip_hi, depth, L_l=31, L_h=248, eightBits = True):
        """
        Helper function for histogram equalization
        """

        hist,bins = np.histogram(img.flatten(),depth,[0,depth-1])

        cdf = hist.cumsum()
        cdf_n = cdf/ cdf.max()

        index_lo = np.argmax(cdf_n > clip_lo)
        index_hi = np.argmax(cdf_n > clip_hi)

        # create the lookup table:
        if eightBits == True:
                lut = np.zeros(depth,np.uint8)
        else:
                lut = np.zeros(depth,np.uint16)

        if(index_hi == index_lo):
                m = L_h - L_l
        else:
                m = (L_h - L_l)/(index_hi - index_lo)

        b = L_l - m*index_lo

        for i in np.arange(0,depth):
                if i < index_lo:
                        lut[i] = L_l
                elif i > index_hi:
                        lut[i] = depth-1
                else:
                        if eightBits == True:
                                lut[i] = (m*i + b).astype(np.uint8)
                                if lut[i] < L_l:
                                        lut[i] = L_l
                        else:
                                lut[i] = (m*i + b).astype(np.uint16)
                                if lut[i] < L_l:
                                        lut[i] = L_l

        return lut

def apply_lut(img,lut):
        """
        Helper function for histogram equalization
        """
        img_out = np.empty_like(img)
        img_out[:,:] = lut[img[:,:]]
        return img_out