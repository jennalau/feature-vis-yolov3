import matplotlib.pyplot as plt
import collections
import numpy as np
import cv2
import os

from models.keras_yolo3 import Yolo

from deep_vis_keras.guided_backprop import GuidedBackprop
from deep_vis_keras.saliency import GradientSaliency
from deep_vis_keras.visual_backprop import VisualBackprop
from deep_vis_keras.integrated_gradients import IntegratedGradients

from visualize.vis_utils.helpers import *
from visualize.vis_utils.image_processing import *

import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import save_model, load_model, Model
import tensorflow.compat.v1.keras.backend as K

def show_saliency_backprop(model, model_name, 
                            output_path, sat_img_name, 
                            input_image, preds, gt_coords, 
                            saliency_results, confidence):
    """ 
    Produce saliency maps on backpropagation method (guided backprop)

    Arguments: 
        model: training model or inference model
        model_name: name of model being used (train or infer)
        output_path: path to new 'Visualizations' directory
        sat_img_name: name of specific image file of a satellite 
                    (ex: sat_29155.0001.json, sat_img_name = sat_29155.0001)
        input_image: a single 4-dimensional normalized image (1,512,512,1 shape)
        preds: 2D numpy array of predictions
        gt_coords: 2D numpy array of ground truth box coordinates
        saliency_results: list of paths to the output PNGs
        confidence: confidence threshold [0,1]
    """
    print("Computing saliency maps (via backprop) ...")    

    backprop_methods_dict = {
        'Guided Backprop': GuidedBackprop
    }

    # Create output directory
    sat_save_dir = make_dir(os.path.join(output_path, sat_img_name))

    for method, construct in backprop_methods_dict.items():
        # Generate mask
        vis = construct(model, model_name, preds, confidence, gt_coords)
        mask = vis.get_mask(input_image)
        
        # Draw prediction/ground truth boxes
        image = mark_images(mask, preds, gt_coords, confidence)
        
        # Save individual result to .png
        saved_out_path = save_to_png(image, sat_save_dir, sat_img_name, method)
        saliency_results.append(saved_out_path)
    
    print(f"Saliency maps (via backprop) completed ..." )
    return saliency_results




def show_saliency_gradients(model, model_name, 
                            output_path, sat_img_name, 
                            input_image, preds, gt_coords, 
                            saliency_results, confidence):
    """ 
    Produce gradient-based saliency maps (vanilla and integrated gradients)

    Arguments: 
        model: training model or inference model
        model_name: name of model being used (train or infer)
        output_path: path to new 'Visualizations' directory
        sat_img_name: name of specific image file of a satellite 
                    (ex: sat_29155.0001.json, sat_img_name = sat_29155.0001)
        input_image: a single 4-dimensional normalized image (1,512,512,1 shape)
        preds: 2D numpy array of predictions
        gt_coords: 2D numpy array of ground truth box coordinates
        saliency_results: list of paths to the output PNGs
        confidence: confidence threshold [0,1]
    """

    print("Computing saliency maps (via gradients) ...")    

    gradient_methods_dict = {
        'Vanilla Gradient': GradientSaliency,
        'Integrated Gradient': IntegratedGradients
    }

    # Create output directory
    sat_save_dir = make_dir(os.path.join(output_path, sat_img_name))

    for method, construct in gradient_methods_dict.items():
        # Generate gradient-calculated mask
        vis = construct(model, model_name, preds, confidence, gt_coords)
        mask = vis.get_mask(input_image)

        # Reshape to 3D if necessary
        if len(mask.shape) == 4:
            mask = np.squeeze(mask, axis = 0)

        # Draw prediction/ground truth boxes
        image = mark_images(mask, preds, gt_coords, confidence)

        # Save individual result to .png
        saved_out_path = save_to_png(image, sat_save_dir, sat_img_name, method)
        saliency_results.append(saved_out_path)

    print(f"Saliency maps (via gradients) completed..." )
    return saliency_results
