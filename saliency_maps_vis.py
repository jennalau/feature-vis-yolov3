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
from visualize.vis_utils.saliency_utils import show_saliency_backprop, show_saliency_gradients

import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import save_model, load_model, Model
import tensorflow.compat.v1.keras.backend as K


def main_saliency(flags, yolo_obj):
    """
    Main function for generating saliency maps via different backprop methods
    
    Arguments:
        flags: arguments set in run_visuals.py/run_visuals_warmup.py or command line
        yolo_obj: YOLO model instantiation
    """

    # Get list of paths to satellite folders
    sat_dir_list = get_satellite_dir(flags.input_path)

    if flags.use_train_model:
        # Load training model
        loaded_train_model = load_model('./tmp/gb_keras_train.h5')
        t_inp = loaded_train_model.input
        t_out = loaded_train_model.output

        model = Model(t_inp, t_out)
        model_str = "_train"
    else:
        model = yolo_obj._infer_model
        model_str = "_infer"

    graph = tf.get_default_graph()
    with graph.as_default():
        for sat_dir in sat_dir_list:
            # Get subdirectories
            sat_annot_dir = os.path.join(sat_dir, 'Annotations')
            sat_images_dir = os.path.join(sat_dir, 'ImageFiles')

            # Make directory for visualization outputs
            save_dir = os.path.join(sat_dir, 'Visualizations')
            make_dir(save_dir)

            # Get list of filepaths to .fits images (remove hidden files)
            fits_paths_list = sorted(get_files(sat_images_dir, ".fits"))
            fits_paths_list = [f for f in fits_paths_list if '._' not in f]

            # List of .fits files with corresponding .json files
            valid_fits_paths = [] 

            for f in fits_paths_list:
                annot_path = f.replace('ImageFiles', 'Annotations')
                annot_path = annot_path.replace('.fits', '.json')

                if os.path.exists(annot_path):
                    valid_fits_paths.append(f)
            
            # Convert .fits image data into normalized image
            model_input_images = load_fits_input_images(valid_fits_paths)
        
            # Organize path to .fits files and normalized 1,512,512,1 image into a dictionary
            model_inputs_dict = dict(zip(valid_fits_paths, model_input_images))

            # Make predictions
            preds = yolo_obj.predict(model_input_images, steps=len(model_input_images))

            # Organize path to .fits files and corresponding prediction output
            path_pred_dict = dict(zip(valid_fits_paths, preds))

            for path, pred in path_pred_dict.items():

                # Get ground truth box coordinates
                json_path = path.replace('ImageFiles', 'Annotations').replace('.fits', '.json') 
                gt_coords = get_gt_boxes(json_path)

                sat_img_name = path[-19:-5]

                # Place original input image into list of paths to output results
                saliency_results = []

                # Draw prediction/ground truth boxes on input image
                input_image = model_inputs_dict[path]
                og_image = np.squeeze(input_image, axis=0)
                og_image = mark_images(og_image, pred, gt_coords, flags.confidence_threshold)

                # Save marked, original input image to png file
                sat_save_dir = make_dir(os.path.join(save_dir, sat_img_name))
                saved_out_path = save_to_png(og_image, sat_save_dir, sat_img_name, "SatNet")
                saliency_results.append(saved_out_path)

                if flags.vis_with_gradients:
                    saliency_results = show_saliency_gradients(model, model_str, save_dir, 
                                                                sat_img_name, input_image,
                                                                pred, gt_coords, saliency_results, 
                                                                flags.confidence_threshold)
                if flags.vis_with_backprop: 
                    saliency_results = show_saliency_backprop(model, model_str, save_dir, 
                                                                sat_img_name, input_image,
                                                                pred, gt_coords, saliency_results, 
                                                                flags.confidence_threshold) 

                # Combine all results
                concat_images = []
                for png in saliency_results:
                    img = cv2.imread(png)
                    concat_images.append(img)

                final_output = np.concatenate(concat_images, axis = 1)
                
                # Write out all results to a single file
                write_path = os.path.join(save_dir, sat_img_name, 'final_out.png')
                cv2.imwrite(write_path, final_output)
