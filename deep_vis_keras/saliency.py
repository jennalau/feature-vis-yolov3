# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Courtesy of: https://github.com/experiencor/deep-viz-keras

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from visualize.vis_utils.helpers import *


class SaliencyMask(object):
    """Base class for saliency masks. Alone, this class doesn't do anything."""
    def __init__(self, model):
        """Constructs a SaliencyMask.

        Args:
            model: the keras model used to make prediction
            output_index: the index of the node in the last layer to take derivative on
        """
        pass

    def get_mask(self, input_image):
        """Returns an unsmoothed mask.

        Args:
            input_image: input image with shape (1,512,512,1)
        """
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=15):
        """Returns a mask that is smoothed with the SmoothGrad method.

        Args:
            input_image: input image with shape (1,512,512,1)
        """
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

    
        # return total_gradients / nsamples

        smoothed = total_gradients / nsamples
        print("get smoothed mask shape: ", smoothed.shape)
        smoothed = np.squeeze(smoothed, axis=0)
        return smoothed

class GradientSaliency(SaliencyMask):
    r"""A SaliencyMask class that computes saliency masks with a gradient."""

    def __init__(self, model, model_name, preds, confidence, gt_coords):
        # Define the function to compute the gradient
        input_tensors = [model.input,        # placeholder for input image tensor
                         K.learning_phase(), # placeholder for mode (train or test) tense
                        ]

        # Initialize list to hold all output tensors, 
        # each of which we will calculate gradients from
        self.output_list = []

        if 'train' in model_name:
            batch_idx = 0 # which image in the batch (assume batch size =1)
            anchor_box_idx = 2 # [20,20]
            prob_obj_idx = 4 # index for Prbability of a detection:
            
            # Calculate coordinates of ground truth box (map 64x64 output grid to 512x512 input image)
            grid_hs, grid_ws = grid_coords(gt_coords)
            gt_grids = list(zip(grid_hs, grid_ws))
            print("gt_grids = ", gt_grids)

            # Calculate gradients of each ground truth box
            for grid in gt_grids:
                h = grid[0]
                w = grid[1]
                gradients = K.gradients(model.output[batch_idx, h, w, anchor_box_idx, prob_obj_idx], model.input)
                self.output_list.append(gradients)
        
        else: # inference model
            preds = preds.tolist()
            for idx, p in enumerate(preds):
                p = list(p)
                if p[5] > confidence:
                    gradients = K.gradients(model.output[0,idx,5], model.input)
                    self.output_list.append(gradients)

        self.compute_gradients = K.function(inputs=input_tensors, outputs=self.output_list) 


    def get_mask(self, input_image):
        """Returns a vanilla gradient mask. # compute gradients of input image with respect to input image pixels

        Args:
            input_image: input image with shape (1,512,512,1).
        """
        # Compute the gradient
        gradients = self.compute_gradients([input_image, 0])[0][0] 
        return gradients