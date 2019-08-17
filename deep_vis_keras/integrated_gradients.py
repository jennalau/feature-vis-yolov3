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
from deep_vis_keras.saliency import GradientSaliency

class IntegratedGradients(GradientSaliency):
    
    def __init__(self, model, model_name, preds, confidence, gt_coords):
        super(IntegratedGradients, self).__init__(model, model_name, preds, confidence, gt_coords)

    def get_mask(self, input_image, input_baseline=None, nsamples=20):
        """Returns a integrated gradients mask."""

        # set all pixels to 0 to create all-black image as baseline 
        if input_baseline == None:
            input_baseline = np.zeros_like(input_image) 
        assert input_baseline.shape == input_image.shape

        # calculate difference in pixel values 
        input_diff = input_image - input_baseline 

        # create gradient tensor, initialize all weights set to 0
        total_gradients = np.zeros_like(input_image) 

        # Riemann sum approximation for integral from alpha = [0,1]
        # aggregate gradients in the input that fall in the straight line attribution method
        # between baseline & input pixel value
        for alpha in np.linspace(0, 1, nsamples):
            input_step = input_baseline + (alpha * input_diff) #numerator
            total_gradients += super(IntegratedGradients, self).get_mask(input_step) # calculate partial derivative of gradients with respect to input image
 
        return total_gradients * input_diff
