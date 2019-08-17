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

from deep_vis_keras.saliency import SaliencyMask
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2DTranspose, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Ones, Zeros
from visualize.vis_utils.helpers import *

class VisualBackprop(SaliencyMask):

    def __init__(self, model):
        """Constructs a VisualProp SaliencyMask."""
        inps = [model.input, K.learning_phase()]           # input placeholder
        outs = [layer.output for layer in model.layers[1:]]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function        
        self.model = model

    def get_mask(self, input_image):
        """Returns a VisualBackprop mask."""
        x_value = input_image
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])
        leaky_layers = [l for l in self.model.layers if 'leaky' in l.name]

        # visual backprop after each leaky layer (from back to front)
        for layer in leaky_layers[::-1]:
            # leaky layer
            layer_idx = get_layer_idx(self.model, layer.name)
            
            # conv2D layer
            conv_idx = layer_idx - 2
            conv_layer = self.model.layers[conv_idx]
          
            # calculate filters, stride, padding, and kernel size for transpose convolution later
            filters = conv_layer.output.shape[-1] #int 
            stride = conv_layer.strides[0] #int
            padding = conv_layer.padding

            input_size = conv_layer.input.shape[1]
            output_size = conv_layer.output.shape[1] # output = feature maps 
            p = 1 if padding == 'same' else 0 # amount of paddding pixel border 

            kernel = input_size + (2*p) - stride*(output_size - 1)            

            # feature map = output of leaky layer 
            feature_map = layer_outs[layer_idx - 1]

            # average the feature maps of leakyRELU output
            # process output with depth 'n' to depth 1
            layer = np.mean(feature_map, axis=3, keepdims=True) # average after leaky relu!

            # normalize to the range [0,1]
            layer -= np.min(layer)
            layer = layer/(np.max(layer)-np.min(layer)+1e-6)

            if visual_bpr is not None:
                if visual_bpr.shape != layer.shape: 
                    # AveragePooling2D if hit UpSample2D layer
                    if visual_bpr.shape[1] == 2*layer.shape[1]:
                        visual_bpr = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(visual_bpr)

                    # upsample via deconvolution
                    else:
                        visual_bpr = self._deconv(visual_bpr, filters, kernel, stride)
                       
                visual_bpr *= layer
            else:
                visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map, f, k, s):
        """
        The deconvolution operation to upsample the average feature map downstream
        f = # of filters from previous leaky layer (int)
        k = size of kernel from previous leaky layer
        s = amount of stride from previous leaky layer
        """

        x = Input(shape=(None, None, 1)) 

        y = Conv2DTranspose(filters=1, 
                            kernel_size=(3,3), 
                            strides=(2,2), 
                            padding='same', 
                            kernel_initializer=Ones(), # set all weights to 1
                            bias_initializer=Zeros() # set all biases to 0
                            )(x) 

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input, K.learning_phase()]   # input placeholder        
        outs = [deconv_model.layers[-1].output]           # output placeholder

        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]