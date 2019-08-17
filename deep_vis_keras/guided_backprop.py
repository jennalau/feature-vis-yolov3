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
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import load_model, save_model

class GuidedBackprop(SaliencyMask):

    GuidedReluRegistered = False

    def __init__(self, model, model_name, preds, confidence, gt_coords):
        """Constructs a GuidedBackprop SaliencyMask."""

        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad
        GuidedBackprop.GuidedReluRegistered = True
        
        """ 
            Create a dummy session to set the learning phase to 0 (test mode in keras) without 
            inteferring with the session in the original keras model. This is a workaround
            for the problem that tf.gradients returns error with keras models that contains 
            Dropout or BatchNormalization.

            Basic Idea: save keras model => create new keras model with learning phase set to 0 => save
            the tensorflow graph => create new tensorflow graph with ReLU replaced by GuiededReLU.
        """   
        # Set to test phase
        K.set_learning_phase(0) 
        
        # Load training model
        if 'train' in model_name:
            print('Loading model ...')
            model = load_model('./tmp/gb_keras_train.h5')

        session = K.get_session()
        tf.compat.v1.train.export_meta_graph()
        
        saver = tf.compat.v1.train.Saver()
        saver.save(session, './tmp/guided_backprop_ckpt')

        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph = self.guided_graph)

            with self.guided_graph.gradient_override_map({'LeakyRelu': 'GuidedRelu'}): # replace LeakyRelu with GuidedRelu
                saver = tf.compat.v1.train.import_meta_graph('./tmp/guided_backprop_ckpt.meta')
                saver.restore(self.guided_sess, './tmp/guided_backprop_ckpt')

                output_list = []

                if 'train' in model_name: 
                    batch_idx = 0 # which image in the batch (assume batch size =1)
                    anchor_box_idx = 2 # [20,20]
                    prob_obj_idx = 4 # index for probability of a detection

                    grid_hs, grid_ws = grid_coords(gt_coords)
                    gt_grids = list(zip(grid_hs, grid_ws))

                    train_output = self.guided_graph.get_tensor_by_name(model.output.name) # 64,64,3,6
                    
                    for grid in gt_grids:
                        h = grid[0]
                        w = grid[1]
                        out_tensor = self.guided_graph.get_tensor_by_name(model.output.name)[batch_idx, h, w, anchor_box_idx, prob_obj_idx]
                        output_list.append(out_tensor)
                    
                elif 'infer' in model_name:
                    preds = preds.tolist()
                    for idx, p in enumerate(preds):
                        p = list(p)
                        if p[5] > confidence:
                            out_tensor = self.guided_graph.get_tensor_by_name(model.output.name)[0,idx,5]
                            output_list.append(out_tensor)
                
                self.imported_y = output_list
                self.imported_x = self.guided_graph.get_tensor_by_name(model.input.name)
                self.guided_grads_node = tf.gradients(self.imported_y, self.imported_x) # calculate gradient of class score with respect to input


    def get_mask(self, input_image):
        """Returns a GuidedBackprop mask."""        
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = input_image     
    
        gradients = self.guided_sess.run(self.guided_grads_node, feed_dict = guided_feed_dict)[0][0]
        return gradients