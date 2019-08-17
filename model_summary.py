import os
from contextlib import redirect_stdout
from models.keras_yolo3 import Yolo
from visualize.vis_utils.helpers import *
from visualize.vis_utils.image_processing import *


def summarize(model, path):
        """ 
        Provides a summary of trained YOLOv3 model with weights 
        from best checkpoint in designated .txt file 

        Arguments:
                model: YOLOv3 object
                path: path to output file's location
        """
        infer_model = model._infer_model

        print("Computing model summary ...")  

        filename = 'model_summary'
        ext = '.txt'
        dest = make_dir(path, filename, ext)

        with open(dest, 'w') as file:
                with redirect_stdout(file):
                        infer_model.summary()

        print(f"Model summary completed (saved to {dest})..." )