from predict import read_fits, normalize_image, markup_images, get_files, get_satnet_images_dir_list
import os
import errno
import numpy as np
import csv
import json

def make_dir(path, filename=None, ext=None):
        """
        Create directory to hold all visualization output files

        Arguments:
                path: path to create
                filename + ext: filename.ext to create inside path
        Returns:
                path: path to newly created directory 
                                or
                destination: path to newly created file
        """

        # Create file structure
        try:
                os.makedirs(path)
                print("Directory " , path ,  " Created ")
        except OSError as e:
                if e.errno != errno.EEXIST: # path does not exist
                        raise

        # Specify new path new designated file
        if filename and ext:
                destination = os.path.normcase(os.path.join(path, filename + ext))
                return destination

        return path

def get_satellite_dir(root_dir):
        """
        Given root_dir, return list of paths to satellite directories, each of 
        which contain an Annotations & ImageFiles folder
        
        Arguments:
                root_dir: path to directory holding all satellite images

        Returns:
                sat_dir_list: list of paths to satellite directories 
        """
        sat_dir_list = []

        for sat_dir in os.listdir(root_dir):
                new_sat_path = os.path.relpath(os.path.join(root_dir, sat_dir))
                sat_dir_list.append(new_sat_path)

        sat_dir_list = sorted(sat_dir_list)

        return sat_dir_list


def get_layer_idx(model, name):
        """
        Return index of specified layer name within the model 
        """
        
        layer_names = [l.name for l in model.layers]

        try:
                idx = layer_names.index(name)
                return idx
        except ValueError:
                print("That layer name does not exist for this model")
                raise


def make_config(flags, run_path):
        """
        Make config file to hold all arguments
        specified for designated run
        """

        # Create directory for designated run
        f = flags.save_config_filename
        filename, ext = f[:f.index(".")], f[f.index("."):]
        dest = make_dir(run_path, filename, ext)

        # Write arguments to config file
        mydict = vars(flags)

        with open(dest, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in mydict.items():
                        writer.writerow([key, value])


def get_custom_layers(path):
        """
        Reads from model summary JSON file to extract Lambda layers
        """
        custom_layers = dict()

        with open(path) as json_file:
                data = json.load(json_file)

                layers = data['config']['layers']

                for l in layers:
                        if l['class_name'] == 'Lambda':
                                new_element = l['config']['name']
                                custom_layers[new_element] = new_element
        return custom_layers


def get_gt_boxes(json_path):
        """
        Parse JSON file associated with a SatNet image
        to get coordinates of ground truth box

        Arguments:
                json_path:path to JSON file
        Returns:
                coords: 2D numpy array of coordinates, 
                        each element = [ymin, xmin, ymax, xmax]
        """
        coords = []

        with open(json_path) as json_file:
                satnet = json.load(json_file)

                objects = satnet['data']['objects']

                for obj in objects:
                        if obj['class_name'] == 'Satellite':
                                sat_coords = []
                                if obj['source'] == 'satsim':
                                        h = obj['bbox_height']
                                        w = obj['bbox_width']

                                        y_min = (obj['y_center'] - (0.5 * h))
                                        x_min = (obj['x_center'] - (0.5 * w))

                                        y_max = (obj['y_center'] + (0.5 * h))
                                        x_max = (obj['x_center'] + (0.5 * w))

                                else:
                                        y_min = obj['y_min']
                                        x_min = obj['x_min']

                                        y_max = obj['y_max']
                                        x_max = obj['x_max']

                                sat_coords.extend((y_min, x_min, y_max, x_max))       
                                coords.append(sat_coords)

        coords = np.asarray(coords)

        return coords



def grid_coords(bboxes, h=512, w=512, grid_del_h=512/64, grid_del_w=512/64):
        """
        Maps 64x64 output grid from training model into
        512x512 input image grid coordinates
        """
        grid_idxs_h = list()
        grid_idxs_w = list()
    
        for j in range(bboxes.shape[0]):
            box = bboxes[j, :]
    
            ctr_y = h * (box[0]+ box[2])/2.0 # ymin, ymax
            ctr_x = w * (box[1]+ box[3])/2.0 # xmin, xmax
                
            grid_idx_h = int(ctr_y/grid_del_h)
            grid_idx_w = int(ctr_x/grid_del_w)
    
            grid_idxs_h.append(grid_idx_h)
            grid_idxs_w.append(grid_idx_w)

        return grid_idxs_h, grid_idxs_w