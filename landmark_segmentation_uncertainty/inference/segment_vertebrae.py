#!/usr/bin/python
import sys
import csv
sys.path.append('../MedicalDataAugmentationTool-master')

import time
import datetime
import argparse
from collections import OrderedDict
from glob import glob

import SimpleITK as sitk
import numpy as np
import os
import traceback
import tensorflow as tf

import utils.io.image
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
#from network import network_u, UnetClassicAvgLinear3d
from network import network_u, UnetClassicAvgLinear3d, spatial_configuration_net
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders_tuple
from utils.segmentation.segmentation_test import SegmentationTest
from utils.sitk_np import np_to_sitk
import utils.np_image

# from main_vertebrae_localization.py
import json
from copy import deepcopy
import utils.io.common
import utils.io.landmark
from utils.image_tiler import ImageTiler
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.spine_postprocessing import SpinePostprocessing
from collections import OrderedDict

# from main_spine_localization.py
from utils.landmark.common import Landmark

import itk
import shutil



def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented
    


class MainLoop1(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.data_format = data_format
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('spine_heatmap', [1] + network_image_size)
                                                  ])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('spine_heatmap', network_image_size + [1])])

        data_generator_types = {'image': tf.float32,
                                'spine_heatmap': tf.float32}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val, self.target_spine_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)

    def test_full_image(self, image):
        feed_dict = {self.data_val: np.expand_dims(image, axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        return prediction


class InferenceLoop1(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder, base_name):
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.base_name = base_name
        self.data_format = 'channels_last'
        self.image_size = [64, 64, 128]
        self.image_spacing = [8] * 3
        self.output_folder = os.path.join(output_base_folder, 'spine_localization')
        self.save_debug_images = False
        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 3.0,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

        self.network_loop = MainLoop1(network, unet, network_parameters, self.image_size, self.image_spacing, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, (self.base_name + '.nii.gz'))))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_full_image(generators['image'])
            predictions.append(prediction)

        prediction = np.mean(np.stack(predictions, axis=0), axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()

        landmarks = {}

        filenames = os.listdir(self.image_base_folder)
        filenames = [i for i in filenames if i[-len('.nii.gz'):]]
        #file_number = len(filenames)
        #idx = 1

        for image_id in self.image_id_list:

            try:
                #print("[{}/{}] Spine Localization: {} ".format(idx, file_number, image_id))
                print("Spine Localization: {} ".format(image_id))

                dataset_entry = self.dataset_val.get({'image_id': image_id})
                current_id = dataset_entry['id']['image_id']
                datasources = dataset_entry['datasources']
                input_image = datasources['image']
                
                image, prediction, transformation = self.test_full_image(dataset_entry)
                
                predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                    output_spacing=self.image_spacing,
                                                                                    channel_axis=channel_axis,
                                                                                    input_image_sitk=input_image,
                                                                                    transform=transformation,
                                                                                    interpolator='linear',
                                                                                    output_pixel_type=sitk.sitkFloat32)
                
                if self.save_debug_images:
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    heatmap_normalization_mode = (0, 1)
                    utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_input.mha'), output_normalization_mode='min_max', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), output_normalization_mode=heatmap_normalization_mode, data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write(predictions_sitk[0], self.output_file_for_current_iteration(current_id + '_prediction_original.mha'))

                predictions_com = input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(utils.np_image.center_of_mass(utils.sitk_np.sitk_to_np_no_copy(predictions_sitk[0])))))
                landmarks[current_id] = [Landmark(predictions_com)]
                #idx += 1

            except MemoryError as merr:
                print(merr,type(merr))
                raise MemoryError
            except tf.errors.ResourceExhaustedError as gpuerr:
                print(gpuerr,type(gpuerr))
                print('GPU Memory ERROR')
                raise MemoryError
            except Exception as e:
                #idx += 1
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                raise e

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('landmarks.csv'))
        with open(self.output_file_for_current_iteration('landmarks.csv'), "r", encoding="utf-8", newline='') as fin:
            reader = csv.reader(fin)
            indata = list(reader)
        with open(self.output_file_for_current_iteration('landmarks.csv'), "w", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(indata[0])



class MainLoop2(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, cropped_inc, data_format):
        super().__init__()
        self.num_labels = 25
        self.data_format = data_format
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.cropped_inc = cropped_inc

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1])])

        data_generator_types = {'image': tf.float32}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.prediction_val, self.local_prediction_val, self.spatial_prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)

    def test_cropped_image(self, full_image):
        image_size_np = [1] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [1]
        labels_size_np = [self.num_labels] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [self.num_labels]
        predictions_full_size_np = [self.num_labels] + list(full_image.shape[1:]) if self.data_format == 'channels_first' else list(full_image.shape[:-1]) + [self.num_labels]
        cropped_inc = [0] + self.cropped_inc if self.data_format == 'channels_first' else self.cropped_inc + [0]
        image_tiler = ImageTiler(full_image.shape, image_size_np, cropped_inc, True, -1)
        prediction_tiler = ImageTiler(predictions_full_size_np, labels_size_np, cropped_inc, True, 0)

        for image_tiler, prediction_tiler in zip(image_tiler, prediction_tiler):
            current_image = image_tiler.get_current_data(full_image)
            feed_dict = {self.data_val: np.expand_dims(current_image, axis=0)}
            run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)
            prediction = np.squeeze(run_tuple[0], axis=0)
            image_tiler.set_current_data(current_image)
            prediction_tiler.set_current_data(prediction)

        return prediction_tiler.output_image


class InferenceLoop2(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder, base_name):
        super().__init__()
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.base_name = base_name
        self.data_format = 'channels_last'
        self.num_landmarks = 25
        self.image_size = [96, 96, 128]
        self.image_spacing = [2] * 3
        self.cropped_inc = [64, 0, 0]
        self.save_debug_images = False
        self.output_folder = os.path.join(output_base_folder, 'vertebrae_localization')
        self.landmark_file_output_folder = os.path.join(self.output_folder, 'verse_landmarks')
        # utils.io.common.create_directories(self.landmark_file_output_folder)
        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 0.75,
                              'load_spine_landmarks': True,
                              'translate_to_center_landmarks': True,
                              'translate_by_random_factor': True,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()


        self.network_loop = MainLoop2(network, unet, network_parameters, self.image_size, self.image_spacing, self.cropped_inc, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, (self.base_name + '.nii.gz'))))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)
        print(self.image_id_list)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_cropped_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_cropped_image(generators['image'])
            predictions.append(prediction)

        prediction = np.mean(predictions, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def convert_landmarks_to_verse_indexing(self, landmarks, image):
        new_landmarks = []
        spacing = np.array(image.GetSpacing())
        size = np.array(image.GetSize())
        for landmark in landmarks:
            new_landmark = deepcopy(landmark)
            if not landmark.is_valid:
                new_landmarks.append(new_landmark)
                continue
            index = np.array(image.TransformPhysicalPointToContinuousIndex(landmark.coords.tolist()))
            index_with_spacing = index * spacing
            new_coord = np.array([size[2] * spacing[2] - index_with_spacing[2], index_with_spacing[1], index_with_spacing[0]])
            new_landmark.coords = new_coord
            new_landmarks.append(new_landmark)
        return new_landmarks

    def save_landmarks_verse_json(self, landmarks, filename):
        verse_landmarks_list = []
        for i, landmark in enumerate(landmarks):
            if landmark.is_valid:
                verse_landmarks_list.append({'Y': landmark.coords[1],
                                             'X': landmark.coords[0],
                                             'Z': landmark.coords[2],
                                             'label': i + 1})
        with open(filename, 'w') as f:
            json.dump(verse_landmarks_list, f)

    def save_valid_landmarks_list(self, landmarks_dict, filename):
        valid_landmarks = {}
        for image_id, landmarks in landmarks_dict.items():
            current_valid_landmarks = []
            for landmark_id, landmark in enumerate(landmarks):
                if landmark.is_valid:
                    current_valid_landmarks.append(landmark_id)
            valid_landmarks[image_id] = current_valid_landmarks
        utils.io.text.save_dict_csv(valid_landmarks, filename)

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()

        # heatmap_maxima = HeatmapTest(channel_axis, False, return_multiple_maxima=True, min_max_distance=7, min_max_value=0.25, multiple_min_max_value_factor=0.1)
        heatmap_maxima = HeatmapTest(channel_axis, False, return_multiple_maxima=True, min_max_distance=7, min_max_value=0.3, multiple_min_max_value_factor=0.15)
        spine_postprocessing = SpinePostprocessing(num_landmarks=self.num_landmarks, image_spacing=self.image_spacing)

        filenames = os.listdir(self.image_base_folder)
        filenames = [i for i in filenames if i[-len('.nii.gz'):]]
        #file_number = len(filenames)
        #idx = 1

        landmarks = {}
        for image_id in self.image_id_list:
            try:
                #print("[{}/{}] Vertebrae Localization: {} ".format(idx, file_number, image_id))
                print("Vertebrae Localization: {} ".format(image_id))

                dataset_entry = self.dataset_val.get({'image_id': image_id})
                current_id = dataset_entry['id']['image_id']
                datasources = dataset_entry['datasources']
                input_image = datasources['image']

                image, prediction, transformation = self.test_cropped_image(dataset_entry)
                
                if self.save_debug_images:
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    heatmap_normalization_mode = (0, 1)
                    utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_input.mha'), output_normalization_mode='min_max', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), output_normalization_mode=heatmap_normalization_mode, data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                
                predicted_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)
                landmark_sequence = spine_postprocessing.postprocess_landmarks(predicted_landmarks, prediction.shape)

                landmarks[current_id] = landmark_sequence
                verse_landmarks = self.convert_landmarks_to_verse_indexing(landmark_sequence, input_image)
                # self.save_landmarks_verse_json(verse_landmarks, os.path.join(self.landmark_file_output_folder, image_id + '_ctd.json'))

                #idx += 1

            except MemoryError as merr:
                print(merr,type(merr))
                raise MemoryError
            except tf.errors.ResourceExhaustedError as gpuerr:
                print(gpuerr,type(gpuerr))
                print('GPU Memory ERROR')
                raise MemoryError
            except Exception as e:
                #idx += 1
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                raise e

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('landmarks.csv'))
        with open(self.output_file_for_current_iteration('landmarks.csv'), "r", encoding="utf-8", newline='') as fin:
            reader = csv.reader(fin)
            indata = list(reader)
        with open(self.output_file_for_current_iteration('landmarks.csv'), "w", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(indata[0])

        self.save_valid_landmarks_list(landmarks, self.output_file_for_current_iteration('valid_landmarks.csv'))
        with open(self.output_file_for_current_iteration('valid_landmarks.csv'), "r", encoding="utf-8", newline='') as fin:
            reader = csv.reader(fin)
            indata = list(reader)
        with open(self.output_file_for_current_iteration('valid_landmarks.csv'), "w", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(indata[0])


class MainLoop3(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = data_format
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('single_heatmap', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('single_heatmap', network_image_size + [1])])

        data_generator_types = {'image': tf.float32, 'single_heatmap': tf.float32}


        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val, self.single_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        concat_axis = 1 if self.data_format == 'channels_first' else 4
        self.data_heatmap_concat_val = tf.concat([self.data_val, self.single_heatmap_val], axis=concat_axis)
        self.prediction_val = training_net(self.data_heatmap_concat_val, num_labels=self.num_labels, is_training=True, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)
        self.prediction_softmax_val = tf.nn.sigmoid(self.prediction_val)

    def test_full_image(self, image, heatmap):
        feed_dict = {self.data_val: np.expand_dims(image, axis=0),
                     self.single_heatmap_val: np.expand_dims(heatmap, axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_softmax_val,), feed_dict=feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)

        return prediction


class InferenceLoop3(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder, log_path, base_name):
        super().__init__()
        #self.load_model_filenames = ['/models/vertebrae_segmentation/model']
        #self.image_base_folder = '/tmp/data_reoriented'
        #self.setup_base_folder = '/tmp/'
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.base_name = base_name
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = 'channels_last'
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = [128, 128, 96]
        self.image_spacing = [1] * 3
        self.save_debug_images = False
        self.uncert_cal_times = 10
        # self.output_folder = os.path.join(output_base_folder, 'result'+'_{}'.format(time_idx))
        self.output_folder = output_base_folder
        self.log_path = log_path

        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 0.75,
                              'label_gaussian_sigma': 1.0,
                              'heatmap_sigma': 3.0,
                              'generate_single_vertebrae_heatmap': True,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

        self.network_loop = MainLoop3(network, unet, network_parameters, self.image_size, self.image_spacing, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, (self.base_name + '.nii.gz'))))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'vertebrae_localization/valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_full_image(generators['image'], generators['single_heatmap'])
            predictions.append(prediction)

        prediction = np.mean(predictions, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()


        labels = list(range(self.num_labels_all))
        interpolator = 'linear'
        filter_largest_cc = True
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator=interpolator,
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)


        filenames = os.listdir(self.image_base_folder)
        filenames = [i for i in filenames if i[-len('.nii.gz'):]]
        #file_number = len(filenames)
        #idx = 1

        # Create folder to save

        # Process each image
        for image_id in self.image_id_list:
            
            file_save_folder = os.path.join(self.output_folder, image_id)
            if not os.path.exists(file_save_folder):
                os.makedirs(file_save_folder)

            try:
                #print("[{}/{}] Vertebrae Segmentation: {} ".format(idx, file_number, image_id))
                print("Vertebrae Segmentation: {} ".format(image_id))

                first = True
                uncert_prediction_resampled_np = None
                input_image = None

                count = 0
                for landmark_id in self.valid_landmarks[image_id]:

                    print('Landmark ID: {}'.format(landmark_id))

                    # Load patch input
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    datasources = dataset_entry['datasources']

                    # Create numpy array to save results with original size
                    if first:
                        input_image = datasources['image']
                        if self.data_format == 'channels_first':
                            uncert_prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                            uncert_prediction_resampled_np[0, ...] = 0.5
                            label_prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                            label_prediction_resampled_np[0, ...] = 0.5
                        else:
                            uncert_prediction_resampled_np = np.zeros(list(reversed(input_image.GetSize())) + [self.num_labels_all], dtype=np.float16)
                            label_prediction_resampled_np = np.zeros(list(reversed(input_image.GetSize())) + [self.num_labels_all], dtype=np.float16)
                            label_prediction_resampled_np[..., 0] = 0.5
                        first = False


                    # Calculate "self.uncert_cal_times" to utilize Bayesian CNN strategy
                    patch_cal_tmp_np = np.zeros( [self.uncert_cal_times] + list(reversed(self.image_size)) + [1])
                    for time_idx in range(self.uncert_cal_times):
                        image, prediction, transformation = self.test_full_image(dataset_entry)
                        patch_cal_tmp_np[time_idx,...] = prediction

                    patch_cal_label_np = np.mean(patch_cal_tmp_np, axis=0)
                    patch_cal_uncert_np = np.var(patch_cal_tmp_np, axis=0)
                        
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))

                    # Post-processing for eliminate small region
                    if filter_largest_cc:
                        prediction_thresh_np = (patch_cal_label_np > 0.5).astype(np.uint8)
                        if self.data_format == 'channels_first':
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[0])
                            prediction_thresh_np[largest_connected_component[None, ...] == 1] = 0
                        else:
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[..., 0])
                            prediction_thresh_np[largest_connected_component[..., None] == 1] = 0
                        patch_cal_label_np[prediction_thresh_np == 1] = 0

                    if self.save_debug_images:
                        utils.io.image.write_multichannel_np(image, os.path.join(file_save_folder, image_id + '_' + landmark_id + '_input.mha'), output_normalization_mode='min_max', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(patch_cal_label_np, os.path.join(file_save_folder, image_id + '_' + landmark_id + '_prediction.mha'), data_format=self.data_format, image_type=np.double, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(patch_cal_uncert_np, os.path.join(file_save_folder, image_id + '_' + landmark_id + 'uncert.mha'), data_format=self.data_format, image_type=np.double, spacing=self.image_spacing, origin=origin)

                    # Segmentation mask calculation processing
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=patch_cal_label_np,
                                                                                        output_spacing=self.image_spacing,
                                                                                        channel_axis=channel_axis,
                                                                                        input_image_sitk=input_image,
                                                                                        transform=transformation,
                                                                                        interpolator=interpolator,
                                                                                        output_pixel_type=sitk.sitkFloat32)
                    if self.data_format == 'channels_first':
                        label_prediction_resampled_np[int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    else:
                        label_prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    
                    # Uncertainty calculation processing
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=patch_cal_uncert_np,
                                                                                        output_spacing=self.image_spacing,
                                                                                        channel_axis=channel_axis,
                                                                                        input_image_sitk=input_image,
                                                                                        transform=transformation,
                                                                                        interpolator=interpolator,
                                                                                        output_pixel_type=sitk.sitkFloat32)
                    if self.data_format == 'channels_first':
                        uncert_prediction_resampled_np[int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    else:
                        uncert_prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])

                    count += 1
                    with open(log_path, mode="w") as f:
                        f.write(str(int(25+60*count/len(self.valid_landmarks[image_id]))))
                        
                prediction_labels = segmentation_test.get_label_image(label_prediction_resampled_np, reference_sitk=input_image, image_type=np.uint16)
                utils.io.image.write(prediction_labels, os.path.join(file_save_folder, image_id + '_label_mask.nii.gz'))

                #prediction_uncert_np = np.mean(uncert_prediction_resampled_np, axis=-1)     
                #prediction_labels_sitk = np_to_sitk(prediction_uncert_np, type=np.float32)
                #prediction_labels_sitk.CopyInformation(input_image)
                #sitk.WriteImage(prediction_labels_sitk,os.path.join(file_save_folder, image_id + '_pred_uncert.nii.gz'))

                #idx += 1

            except MemoryError as merr:
                print(merr,type(merr))
                raise MemoryError
            except tf.errors.ResourceExhaustedError as gpuerr:
                print(gpuerr,type(gpuerr))
                print('GPU Memory ERROR')
                raise MemoryError
            except Exception as e:
                #idx += 1
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                raise e


def reorient_to_reference(image, reference):
    """
    Reorient image to reference. See itk.OrientImageFilter.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image reoriented to reference image.
    """
    filter = itk.OrientImageFilter[type(image), type(image)].New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    filter.SetDesiredCoordinateDirection(reference.GetDirection())
    filter.Update()
    return filter.GetOutput()


def change_image_type(old_image, new_type):
    """
    Cast image to reference image type.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image cast to reference image type.
    """
    filter = itk.CastImageFilter[type(image), new_type].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()

def cast(image, reference):
    """
    Cast image to reference image type.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image cast to reference image type.
    """
    filter = itk.CastImageFilter[type(image), type(reference)].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()


def copy_information(image, reference):
    """
    Copy image information (spacing, origin, direction) from reference image to input image.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image with image information from reference image.
    """
    filter = itk.ChangeInformationImageFilter[type(image)].New()
    filter.SetInput(image)
    filter.SetReferenceImage(reference)
    filter.UseReferenceImageOn()
    filter.ChangeSpacingOn()
    filter.ChangeOriginOn()
    filter.ChangeDirectionOn()
    filter.Update()
    return filter.GetOutput()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files1', nargs='+', type=str, required=True)
    parser.add_argument('--model_files2', nargs='+', type=str, required=True)
    parser.add_argument('--model_files3', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--base_name', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser_args = parser.parse_args()

    try:
        log_path = os.path.join(parser_args.output_folder, (parser_args.base_name + '_ailog.txt'))
        with open(log_path, mode="w") as f:
            f.write("10")

        os.environ['CUDA_VISIBLE_DEVICES'] = parser_args.gpu

        # 1. reorient_reference_to_rai.py
        if os.path.exists(parser_args.setup_folder):
            shutil.rmtree(parser_args.setup_folder)
        if not os.path.exists(parser_args.setup_folder):
            os.makedirs(parser_args.setup_folder)
        #filenames = glob(os.path.join(parser_args.image_folder, '*.nii.gz'))
        filenames = glob(os.path.join(parser_args.image_folder, (parser_args.base_name + '.nii.gz')))

        file_number = len(filenames)
        idx = 1

        for filename in sorted(filenames):

            basename = os.path.basename(filename)
            basename_wo_ext = basename[:basename.find('.nii.gz')]

            print("[{}/{}] Reference to RAI: {}".format(idx, file_number, basename_wo_ext))
            idx += 1

            is_seg = basename_wo_ext.endswith('_seg')
            image = itk.imread(filename, itk.UC if is_seg else itk.SS)
            reoriented = reorient_to_rai(image)
            
            reoriented.SetOrigin([0, 0, 0])
            m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
            reoriented.SetDirection(m)
            reoriented.Update()
            itk.imwrite(reoriented, os.path.join(parser_args.setup_folder, basename_wo_ext + '.nii.gz'))

        with open(log_path, mode="w") as f:
            f.write("15")

        # 2. main_spine_localization.py
        network_parameters1 = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])
        loop1 = InferenceLoop1(network_u, UnetClassicAvgLinear3d, network_parameters1, parser_args.setup_folder, parser_args.setup_folder, parser_args.model_files1, parser_args.setup_folder, parser_args.base_name)
        loop1.test()
        tf.reset_default_graph()

        with open(log_path, mode="w") as f:
            f.write("20")

        # 3. main_vertebrae_localization.py
        network_parameters2 = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu'), ('spatial_downsample', 4)])
        loop2 = InferenceLoop2(spatial_configuration_net, UnetClassicAvgLinear3d, network_parameters2, parser_args.setup_folder, parser_args.setup_folder, parser_args.model_files2, parser_args.setup_folder, parser_args.base_name)
        loop2.test()
        tf.reset_default_graph()

        with open(log_path, mode="w") as f:
            f.write("25")

        # 4. main_vertebrae_segmentation.py
        network_parameters3 = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])

        # Create dir for saving multiple results
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_folder = 'vertebrae_bayesian_segmentation_rai'
            
        loop3 = InferenceLoop3(network_u, UnetClassicAvgLinear3d, network_parameters3, parser_args.setup_folder, parser_args.setup_folder, parser_args.model_files3, os.path.join(parser_args.setup_folder, result_folder), log_path, parser_args.base_name)
        loop3.test()
        tf.reset_default_graph()

        with open(log_path, mode="w") as f:
            f.write("90")
        
        # 5. reorient_prediction_to_reference.py
        sub_folders = os.listdir(os.path.join(parser_args.setup_folder, result_folder))
        reference_folder = parser_args.setup_folder
        
        for sub_folder in sorted(sub_folders):
            print("RAI to Reference: {}".format(sub_folder))

            sub_folder_path = os.path.join(os.path.join(parser_args.setup_folder, result_folder), sub_folder)
            filenames = os.listdir(sub_folder_path)

            for filename in sorted(filenames):
                basename = os.path.basename(filename)
                img_path = os.path.join(sub_folder_path,filename)
                image = itk.imread(img_path)

                reference = itk.imread(os.path.join(reference_folder, sub_folder + '.nii.gz'), itk.UC)
                reference = change_image_type(reference, itk.Image[itk.F,3])
                reoriented = cast(image, reference)

                reoriented = reorient_to_reference(reoriented, reference)
                reoriented = copy_information(reoriented, reference)
                save_sub_folder = parser_args.output_folder

                itk.imwrite(reoriented, os.path.join(save_sub_folder, filename + '.nii.gz'))

        shutil.rmtree(parser_args.setup_folder)

        with open(log_path, mode="w") as f:
            f.write("100")
        
    except MemoryError as merr:
        print(merr.__class__.__name__)
        sys.exit(-3)
    except Exception as e:
        print(e.__class__.__name__)
        sys.exit(-1)

