#!/usr/bin/python
import sys
import csv
sys.path.append('../MedicalDataAugmentationTool-master')

import time
import datetime
import argparse
from collections import OrderedDict
from glob import glob
from glob import escape as glob_escape
import ctypes

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
import gc


# Error handling
EXIT_FAILURE = -1
EXIT_OUT_OF_MEMORY = -3
EXIT_NO_VERTEBRA = -4
EXIT_MODEL_NOT_FOUND = -5
EXIT_MODEL_LOAD = -6
EXIT_DRIVER_OLD = -8

CUDA_DRIVER_MIN_VERSION = 10000  # CUDA 10.0 (required by tensorflow-gpu 1.15)
GPU_CC_MIN = (3, 5)              # oldest GPU generation supported by tensorflow-gpu 1.15
GPU_CC_MAX = (7, 5)              # newest GPU generation verified with this build (Turing)


class SegmentError(Exception):
    """Base class of errors with a specific exit code."""
    exit_code = EXIT_FAILURE


class InputNotFoundError(SegmentError):
    exit_code = EXIT_FAILURE


class NoVertebraError(SegmentError):
    exit_code = EXIT_NO_VERTEBRA


class ModelNotFoundError(SegmentError):
    exit_code = EXIT_MODEL_NOT_FOUND


class ModelLoadError(SegmentError):
    exit_code = EXIT_MODEL_LOAD


class DriverTooOldError(SegmentError):
    exit_code = EXIT_DRIVER_OLD


def is_out_of_memory(e):
    """True if the exception means out of memory (CPU RAM or GPU VRAM)."""
    if isinstance(e, (MemoryError, tf.errors.ResourceExhaustedError)):
        return True
    msg = str(e).lower()
    return ('out of memory' in msg                # CUDA
            or 'oom when allocating' in msg       # TensorFlow
            or 'not enough memory' in msg
            or "can't allocate memory" in msg
            or 'bad allocation' in msg            # std::bad_alloc (ITK, MSVC)
            or 'bad_alloc' in msg
            or 'failed to allocate' in msg)       # SimpleITK / ITK


def get_exit_code(e):
    """Exit code from an exception."""
    if is_out_of_memory(e):
        return EXIT_OUT_OF_MEMORY
    if isinstance(e, SegmentError):
        return e.exit_code
    return EXIT_FAILURE


def check_gpu():
    """
    Check the NVIDIA driver and the selected GPU (call after CUDA_VISIBLE_DEVICES is set; the selected GPU is index 0).
    Uses the CUDA driver API directly, so no GPU memory is allocated.
    :return: True if the GPU can be used. False -> run on CPU.
    :raise DriverTooOldError: The NVIDIA driver does not support CUDA 10.0.
    """
    try:
        cuda = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
    except OSError:
        print('NVIDIA driver not found.')
        return False

    version = ctypes.c_int(0)
    if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
        print('Cannot get NVIDIA driver version.')
        return False
    print('NVIDIA driver: CUDA {}.{}'.format(version.value // 1000, (version.value % 1000) // 10))
    if version.value < CUDA_DRIVER_MIN_VERSION:
        raise DriverTooOldError('NVIDIA driver is too old for this build (CUDA 10.0 required, driver supports CUDA {}.{})'.format(
            version.value // 1000, (version.value % 1000) // 10))

    if cuda.cuInit(0) != 0:
        print('CUDA initialization failed (no GPU?).')
        return False
    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
        print('No CUDA GPU found.')
        return False
    device = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
        print('Cannot get CUDA GPU.')
        return False
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    cuda.cuDeviceGetName(name, len(name), device)
    capability = (major.value, minor.value)
    print('GPU name: {} (sm_{}{})'.format(name.value.decode(errors='replace'), major.value, minor.value))

    if not (GPU_CC_MIN <= capability <= GPU_CC_MAX):
        print('GPU sm_{}{} is not supported by this build (supported: sm_{}{} - sm_{}{}).'.format(
            major.value, minor.value, GPU_CC_MIN[0], GPU_CC_MIN[1], GPU_CC_MAX[0], GPU_CC_MAX[1]))
        return False
    return True


def use_cpu_only():
    """
    Make TensorFlow ignore all GPUs (must be called before the first tf.Session is created).
    CUDA_VISIBLE_DEVICES cannot be used here because it is fixed when CUDA is initialized in check_gpu().
    TensorFlow ignores GPUs with fewer multiprocessors than TF_MIN_GPU_MULTIPROCESSOR_COUNT.
    """
    os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '100000'


def check_model_files(model_files_list):
    """Check that all TensorFlow checkpoint files exist (<prefix>.index and <prefix>.data-*)."""
    for model_files in model_files_list:
        for prefix in model_files:
            if not os.path.isfile(prefix + '.index') or not glob(glob_escape(prefix) + '.data-*'):
                raise ModelNotFoundError('Model file not found: {}'.format(prefix))


def convert_mhd_to_nii(mhd_path, nii_path):
    """
    MHD/RAW (MetaImage) -> NIfTI.
    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(mhd_path)
    image = reader.Execute()

    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(nii_path)
    writer.Execute(image)


def convert_nii_to_mhd(nii_path, mhd_path):
    """
    Label NIfTI -> MHD/RAW (MetaImage, int16).
    """
    img = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(img)
    data = data.astype(np.int16)
    nimg = sitk.GetImageFromArray(data)
    del data
    nimg.CopyInformation(img)
    sitk.WriteImage(nimg, mhd_path)


class MainLoopChecked(MainLoopBase):
    """MainLoopBase with model load errors reported as ModelLoadError."""
    def load_model(self):
        try:
            super().load_model()
        except Exception as e:
            if is_out_of_memory(e):
                raise
            raise ModelLoadError('Model load error: {}: {}'.format(self.load_model_filename, e)) from e



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
    


class MainLoop1(MainLoopChecked):
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

                spine_com = utils.np_image.center_of_mass(utils.sitk_np.sitk_to_np_no_copy(predictions_sitk[0]))
                if not np.all(np.isfinite(spine_com)):
                    raise NoVertebraError('Spine not detected: {}'.format(image_id))
                predictions_com = input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(spine_com)))
                landmarks[current_id] = [Landmark(predictions_com)]
                #idx += 1

            except Exception:
                # Errors are classified and reported in __main__
                print('ERROR predicting (Spine Localization)', image_id)
                raise

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('landmarks.csv'))
        with open(self.output_file_for_current_iteration('landmarks.csv'), "r", encoding="utf-8", newline='') as fin:
            reader = csv.reader(fin)
            indata = list(reader)
        with open(self.output_file_for_current_iteration('landmarks.csv'), "w", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(indata[0])



class MainLoop2(MainLoopChecked):
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
                if landmark_sequence is None or not any(landmark.is_valid for landmark in landmark_sequence):
                    raise NoVertebraError('No vertebra detected: {}'.format(image_id))

                landmarks[current_id] = landmark_sequence
                verse_landmarks = self.convert_landmarks_to_verse_indexing(landmark_sequence, input_image)
                # self.save_landmarks_verse_json(verse_landmarks, os.path.join(self.landmark_file_output_folder, image_id + '_ctd.json'))

                #idx += 1

            except Exception:
                # Errors are classified and reported in __main__
                print('ERROR predicting (Vertebrae Localization)', image_id)
                raise

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


class MainLoop3(MainLoopChecked):
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
        # squeeze=False: with squeeze=True a single valid landmark (e.g. '12') would be loaded as a string
        # instead of a list, and the loop below would iterate over its characters ('1', '2').
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)

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


        interpolator = 'linear'
        filter_largest_cc = True
        # [Memory fix] SegmentationTest.get_label_image() is no longer used.
        # With largest_connected_component=False and all_labels_are_connected=False it only performs
        # an argmax over 26 full-size channels (plus two internal full-size copies), which is replaced
        # below by an equivalent running argmax that needs only 3 bytes/voxel instead of 52 bytes/voxel.


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
                input_image = None
                best_prob_np = None
                best_label_np = None

                # Process landmarks in ascending order so that ties are resolved exactly like np.argmax (lowest index wins)
                landmark_ids = sorted(self.valid_landmarks[image_id], key=int)

                count = 0
                for landmark_id in landmark_ids:

                    print('Landmark ID: {}'.format(landmark_id))

                    # Load patch input
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    datasources = dataset_entry['datasources']

                    # Create numpy arrays to save results with original size
                    # [Memory fix] Instead of a 26-channel float16 array (52 bytes/voxel) plus an unused 26-channel
                    # uncertainty array, keep only the current maximum probability (float16) and its label (uint8).
                    # Background (label 0) is initialized with 0.5, same as the former channel 0.
                    if first:
                        input_image = datasources['image']
                        image_shape_np = list(reversed(input_image.GetSize()))
                        best_prob_np = np.full(image_shape_np, 0.5, dtype=np.float16)
                        best_label_np = np.zeros(image_shape_np, dtype=np.uint8)
                        first = False


                    # Calculate "self.uncert_cal_times" to utilize Bayesian CNN strategy
                    patch_cal_tmp_np = np.zeros( [self.uncert_cal_times] + list(reversed(self.image_size)) + [1])
                    for time_idx in range(self.uncert_cal_times):
                        image, prediction, transformation = self.test_full_image(dataset_entry)
                        patch_cal_tmp_np[time_idx,...] = prediction

                    patch_cal_label_np = np.mean(patch_cal_tmp_np, axis=0)
                    # [Memory fix] Uncertainty is only needed for debug output
                    patch_cal_uncert_np = np.var(patch_cal_tmp_np, axis=0) if self.save_debug_images else None
                    del patch_cal_tmp_np

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

                    # [Memory fix] Running argmax (equivalent to the former argmax over 26 float16 channels).
                    # Cast to float16 first so that comparisons are identical to the former float16 array.
                    # A strict '>' keeps the lower label on ties, same as np.argmax.
                    current_prob_np = utils.sitk_np.sitk_to_np_no_copy(prediction_resampled_sitk[0]).astype(np.float16)
                    del prediction_resampled_sitk
                    update_mask_np = current_prob_np > best_prob_np
                    np.copyto(best_prob_np, current_prob_np, where=update_mask_np)
                    np.copyto(best_label_np, np.uint8(int(landmark_id) + 1), where=update_mask_np)
                    del current_prob_np, update_mask_np

                    # [Memory fix] Uncertainty map resampling removed (its result was never written)

                    count += 1
                    with open(self.log_path, mode="w") as f:
                        f.write(str(int(25+60*count/len(self.valid_landmarks[image_id]))))
                        
                if input_image is None:
                    raise NoVertebraError('No valid landmarks found for {}'.format(image_id))

                del best_prob_np
                prediction_labels = utils.sitk_np.np_to_sitk(best_label_np, type=np.uint16)
                prediction_labels.CopyInformation(input_image)
                del best_label_np
                utils.io.image.write(prediction_labels, os.path.join(file_save_folder, image_id + '_label_mask.nii.gz'))
                del prediction_labels
                gc.collect()

                #prediction_uncert_np = np.mean(uncert_prediction_resampled_np, axis=-1)     
                #prediction_labels_sitk = np_to_sitk(prediction_uncert_np, type=np.float32)
                #prediction_labels_sitk.CopyInformation(input_image)
                #sitk.WriteImage(prediction_labels_sitk,os.path.join(file_save_folder, image_id + '_pred_uncert.nii.gz'))

                #idx += 1

            except Exception:
                # Errors are classified and reported in __main__
                print('ERROR predicting (Vertebrae Segmentation)', image_id)
                raise


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

        # GPU / NVIDIA driver check (driver too old -> exit -8, unsupported or no GPU -> run on CPU)
        if parser_args.gpu.strip() != '-1':
            print('--------------------')
            print(' GPU mode (GPU: {})'.format(parser_args.gpu))
            if not check_gpu():
                print(' -> Run on CPU')
                use_cpu_only()
            print('--------------------')
        else:
            print('--------------------')
            print(' CPU mode ')
            print('--------------------')

        # Model files check (not found -> exit -5)
        check_model_files([parser_args.model_files1, parser_args.model_files2, parser_args.model_files3])

        # Input file check (not found -> exit -1)
        mhd_input_path = os.path.join(parser_args.image_folder, parser_args.base_name + '.mhd')
        if not os.path.isfile(mhd_input_path):
            raise InputNotFoundError('Input file not found: {}'.format(mhd_input_path))

        # 0. mhd -> nii.gz
        print('Convert mhd -> nii.gz: {}'.format(parser_args.base_name))
        input_path = os.path.join(parser_args.image_folder, parser_args.base_name + '.nii.gz')
        convert_mhd_to_nii(mhd_input_path, input_path)

        # 1. reorient_reference_to_rai.py
        if os.path.exists(parser_args.setup_folder):
            shutil.rmtree(parser_args.setup_folder)
        if not os.path.exists(parser_args.setup_folder):
            os.makedirs(parser_args.setup_folder)
        #filenames = glob(os.path.join(parser_args.image_folder, '*.nii.gz'))
        filenames = glob(os.path.join(glob_escape(parser_args.image_folder), (parser_args.base_name + '.nii.gz')))

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
        # [Memory fix] Release session and cached images of loop1
        loop1.network_loop.sess.close()
        del loop1
        gc.collect()
        tf.reset_default_graph()

        with open(log_path, mode="w") as f:
            f.write("20")

        # 3. main_vertebrae_localization.py
        network_parameters2 = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu'), ('spatial_downsample', 4)])
        loop2 = InferenceLoop2(spatial_configuration_net, UnetClassicAvgLinear3d, network_parameters2, parser_args.setup_folder, parser_args.setup_folder, parser_args.model_files2, parser_args.setup_folder, parser_args.base_name)
        loop2.test()
        # [Memory fix] Release session and cached images of loop2
        loop2.network_loop.sess.close()
        del loop2
        gc.collect()
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
        # [Memory fix] Release session and cached images of loop3
        loop3.network_loop.sess.close()
        del loop3
        gc.collect()
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

                # Note: filename is already "<base_name>_label_mask.nii.gz"; the ITK NIfTI writer collapses
                # the doubled extension, so the file is written as "<base_name>_label_mask.nii.gz".
                itk.imwrite(reoriented, os.path.join(save_sub_folder, filename + '.nii.gz'))

        # 6. label nii.gz -> mhd
        label_nii_path = os.path.join(parser_args.output_folder, parser_args.base_name + '_label_mask.nii.gz')
        label_mhd_path = os.path.join(parser_args.output_folder, parser_args.base_name + '_label.mhd')
        if not os.path.isfile(label_nii_path):
            raise SegmentError('Label file was not created: {}'.format(label_nii_path))
        print('Convert nii.gz -> mhd: {}'.format(parser_args.base_name))
        convert_nii_to_mhd(label_nii_path, label_mhd_path)

        with open(log_path, mode="w") as f:
            f.write("100")

    except Exception as e:
        # All errors are classified here (exit code).
        exit_code = get_exit_code(e)
        print(traceback.format_exc())
        print('ERROR (exit code {}): {}'.format(exit_code, e))
        print(e.__class__.__name__)
        sys.exit(exit_code)

    finally:
        # Remove intermediate files also on failure
        shutil.rmtree(parser_args.setup_folder, ignore_errors=True)

