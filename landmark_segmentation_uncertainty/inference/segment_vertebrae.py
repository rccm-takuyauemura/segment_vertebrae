#!/usr/bin/python
# ------------------------------------------------------------------------------------------------------------
# Modified by RCCM, 2026-11:
#   - Inference runs with ONNX Runtime on the CPU instead of TensorFlow 1.15 (MainLoopOnnx, MainLoop1/2/3).
#     The models are '<prefix>.onnx' files converted from the TensorFlow checkpoints with
#     freeze_vertebrae_onnx.py and convert_pb_to_onnx.py.
#   - The GPU / NVIDIA driver check is removed (CPU only); --gpu is accepted and ignored.
#   - The random seed of the MC dropout is fixed (ONNX_RANDOM_SEED), so the same CT gives the same result.
#   - Options for evaluation: --mc_runs N (number of MC dropout runs per vertebra, default 10), --seed S (random
#     seed, default ONNX_RANDOM_SEED), --no_dropout (stage 3 without dropout, one run per vertebra, with the
#     model '<prefix>_nodropout.onnx').
#   - Stage 3 speed-up with the same result: the prediction of each vertebra is resampled and compared only inside
#     the region of the input image that its patch covers (patch_region_in_input), not over the whole image;
#     the CT image is loaded and smoothed once for all vertebrae instead of once per vertebra.
# ------------------------------------------------------------------------------------------------------------
import sys
import csv
sys.path.append('../MedicalDataAugmentationTool-master')

import time
import datetime
import argparse
from collections import OrderedDict
from glob import glob
from glob import escape as glob_escape

import SimpleITK as sitk
import numpy as np
import os
import traceback
import onnxruntime as ort

import utils.io.image
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
# The networks are inside the .onnx files. These names are only passed through to InferenceLoop1/2/3.
network_u = UnetClassicAvgLinear3d = spatial_configuration_net = None
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
EXIT_DRIVER_OLD = -8              # no longer used (CPU only); kept so that the exit codes stay the same

# Seed of the random numbers of the MC dropout (stage 3). ONNX Runtime gives the dropout layers the seeds
# ONNX_RANDOM_SEED, +1, +2, ... and each session continues its sequence over the uncert_cal_times runs and the
# vertebrae, so the masks still differ between layers and runs, but the same CT gives the same result every time.
# (Without a fixed seed ONNX Runtime takes it from the clock, and every run differs slightly, as with TF.)
ONNX_RANDOM_SEED = 2026

# --no_dropout: stage 3 uses '<model_files3 prefix>' + NO_DROPOUT_SUFFIX + '.onnx', the same network with dropout
# off (made by freeze_vertebrae_onnx.py / convert_pb_to_onnx.py as 3_vertebrae_segmentation_<i>_check.onnx), and
# runs it once per vertebra. The result is deterministic and stage 3 is about 10 times faster, but it is NOT the
# MC dropout method of the original program: evaluate it before using it.
NO_DROPOUT_SUFFIX = '_nodropout'


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


class SetupFolderError(SegmentError):
    exit_code = EXIT_FAILURE


def check_setup_folder(setup_folder, protected_paths):
    """--setup_folder is deleted (shutil.rmtree) at the start and at the end of the run, so it must be a folder
    used only for intermediate files. Refuse it if it is, or contains, the current folder, the folder of this
    script / exe, a drive root, or any input, output or model path."""
    setup = os.path.normcase(os.path.realpath(setup_folder))
    if os.path.dirname(setup) == setup:
        raise SetupFolderError('setup_folder must not be a drive root: {}'.format(setup_folder))
    for path in protected_paths:
        p = os.path.normcase(os.path.realpath(path))
        if p == setup or p.startswith(setup.rstrip(os.sep) + os.sep):
            raise SetupFolderError('setup_folder must be a separate folder for intermediate files '
                                   '(it is deleted after the run), but it contains: {}'.format(path))


def is_out_of_memory(e):
    """True if the exception means out of memory (CPU RAM or GPU VRAM)."""
    if isinstance(e, MemoryError):
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


def onnx_path(prefix):
    """Model file for a --model_filesN prefix: '<prefix>.onnx' (the prefix may also be given with .onnx)."""
    return prefix if prefix.lower().endswith('.onnx') else prefix + '.onnx'


def check_model_files(model_files_list):
    """Check that all ONNX model files exist (<prefix>.onnx)."""
    for model_files in model_files_list:
        for prefix in model_files:
            if not os.path.isfile(onnx_path(prefix)):
                raise ModelNotFoundError('Model file not found: {}'.format(onnx_path(prefix)))


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


class _NoSession(object):
    """Stands in for tf.Session so that `loopN.network_loop.sess.close()` in __main__ keeps working:
    close() releases the ONNX Runtime sessions of that stage."""
    def __init__(self, sessions):
        self.sessions = sessions

    def close(self):
        self.sessions.clear()


class MainLoopOnnx(object):
    """TensorFlow-free replacement of MainLoopBase: runs one exported ONNX model on the CPU.

    The ONNX models (network structure + weights) are made from the TF checkpoints with
    freeze_vertebrae_onnx.py and convert_pb_to_onnx.py; the network code of network.py is inside them.
    Sessions are created once per model file and kept: besides being faster, this matters for MC dropout
    (stage 3), because a new session restarts the random generator."""
    def __init__(self):
        self._sessions = {}                         # path -> onnxruntime.InferenceSession
        self.sess = _NoSession(self._sessions)
        self.load_model_filename = None
        self.model = None
        self.input_names = []

    # called by InferenceLoop1/2/3; nothing to build any more, the graph is inside the .onnx file
    def init_networks(self):
        pass

    def init_variables(self):
        pass

    def init_saver(self):
        pass

    def load_model(self):
        path = onnx_path(self.load_model_filename)
        if path not in self._sessions:
            print('Restoring model ' + path)
            try:
                self._sessions[path] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            except Exception as e:
                if is_out_of_memory(e):
                    raise
                raise ModelLoadError('Model load error: {}: {}'.format(path, e)) from e
        self.model = self._sessions[path]
        self.input_names = [i.name for i in self.model.get_inputs()]

    def run(self, **arrays):
        """run(image=..., single_heatmap=...) -> first output. Input names are 'placeholder_<key>:0'."""
        feed = {}
        for name in self.input_names:
            key = name.split(':')[0].replace('placeholder_', '')
            feed[name] = np.ascontiguousarray(arrays[key], dtype=np.float32)
        return self.model.run(None, feed)[0]


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
    


class MainLoop1(MainLoopOnnx):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.data_format = data_format
        self.image_size = image_size
        self.image_spacing = image_spacing

    def test_full_image(self, image):
        prediction = self.run(image=np.expand_dims(image, axis=0))
        return np.squeeze(prediction, axis=0)


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



class MainLoop2(MainLoopOnnx):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, cropped_inc, data_format):
        super().__init__()
        self.num_labels = 25
        self.data_format = data_format
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.cropped_inc = cropped_inc

    def test_cropped_image(self, full_image):
        image_size_np = [1] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [1]
        labels_size_np = [self.num_labels] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [self.num_labels]
        predictions_full_size_np = [self.num_labels] + list(full_image.shape[1:]) if self.data_format == 'channels_first' else list(full_image.shape[:-1]) + [self.num_labels]
        cropped_inc = [0] + self.cropped_inc if self.data_format == 'channels_first' else self.cropped_inc + [0]
        image_tiler = ImageTiler(full_image.shape, image_size_np, cropped_inc, True, -1)
        prediction_tiler = ImageTiler(predictions_full_size_np, labels_size_np, cropped_inc, True, 0)

        for image_tiler, prediction_tiler in zip(image_tiler, prediction_tiler):
            current_image = image_tiler.get_current_data(full_image)
            prediction = np.squeeze(self.run(image=np.expand_dims(current_image, axis=0)), axis=0)
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


class MainLoop3(MainLoopOnnx):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = data_format
        self.image_size = image_size
        self.image_spacing = image_spacing

    def test_full_image(self, image, heatmap):
        # the exported model already contains the sigmoid and keeps dropout on (MC dropout)
        prediction = self.run(image=np.expand_dims(image, axis=0), single_heatmap=np.expand_dims(heatmap, axis=0))
        return np.squeeze(prediction, axis=0)


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
        # [Speed-up] dataset_val() uses the uncached image data source, which loads the CT (.nii.gz) and smooths it
        # (Gaussian, input_gaussian_sigma) again for every vertebra. Use the cached data source instead: the same
        # image is loaded and smoothed once and reused (identical data; it is released with loop3).
        uncached_datasources = dataset.datasources
        dataset.datasources = lambda iterator, cached: uncached_datasources(iterator, True)
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
                    # [Speed-up] Resample the prediction only into the part of the input image that this patch
                    # covers. Outside it the full-size resampling gave 0, which never wins against best_prob_np
                    # (>= 0.5), so the result is the same; the work drops from the whole CT to about the patch.
                    region = patch_region_in_input(input_image, transformation, self.image_size, self.image_spacing)
                    current_prob_np, sl = None, None
                    if region is not None:
                        start, size = region
                        prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_input(
                            output_image=patch_cal_label_np,
                            output_spacing=self.image_spacing,
                            channel_axis=channel_axis,
                            input_image_size=size,
                            input_image_spacing=input_image.GetSpacing(),
                            input_image_origin=input_image.TransformIndexToPhysicalPoint(start),
                            input_image_direction=input_image.GetDirection(),
                            transform=transformation,
                            interpolator=interpolator,
                            output_pixel_type=sitk.sitkFloat32)
                        current_prob_np = utils.sitk_np.sitk_to_np_no_copy(prediction_resampled_sitk[0]).astype(np.float16)
                        del prediction_resampled_sitk
                        if region_border_is_zero(current_prob_np, start, size, input_image.GetSize()):
                            sl = (slice(start[2], start[2] + size[2]), slice(start[1], start[1] + size[1]),
                                  slice(start[0], start[0] + size[0]))
                        else:                               # should not happen: fall back to the whole image
                            print('  (landmark {}: patch region check failed, resampling the whole image)'.format(landmark_id))
                            current_prob_np = None
                    if region is not None and current_prob_np is None:
                        prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=patch_cal_label_np,
                                                                                            output_spacing=self.image_spacing,
                                                                                            channel_axis=channel_axis,
                                                                                            input_image_sitk=input_image,
                                                                                            transform=transformation,
                                                                                            interpolator=interpolator,
                                                                                            output_pixel_type=sitk.sitkFloat32)
                        current_prob_np = utils.sitk_np.sitk_to_np_no_copy(prediction_resampled_sitk[0]).astype(np.float16)
                        del prediction_resampled_sitk
                        sl = (slice(None), slice(None), slice(None))

                    # [Memory fix] Running argmax (equivalent to the former argmax over 26 float16 channels).
                    # Cast to float16 first so that comparisons are identical to the former float16 array.
                    # A strict '>' keeps the lower label on ties, same as np.argmax.
                    if current_prob_np is not None:         # None: patch completely outside the image, nothing to do
                        best_prob_view = best_prob_np[sl]       # views: updated in place
                        best_label_view = best_label_np[sl]
                        update_mask_np = current_prob_np > best_prob_view
                        np.copyto(best_prob_view, current_prob_np, where=update_mask_np)
                        np.copyto(best_label_view, np.uint8(int(landmark_id) + 1), where=update_mask_np)
                        del current_prob_np, update_mask_np, best_prob_view, best_label_view

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


def patch_region_in_input(input_image, transformation, patch_size, patch_spacing, margin=2):
    """Region (start, size) [x, y, z] of the input image grid that the patch can reach when the patch is resampled
    back to the input image (linear interpolation). Outside this region the resampled value is the default 0.
    The patch corners (continuous index -0.5 and size - 0.5, i.e. the limits ITK interpolates in) are mapped with
    the patch -> input transformation (translations / affine for the test data) and widened by `margin` voxels.
    Returns None if the patch lies completely outside the input image."""
    corners = []
    for ix in (-0.5, patch_size[0] - 0.5):
        for iy in (-0.5, patch_size[1] - 0.5):
            for iz in (-0.5, patch_size[2] - 0.5):
                q = (ix * patch_spacing[0], iy * patch_spacing[1], iz * patch_spacing[2])
                corners.append(input_image.TransformPhysicalPointToContinuousIndex(transformation.TransformPoint(q)))
    corners = np.array(corners)
    full = input_image.GetSize()
    start = np.maximum(np.floor(corners.min(axis=0)).astype(int) - margin, 0)
    end = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + margin, np.array(full) - 1)
    if np.any(end < start):
        return None
    return [int(v) for v in start], [int(v) for v in end - start + 1]


def region_border_is_zero(region_np, start, size, full_size):
    """True if the resampled region is 0 on every face that is not on the border of the whole image, i.e. nothing of
    the patch was cut off by the region (safety check of patch_region_in_input)."""
    for k, axis in enumerate((2, 1, 0)):                     # numpy axis k <-> [x, y, z] index `axis`
        if start[axis] > 0 and np.any(np.take(region_np, 0, axis=k)):
            return False
        if start[axis] + size[axis] < full_size[axis] and np.any(np.take(region_np, -1, axis=k)):
            return False
    return True


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
    parser.add_argument('--gpu', type=str, default='0')   # ignored (CPU only), kept for MF
    parser.add_argument('--no_dropout', action='store_true',
                        help='stage 3 without MC dropout: one run per vertebra with <prefix>%s.onnx' % NO_DROPOUT_SUFFIX)
    parser.add_argument('--mc_runs', type=int, default=10,
                        help='evaluation: MC dropout runs per vertebra in stage 3 (default 10, as the original program)')
    parser.add_argument('--seed', type=int, default=ONNX_RANDOM_SEED,
                        help='evaluation: random seed of the MC dropout (default %d)' % ONNX_RANDOM_SEED)
    parser_args = parser.parse_args()
    if parser_args.mc_runs < 1:
        parser.error('--mc_runs must be 1 or more')
    if parser_args.no_dropout and parser_args.mc_runs != 10:
        parser.error('--no_dropout always runs once per vertebra; do not combine it with --mc_runs')
    if parser_args.no_dropout:
        parser_args.model_files3 = [p[:-len('.onnx')] if p.lower().endswith('.onnx') else p
                                    for p in parser_args.model_files3]
        parser_args.model_files3 = [p + NO_DROPOUT_SUFFIX for p in parser_args.model_files3]
    setup_folder_checked = False   # the setup folder is deleted only after it passed check_setup_folder()

    try:
        # setup_folder is deleted below: never accept a folder that holds anything else
        script_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
        check_setup_folder(parser_args.setup_folder,
                           [os.getcwd(), script_dir, parser_args.image_folder, parser_args.output_folder]
                           + [onnx_path(p) for p in parser_args.model_files1 + parser_args.model_files2 + parser_args.model_files3])
        setup_folder_checked = True

        log_path = os.path.join(parser_args.output_folder, (parser_args.base_name + '_ailog.txt'))
        with open(log_path, mode="w") as f:
            f.write("10")

        # ONNX Runtime, CPU only (--gpu is ignored)
        print('--------------------')
        print(' CPU mode (ONNX Runtime {})'.format(ort.__version__))
        ort.set_seed(parser_args.seed)      # before any session is created: reproducible MC dropout
        if parser_args.no_dropout:
            print(' Stage 3: no dropout, 1 run per vertebra ({})'.format(onnx_path(parser_args.model_files3[0])))
        elif parser_args.mc_runs != 10 or parser_args.seed != ONNX_RANDOM_SEED:
            print(' Stage 3: MC dropout, {} runs per vertebra, seed {}'.format(parser_args.mc_runs, parser_args.seed))
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

        with open(log_path, mode="w") as f:
            f.write("25")

        # 4. main_vertebrae_segmentation.py
        network_parameters3 = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])

        # Create dir for saving multiple results
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_folder = 'vertebrae_bayesian_segmentation_rai'
            
        loop3 = InferenceLoop3(network_u, UnetClassicAvgLinear3d, network_parameters3, parser_args.setup_folder, parser_args.setup_folder, parser_args.model_files3, os.path.join(parser_args.setup_folder, result_folder), log_path, parser_args.base_name)
        if parser_args.no_dropout:
            loop3.uncert_cal_times = 1      # the network has no dropout: every run would give the same output
        else:
            loop3.uncert_cal_times = parser_args.mc_runs
        loop3.test()
        # [Memory fix] Release session and cached images of loop3
        loop3.network_loop.sess.close()
        del loop3
        gc.collect()

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
        if setup_folder_checked:
            shutil.rmtree(parser_args.setup_folder, ignore_errors=True)

