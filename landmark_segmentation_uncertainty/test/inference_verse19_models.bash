python ../inference/reorient_reference_to_rai.py --image_folder ./img --output_folder ./img_rai
python ../inference/main_spine_localization.py --image_folder ./img_rai --output_folder ./results --setup_folder ./results --model_files ../models/verse19/spine_localization/model
python ../inference/main_vertebrae_localization.py --image_folder ./img_rai --output_folder ./results --setup_folder ./results --model_files ../models/verse19/vertebrae_localization/model
python ../inference/main_vertebrae_segmentation_bn.py --image_folder ./img_rai --output_folder ./results --setup_folder ./results --model_files ../models/verse19/vertebrae_segmentation/model
python ../inference/reorient_prediction_to_reference.py --image_folder ./results/vertebrae_bayesian_segmentation_rai --reference_folder ./img --output_folder ./results/vertebrae_bayesian_segmentation_original