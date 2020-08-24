from SEFA import SEFA

sefa_object_test = SEFA(dataset_dir='path_to_dataset_images', cloud_dir='https://saliency-map-annotations.s3.amazonaws.com/imagenet_fish/', annotations_num=300,
                        analysis_method='statistical testing', representation_values='binary', representation_option='all', feature_combinations=True)
# The local explanation extraction output is already extracted
# sefa_object_test.extract_local_explanations()
sefa_object_test.extract_semantic_features()
sefa_object_test.extract_representation()
sefa_object_test.analyse_representation()
