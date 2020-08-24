from local_interpretability_extraction import saliency_maps
from semantic_feature_annotation import semantic_features
from representation_extraction import semantic_representation
from representation_analysis import analysis_tools
import pandas as pd
import json


class SEFA:
    def __init__(self, dataset_dir, cloud_dir, annotations_num, analysis_method, representation_values, representation_option, feature_combinations=True, not_operator=False,
                heatmap_dir='./local_interpretability_extraction/heatmaps/', random_state=None):
        self.dataset_dir = dataset_dir
        self.cloud_dir = cloud_dir
        self.heatmap_dir = heatmap_dir
        self.annotations_num = annotations_num
        if analysis_method not in ['statistical testing', 'rule mining', 'decision tree']:
            raise ValueError('The analysis method has to be statistical testing OR rule mining OR decision tree.')
        else:
            self.analysis_method = analysis_method
        self.representation_values = representation_values
        self.representation_option = representation_option
        self.feature_combinations = feature_combinations
        if not_operator and representation_values == 'numeric':
            raise ValueError('The NOT operator is unavailable with numeric representation values.')
        self.not_operator = not_operator
        self.random_state = random_state
        self.annotation_output = None
        self.semantic_features = None
        self.semantic_representation = None

    def extract_local_explanations(self):
        saliency_maps.extract_heatmaps(images_dir=self.dataset_dir, heatmap_dir=self.heatmap_dir, num_of_images=self.annotations_num)
        saliency_maps.extract_image_urls(cloud_bucket=self.cloud_dir, heatmap_dir=self.heatmap_dir)
        print('----- Saliency maps extracted! -----')

    def extract_semantic_features(self):
        self.annotation_output = pd.read_csv('./semantic_feature_annotation/annotations_output.csv', delimiter=',')
        with open('./semantic_feature_annotation/word_map.json', 'r') as f:
            word_map_function = json.load(f)
        if self.representation_option == 'all':
            element_features = semantic_features.get_aggregated_annotations(self.annotation_output, word_map_function, 'elements', self.representation_values)
            attribute_features = semantic_features.get_aggregated_annotations(self.annotation_output, word_map_function, 'attributes', self.representation_values)
            pair_features = semantic_features.get_aggregated_annotations(self.annotation_output, word_map_function, 'element_attribute_pairs', self.representation_values)
            if self.not_operator:
                element_features = semantic_features.add_not_features(element_features)
                attribute_features = semantic_features.add_not_features(attribute_features)
            if self.feature_combinations:
                elements_combinations = semantic_features.compute_feature_pairs(element_features, representation_values=self.representation_values)
                attributes_combinations = semantic_features.compute_feature_pairs(attribute_features, representation_values=self.representation_values)
                pairs_combinations = semantic_features.compute_feature_pairs(pair_features, representation_values=self.representation_values)
                # Merge all aggregated dictionaries according to image name(key)
                self.semantic_features = dict([(k, {**element_features[k], **attribute_features[k], **pair_features[k], **elements_combinations[k],
                                                    **attributes_combinations[k], **pairs_combinations[k]}) for k in element_features])
            else:
                self.semantic_features = dict([(k, {**element_features[k], **attribute_features[k], **pair_features[k]}) for k in element_features])
        else:
            self.semantic_features = semantic_features.get_aggregated_annotations(self.annotation_output, word_map_function, self.representation_option, self.representation_values)
            if self.not_operator:
                self.semantic_features = semantic_features.add_not_features(self.semantic_features)
            if self.feature_combinations:
                feature_combinations = semantic_features.compute_feature_pairs(self.semantic_features, representation_values=self.representation_values)
                self.semantic_features = dict([(k, {**self.semantic_features[k], **feature_combinations[k]}) for k in self.semantic_features])
        print('----- Semantic features extracted! -----')

    def extract_representation(self):
        image_labels_predictions = pd.read_csv('./representation_extraction/labels_predictions.csv', delimiter=',')
        self.semantic_representation = semantic_representation.get_structured_representation(image_labels_predictions, self.semantic_features, self.representation_values)
        print('----- Semantic representation extracted! -----')

    def analyse_representation(self):
        if self.analysis_method == 'statistical testing':
            analysis_tools.compute_statistical_tests(self.semantic_representation, print_test_values=True, representation_values=self.representation_values)
        elif self.analysis_method == 'rule mining':
            # TODO: to implement function calls
            print(self.analysis_method)
        elif self.analysis_method == 'decision tree':
            # TODO: to implement function calls
            print(self.analysis_method)
