from symspellpy import SymSpell, Verbosity
from utils import load_image
import pkg_resources
import pandas as pd
import numpy as np
import itertools
import os
import json


def load_sym_spell():
    """
    Create spell checker dictionary.
    Return:
           sym_spell (SymSpell): spell checker object
    """
    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    return sym_spell


def get_corrected_term(raw_annotation_term, spell_checker, w_mapping):
    """
        Maps the provided term based on the word mapping and spell checks it.
        Args:
            raw_annotation_term (str): term to be checked for corrections
            spell_checker (SymSpell): spell checker object
            w_mapping (dict): dictionary that maps annotated words to other terms that aid aggregation
        Return:
            annotation_count (str): returns the updated term or the original word if it is not in mapping or it is a valid word
    """
    annotation_term_lower = raw_annotation_term.lower()  # normalize to lowercase
    normalized_annotation_term = annotation_term_lower.strip()  # strip whitespace
    spell_checked_annotation_term = spell_checker.lookup(normalized_annotation_term, Verbosity.TOP, max_edit_distance=2, include_unknown=True)[0].term
    if spell_checked_annotation_term in w_mapping:
        corrected_term = w_mapping[spell_checked_annotation_term]  # Map to corrected value from mapping .json
    else:
        corrected_term = spell_checked_annotation_term
    return corrected_term


def get_bounding_box_terms(bounding_box, w_mapping, spell_checker, semantic_info, representation_values):
    """
    TODO: add function docstring
    """
    terms = {'description': []}
    if semantic_info == 'elements':
        element_description = get_corrected_term(bounding_box['object_label'], spell_checker, w_mapping)
        terms['description'].append(element_description)
    elif semantic_info == 'attributes':
        attributes = bounding_box['attributes_label'].replace(', ', ',').split(',')
        for attribute in attributes:
            attribute_description = get_corrected_term(attribute, spell_checker, w_mapping)
            terms['description'].append(attribute_description)
    elif semantic_info == 'element_attribute_pairs':
        element_description = get_corrected_term(bounding_box['object_label'], spell_checker, w_mapping)
        attributes = bounding_box['attributes_label'].replace(', ', ',').split(',')
        for attribute in attributes:
            attribute_description = get_corrected_term(attribute, spell_checker, w_mapping)
            terms['description'].append(attribute_description + '-' + element_description)
    # Save the bounding box information if we want to extract numerical values
    if representation_values == 'numeric':
        terms['left'] = bounding_box['left']
        terms['top'] = bounding_box['top']
        terms['width'] = bounding_box['width']
        terms['height'] = bounding_box['height']
    return terms


def get_unique_feature_pairs(image_annotations):
    """
        Computes the unique semantic feature pairs from the dict of image annotations.
        Args:
            image_annotations (dict): dict with image name as key and a second dict as value which in turn contains the aggregated word counts for each image
        Return:
            unique_feature_pairs (set): unique pairs in the form 'feature1-feature2'
    """
    unique_feature_pairs = set()
    for image in image_annotations:
        if len(image_annotations[image]) >= 2:  # If the image has at least two annotated semantic features
            combinations = list(itertools.combinations(image_annotations[image].keys(), 2))
            for combination in combinations:
                if combination[1] + ' AND ' + combination[0] in unique_feature_pairs:  # Filter same pairs in different order (e.g. face-hair vs hair-face)
                    continue
                unique_feature_pairs.add(combination[0] + ' AND ' + combination[1])
    return unique_feature_pairs


def compute_feature_pairs(image_annotations, representation_values='binary'):
    """
       Extract the presence of feature pairs in each image from the aggregated image annotations.
       Args:
           image_annotations (dict): dict with image name as key and a second dict as value which in turn contains the aggregated word counts for each image
           representation_values (str): specifies if the representation values are binary or numeric
       Return:
           annotation_pairs (dict): dict with feature pair as key as key and 1 as a value (presence of feature)
   """
    unique_pairs = get_unique_feature_pairs(image_annotations)
    annotation_pairs = {}
    for image in image_annotations:
        annotation_pairs[image] = {}
        if len(image_annotations[image]) >= 2:
            combinations = list(itertools.combinations(image_annotations[image].keys(), 2))
            for combination in combinations:
                if combination[0] + ' AND ' + combination[1] in unique_pairs:
                    pair_name = combination[0] + ' AND ' + combination[1]
                elif combination[1] + ' AND ' + combination[0] in unique_pairs:
                    pair_name = combination[1] + ' AND ' + combination[0]
                if representation_values == 'binary':
                    annotation_pairs[image][pair_name] = 1
                elif representation_values == 'numeric':
                    annotation_pairs[image][pair_name] = np.round(np.mean([image_annotations[image][combination[0]], image_annotations[image][combination[1]]]), 2)
        else:
            annotation_pairs[image]['no_pairs'] = 1
    return annotation_pairs


def get_aggregated_annotations(annotations, w_mapping, semantic_info, representation_values):
    """
       Extract the semantic features annotated by the crowd workers for each image using majority voting.
       Args:
           annotations (pd.DataFrame): contains aggregated annotations provided by crowd workers
           w_mapping (dict): contains unique words used by crowd workers as semantic features
           semantic_info (str): crowd worker information that should be accounted for when aggregating. Options: elements, attributes, element_attribute_pairs
           representation_values (str): binary values referring to the presence or absence of the concept or numeric values referring to its gradient intensity
       Return:
           annotation_count (dict): dict with image name as key and a second dict as value which in turn contains the aggregated word counts for each image
   """
    annotation_count = {}
    spell_checker = load_sym_spell()
    heatmap_urls = annotations['Input.original_image_url'].unique()
    for heatmap in heatmap_urls:  # For each image
        if representation_values == 'numeric':
            raw_heatmap_name = '/'.join(heatmap.split('/')[-2:])
            raw_heatmap_name = raw_heatmap_name[:-5] + '_heatmap' + raw_heatmap_name[-5:]
            raw_heatmap_rgb = load_image(os.path.join('./local_interpretability_extraction/heatmaps/', raw_heatmap_name))
        heatmap_name = heatmap.rsplit('/', 1)[-1]  # Remove the directory from the string
        annotation_count[heatmap_name] = {}
        heatmap_annotations = annotations.loc[annotations['Input.original_image_url'] == heatmap, ['Input.original_image_url', 'WorkerId', 'Answer.annotation_data']]
        for annotation in heatmap_annotations['Answer.annotation_data']:  # For each annotation/worker
            json_annotation = json.loads(annotation)
            unique_worker_annotations = []  # Used to count unique terms per worker regardless of bounding boxes
            for bbox in json_annotation:  # For each bounding box
                terms = get_bounding_box_terms(bbox, w_mapping, spell_checker, semantic_info, 'binary')
                for term in terms['description']:
                    if term not in unique_worker_annotations:
                        # Compute word count
                        if term not in annotation_count[heatmap_name]:
                            annotation_count[heatmap_name][term] = 1
                        else:
                            annotation_count[heatmap_name][term] += 1
                        unique_worker_annotations.append(term)
        # Aggregate words using majority voting
        worker_count = annotations['WorkerId'].nunique()
        annotation_count[heatmap_name] = {k: v for k, v in annotation_count[heatmap_name].items() if v >= 0.5 * worker_count}
        if representation_values == 'numeric':
            for annotation in heatmap_annotations['Answer.annotation_data']:  # For each annotation/worker
                json_annotation = json.loads(annotation)
                unique_worker_annotations = []  # Used to count unique terms per worker regardless of bounding boxes
                for bbox in json_annotation:  # For each bounding box
                    terms = get_bounding_box_terms(bbox, w_mapping, spell_checker, semantic_info, 'numeric')
                    for term in terms['description']:
                        if term in annotation_count[heatmap_name]:  # Check if term is in the aggregated terms
                            annotated_heatmap_rgb = raw_heatmap_rgb[int(terms["top"] / 3): int((terms["top"] + terms['height']) / 3),
                                        int(terms["left"] / 3): int((terms["left"] + terms['width']) / 3), :]
                            # Ignore single pixel boxes - empty annotated_heatmap_rgb arrays
                            if annotated_heatmap_rgb.size == 0:
                                continue
                            semantic_feature_intensity = np.round(np.mean(annotated_heatmap_rgb)/255, 2)
                            if term not in unique_worker_annotations:
                                # Keep the intensity values of all the workers for this term
                                if type(annotation_count[heatmap_name][term]) == int:
                                    annotation_count[heatmap_name][term] = [semantic_feature_intensity]
                                else:
                                    annotation_count[heatmap_name][term].append(semantic_feature_intensity)
                                unique_worker_annotations.append(term)
            # Compute mean of intensity for each feature based on all workers
            annotation_count[heatmap_name] = {k: np.round(np.mean(v), 2) for k, v in annotation_count[heatmap_name].items()}
    return annotation_count


def add_not_features(aggregated_annotations):
    # TODO: refactor function
    from representation_extraction import semantic_representation
    image_labels_predictions = pd.read_csv('./representation_extraction/labels_predictions.csv', delimiter=',')
    semantic_feature_names = semantic_representation.get_structured_representation(image_labels_predictions, aggregated_annotations, 'binary').iloc[:, 3:-1].columns
    for image_name, image_features in aggregated_annotations.items():
        for semantic_feature in semantic_feature_names:
            if semantic_feature not in image_features.keys():
                aggregated_annotations[image_name]['NOT ' + semantic_feature] = 1
    return aggregated_annotations


# if __name__ == '__main__':
#     semantic_feature_combinations = True
#     output_dir = 'C:/Users/Panos/Desktop/Thesis_Experiments/Annotations/PA_100K_gender'
#     output_csv = 'mturk_output.csv'
#     annotations_df = pd.read_csv(os.path.join(output_dir, output_csv), delimiter=',')
#     with open('./crowd_computing/mapping.json', 'r') as f:
#         word_mapping = json.load(f)
#     aggregated_annotations = get_aggregated_annotations(annotations_df, word_mapping, 'elements', 'binary')
#     file_name = 'aggregated_annotations.json'
#     if semantic_feature_combinations:
#         aggregated_annotations = compute_feature_pairs(aggregated_annotations)
#         file_name = 'aggregated_annotation_pairs.json'
#     with open('./crowd_computing/' + file_name, 'w') as f:
#         json.dump(aggregated_annotations, f)
