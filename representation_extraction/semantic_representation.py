import pandas as pd


def get_vocabulary(annotations):
    """
        Compute the unique words used as semantic features by the crowd workers.
        Args:
            annotations (dict): dictionary containing aggregated annotations provided by crowd workers
        Return:
           annotation_vocabulary (dict): keys correspond to words and value to the word index
    """
    annotation_vocabulary = {}
    word_index = 0  # index of each unique word in annotations
    # Get unique words in annotations
    for image in annotations:
        for semantic_feature in annotations[image]:
            if semantic_feature not in annotation_vocabulary:
                annotation_vocabulary[semantic_feature] = word_index
                word_index = word_index + 1
    return annotation_vocabulary


def get_structured_representation(predictions, annotations, representation_values):
    """
        Compute the unique words used as semantic features by the crowd workers.
        Args:
            predictions (pd.DataFrame): contains the true and predicted labels for each image
            annotations (dict): contains aggregated annotations provided by crowd workers
            representation_values (str): binary values referring to the presence or absence of the concept or numeric values referring to its gradient intensity
        Return:
           representation_df (pd.DataFrame): keys correspond to words and value to the word index
    """
    # Create structured representation column names
    structured_representation_cols = ['image_name', 'true_label', 'predicted_label']
    # Get unique words used as semantic features
    annotation_vocabulary = get_vocabulary(annotations)
    # Create a column for each semantic feature in the vocabulary
    for semantic_feature in annotation_vocabulary.keys():
        structured_representation_cols.append(semantic_feature)
    representation_df = pd.DataFrame(columns=structured_representation_cols)
    # Populate structured representation
    for i, image in enumerate(annotations):
        image_predicted_label = predictions.loc[predictions['image'] == image].iloc[0]['predicted_label']
        representation_df.loc[i, 'image_name'] = image
        representation_df.loc[i, 'true_label'] = predictions.loc[predictions['image'] == image].iloc[0]['true_label']
        representation_df.loc[i, 'predicted_label'] = image_predicted_label
        if representation_values == 'numeric':
            for semantic_feature in annotations[image]:
                representation_df.loc[i, semantic_feature] = annotations[image][semantic_feature]
        else:
            for semantic_feature in annotations[image]:
                representation_df.loc[i, semantic_feature] = 1
    representation_df = representation_df.fillna(0)  # fill empty values with 0
    # Add column with that informs whether the samples was correctly or wrongly classified
    representation_df['classification_check'] = (representation_df['true_label'] == representation_df['predicted_label'])
    representation_df['classification_check'] = representation_df['classification_check'].map({True: 'Correctly classified', False: 'Misclassified'})
    return representation_df

#
# if __name__ == "__main__":
#     with open("./crowd_computing/aggregated_annotations.json") as json_file:
#         labels_predictions = "./semantic_feature_representation/predictions_labels.csv"
#         image_labels_predictions = pd.read_csv(labels_predictions, delimiter=",")
#         aggregated_annotations = json.load(json_file)
#         structured_representation_df = get_structured_representation(image_labels_predictions, aggregated_annotations, 'numeric')
#         structured_representation_df.to_csv("./semantic_feature_representation/representation.csv", index=False)
