import local_interpretability_extraction
import semantic_feature_annotation
import representation_extraction
import representation_analysis


class SEFA:
    def __init__(self, annotations_num, random_state):
        self.annotations_num = annotations_num
        self.random_state = random_state
        self.semantic_features = None
        self.semantic_representation = None

    def extract_local_explanations(self):
        print('Use SmoothGrad')

    def extract_semantic_features(self):
        self.semantic_features = 'call semantic_feature_annotation'

    def extract_representation(self):
        self.semantic_representation = 'call representation_extraction'

    def analyse_representation(self):
        print('analyse semantic representation')
