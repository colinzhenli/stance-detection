import numpy as np
from stance_detection.util.feature_engineering import word_overlap_features, refuting_features, polarity_features, hand_features, gen_or_load_feats


class External_feat():
    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.split = "train"

    def __call__(self, stance):
        headline = stance["headline"][0]
        body = stance["body"][0]
        overlap = gen_or_load_feats(word_overlap_features, headline, body, "overlap."+self.split+".npy")
        refuting = gen_or_load_feats(refuting_features, headline, body, "refuting."+self.split+".npy")
        polarity = gen_or_load_feats(polarity_features, headline, body, "polarity."+self.split+".npy")
        hand = gen_or_load_feats(hand_features, headline, body, "hand."+self.split+".npy")
        feature = np.r_[hand, polarity, refuting, overlap]
        return feature

