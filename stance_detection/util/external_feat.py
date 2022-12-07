import numpy as np
from stance_detection.util.feature_engineering import word_overlap_features, refuting_features, polarity_features, hand_features, gen_or_load_feats


class External_feat():
    def __init__(self, feature_file, split):
        self.feature_file = feature_file
        self.split = split

    def __call__(self, stances):
        headlines = []
        bodies = []
        for stance in stances:
            headlines.append(stance['Headline'])
            bodies.append(stance['body'])
        headline = stance["headline"]
        body = stance["body"][0]
        overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "overlap."+self.split+".npy")
        refuting = gen_or_load_feats(refuting_features, headlines, bodies, "refuting."+self.split+".npy")
        polarity = gen_or_load_feats(polarity_features, headlines, bodies, "polarity."+self.split+".npy")
        hand = gen_or_load_feats(hand_features, headlines, bodies, "hand."+self.split+".npy")
        feature = np.c_[hand, polarity, refuting, overlap]
        return feature

