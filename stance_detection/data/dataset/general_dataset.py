import os
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import h5py
import torch
from torch.utils.data import Dataset
from csv import DictReader
from stance_detection.util.score import report_score, LABELS, score_submission
from stance_detection.util.external_feat import External_feat
from stance_detection.util.feature_engineering import normalize_word
from stance_detection.util.statistical_feat import get_val_tfidf, get_train_tfidf




class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.encoding_file = os.path.join(cfg.data.encoding_path, split)
        self.external_feat = External_feat(cfg.data.feature_file, split)
        self.stat_feat_path = os.path.join(cfg.data.stat_feat_path, split)
        self._load_from_disk()

    def _load_from_disk(self):
        print("Reading dataset")
        bodies_path = self.split+"_bodies.csv"
        stances_path = self.split+"_stances.csv"
        stances = self.read(stances_path)
        articles = self.read(bodies_path)
        self.stances = stances
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']
            self.articles["articleBody"] = self.clean(article["articleBody"])

        #match body ids with bodies
        for stance in self.stances:
            stance["body"] = self.articles[(stance["Body ID"])]
            stance["headline"] = stance["Headline"]
            if self.cfg.data.use_clean:
                #get clean headline
                stance["headline"] = self.clean(stance["Headline"])
                #get clean body
                stance["body"] = self.clean(stance["body"])

        #get external features
        features = self.external_feat(self.stances)
        for i, stance in enumerate(self.stances):
            stance["external_feat"] = features[i]

        #get statistical features:
        stat_feat_path = self.stat_feat_path+ ".npy"
        self.val_articles = dict()
        if os.path.isfile(stat_feat_path):
            stat_feat_list = np.load(stat_feat_path)
        else:
            if self.split=="train":
                val_stances = self.read("val_stances.csv")
                val_articles = self.read("val_bodies.csv")
                for stance in val_stances:
                    stance['Body ID'] = int(stance['Body ID'])
                    stance["headline"] = self.clean(stance["Headline"])
                for article in val_articles:
                    self.val_articles[int(article['Body ID'])] = self.clean(article['articleBody'])
                stat_feat_list, train_doc, total_doc = get_train_tfidf(self.stances, self.articles, val_stances, self.val_articles, self.cfg.data.lim_unigram)
                if not os.path.isfile(self.cfg.data.dataset_root_path + "train_doc.txt"):
                    self.write(train_doc, "train_doc.txt")
                    self.write(total_doc, "total_doc.txt")
            
            if self.split=="val":
                train_doc = self.read_doc("train_doc.txt")
                total_doc = self.read_doc("total_doc.txt")
                stat_feat_list = get_val_tfidf(self.stances, self.articles, train_doc, total_doc, self.cfg.data.lim_unigram)
            
            np.save(stat_feat_path, stat_feat_list)

        for i, stance in enumerate(self.stances):
            stance["stat_feat"] = stat_feat_list[i]

        #get word one hot encoding
        encoding_path = self.encoding_file + ".npy"
        if os.path.isfile(encoding_path):
            encoding_list = np.load(encoding_path)
            for i, stance in enumerate(self.stances):
                stance["encoding"] = encoding_list[i]
        else:
            self.word_to_ix = {}
            #add the special tokens
            self.word_to_ix["UNK"] = 0
            self.word_to_ix["CLS"] = 1
            self.word_to_ix["SEP"] = 2
            self.word_to_ix["PAD"] = 3
            #get and save word to index for training set
            if self.split=="train":
                if not os.path.isfile(self.cfg.data.word_idx_path):
                    self.get_word_idx(self.stances)
                else:
                    self.word_to_ix = torch.load(self.cfg.data.word_idx_path)

            #load word to index for validation set
            if self.split=="val":
                self.word_to_ix = torch.load(self.cfg.data.word_idx_path)

            #one hot encoding for head-body pair
            self.get_encoding(self.encoding_file)
            print("Vocabulary size: " + str(len(self.word_to_ix)))
        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))


    def __len__(self):
        return len(self.stances)

    def clean(self, s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

    def get_tokenized_with_removal(self, l):
        # Removes stopwords from a list of tokens and stem words
        stopWords = set(stopwords.words('english'))
        return [normalize_word(t) for t in nltk.word_tokenize(l) if t not in stopWords]

    # def get_exteranl_feat(self, stances):
    #     features = self.external_feat(stances)
    #     for i, stance in enumerate(self.stances):
    #         stance["external_feat"] = features[i]

    def get_word_idx(self, stances):
        word_idx_path = self.cfg.data.word_idx_path
        print("get " + self.split + " set vocabulary")
        for stance in tqdm(stances):
            for word in self.get_tokenized_with_removal(stance["headline"]):
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)  
            for word in self.get_tokenized_with_removal(stance["body"]): 
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix) 
        torch.save(self.word_to_ix, word_idx_path)

    def get_encoding(self, encoding_file):
        encoding_list = []
        max_sentence_len = 0
        print("get " + self.split + " set sentence encoding")
        for stance in tqdm(self.stances):
            encoding = []

            #get encoding
            encoding.append(self.word_to_ix["CLS"])
            for token in self.get_tokenized_with_removal(stance["headline"]):
                if token not in self.word_to_ix:
                    encoding.append(self.word_to_ix["UNK"])
                else:
                    encoding.append(self.word_to_ix[token])
            encoding.append(self.word_to_ix["SEP"])
            for token in self.get_tokenized_with_removal(stance["body"]):
                if token not in self.word_to_ix:
                    encoding.append(self.word_to_ix["UNK"])
                else:
                    encoding.append(self.word_to_ix[token])
            sentence_len = len(encoding)
            if sentence_len > max_sentence_len:
                max_sentence_len = sentence_len
            #add padding 
            for i in range(self.cfg.data.max_length-sentence_len):
                encoding.append(self.word_to_ix["PAD"])
            stance["encoding"] = np.array(encoding)
            encoding_list.append(stance["encoding"])
        np.save(encoding_file, encoding_list)
        print("max sentence length for " + self.split + ": " + str(max_sentence_len))
        
        

    def read(self,filename):
        rows = []
        with open(self.dataset_root_path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
    def read_doc(self, filename):
        with open(self.dataset_root_path + "/" + filename) as f:
            lines = f.readlines()
        return lines

    def write(self, data, filename):
        outFile =  open(self.dataset_root_path + "/" + filename, "w", encoding='utf-8')
        for line in data:
            outFile.write(line)
            outFile.write("\n")
        outFile.close()      

    def __getitem__(self, idx):
        stance = self.stances[idx]

        headline = stance["headline"]
        body = stance["body"]  # (N, 3)
        Stance = stance["Stance"]
        external_feat = stance["external_feat"]
        stat_feat = stance["stat_feat"]
        encoding = stance["encoding"]
        stance_class = LABELS.index(Stance)
        data = {}

        data["stance"] = Stance 
        # data["headline"] = headline
        # data["body"] = body
        data["stance_id"] = stance_class
        data["external_feat"] = external_feat
        data["encoding"] = encoding
        data["stat_feat"] = stat_feat
        return data
