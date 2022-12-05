import os
import re
from tqdm import tqdm
import numpy as np
import math
import h5py
import torch
from torch.utils.data import Dataset
from csv import DictReader
from stance_detection.util.score import report_score, LABELS, score_submission


class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self._load_from_disk()

    def _load_from_disk(self):
        print("Reading dataset")
        bodies = self.split+"_bodies.csv"
        stances = self.split+"_stances.csv"
        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        #match body ids with bodies
        for stance in self.stances:
            stance["body"] = self.articles[(stance["Body ID"])]
            stance["headline"] = stance["Headline"]

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

        # with open(self.data_map[self.split]) as f:
        #     self.scene_names = [line.strip() for line in f]
        # self.objects = []

        # for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
        #     scene_path = os.path.join(self.dataset_root_path, self.split, scene_name + self.file_suffix)
        #     scene = torch.load(scene_path)
        #     for object in scene["objects"]:
        #         object["xyz"] -= object["xyz"].mean(axis=0)
        #         object["rgb"] = object["rgb"].astype(np.float32) / 127.5 - 1
        #         object["scene_id"] = scene_name
        #         # if object["obb"]["up"][2] >= 0:
        #         #     object["class"] = np.array([1])
        #         # else:
        #         #     object["class"] = np.array([0])
        #         self.objects.append(object)

    def __len__(self):
        return len(self.stances)

    def clean(self, s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

    def read(self,filename):
        rows = []
        with open(self.dataset_root_path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    def __getitem__(self, idx):
        stance = self.stances[idx]

        headline = stance["headline"]
        body = stance["body"]  # (N, 3)
        Stance = stance["Stance"]
        stance_class = LABELS.index(Stance)
        if self.cfg.data.use_clean:
            #get clean headline
            headline = self.clean(headline)
            #get clean body
            body = self.clean(body)
        data = {}

        data["stance"] = Stance 
        data["headline"] = headline
        data["body"] = body
        data["stance_id"] = stance_class
        return data
