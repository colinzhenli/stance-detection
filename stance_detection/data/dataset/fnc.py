from stance_detection.data.dataset import GeneralDataset


class FNC(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
