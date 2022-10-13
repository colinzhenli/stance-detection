from minsu3d.data.dataset import GeneralDataset


class Multiscan(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
