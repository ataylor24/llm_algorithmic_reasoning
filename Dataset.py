from torch.utils.data import Dataset
import utils
import os

class AlgReasoningDataset(Dataset):
    def __init__(self, config, dataset):
        self.data = utils.load_json(os.path.join(os.path.join(config.global_data_path, config.algorithm),dataset))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
