from torch.utils.data import Dataset
import torch


class SNLITorchDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_map = {index: key for index, key in enumerate(self.data)}
        self.data_map_inv = {v: k for k, v in self.data_map.items()}

    def __getitem__(self, index):
        item = self.data[self.data_map[index]]
        input_ids, attention_mask, token_type_ids, label = item['input_ids'], \
            item['attention_mask'], item['token_type_ids'], item['label']
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(label)

    def __len__(self):
        return len(self.data)
