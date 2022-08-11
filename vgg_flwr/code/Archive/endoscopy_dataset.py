import pickle
import torch.utils.data


class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        X = self.data[idx][0]
        y = self.data[idx][1]
        X = torch.tensor(X)
        y = torch.tensor(y)
        return X, y

    def __len__(self):
        return len(self.data)
