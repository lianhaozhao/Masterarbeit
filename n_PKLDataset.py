import os
import pickle
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class_to_onehot = {
    'R05': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R10': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R15': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'R20': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R25': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'R30': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'R35': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'R40': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'R45': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'R50': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}
class_to_index = {cls: i for i, cls in enumerate(class_to_onehot.keys())}

def z_score(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def min_max(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)
def identity(x):
    return x
TRANSFORM_MAP = {
    "zscore": z_score,
    "minmax": min_max,
    "none": identity,
}
import torch

class FitNormalizer:
    def __init__(self, mode="zscore"):
        assert mode in ["zscore", "minmax"]
        self.mode = mode
        self.fitted = False

    def fit(self, tensors):  # tensors: list of (C, L) torch.Tensors
        x = torch.cat([t.reshape(t.shape[0], -1) for t in tensors], dim=1)  # (C, sumL)
        if self.mode == "zscore":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std  = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        else:
            self.minv = x.min(dim=1, keepdim=True).values
            self.maxv = x.max(dim=1, keepdim=True).values
        self.fitted = True

    def transform(self, x):  # x: (C, L)
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        else:
            return (x - self.minv) / (self.maxv - self.minv + 1e-6)

class N_PKLDataset(Dataset):
    def __init__(self, txt_path, transform_type='none', normalizer=None):
        """
        txt_path: 每行一个 .pkl 路径
        transform_type: 'none' | 'zscore' | 'minmax'（当 normalizer=None 时才生效）
        normalizer: FitNormalizer（source 上拟合后传进来，源/目标共用）
        """
        self.txt_path = txt_path
        self.transform_type = transform_type.lower()
        self.files = self._load_file_paths()
        self.transform_fn = TRANSFORM_MAP.get(self.transform_type, identity)
        self.normalizer = normalizer

    def _load_file_paths(self):
        with open(self.txt_path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        files.sort()
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path)

        class_label = filename.split('_')[0]
        label_index = class_to_index[class_label]
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        with open(file_path, 'rb') as f:
            array_data = pickle.load(f)
        data_tensor = torch.tensor(array_data, dtype=torch.float32)

        # 统一成 (C, L)
        if data_tensor.dim() == 2:
            # 你当前逻辑是取第0通道；如需保留信息，可以改成 mean：
            # data_tensor = data_tensor.mean(dim=0, keepdim=True)
            data_tensor = data_tensor[0].unsqueeze(0)
        elif data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(0)

        # ——关键：source‑fitted —— 先看 normalizer 是否已拟合；否则退回 per‑sample 变换
        if self.normalizer is not None and getattr(self.normalizer, "fitted", False):
            data_tensor = self.normalizer.transform(data_tensor)
        else:
            data_tensor = self.transform_fn(data_tensor)

        return data_tensor, label_tensor
def fit_normalizer_from_txt(txt_path, mode="zscore", use_mean_channel=False, max_files=None):
    norm = FitNormalizer(mode=mode)
    tensors = []

    with open(txt_path, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]
    files.sort()
    if max_files is not None:
        files = files[:max_files]

    for p in files:
        with open(p, "rb") as f:
            arr = pickle.load(f)
        t = torch.tensor(arr, dtype=torch.float32)
        if t.dim() == 2:
            t = t.mean(dim=0, keepdim=True) if use_mean_channel else t[0].unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(0)
        tensors.append(t)

    norm.fit(tensors)
    return norm

if __name__ == '__main__':
    dataset = PKLDataset(txt_path='datasets/DC_T197_AZ.txt', transform_type='zscore')
    print(len(dataset))  # 样本总数
    x, y = dataset[1]  # 获取第一个样本
    print(x.shape, x, y)  # 打印数据和标签
    for i in x:
        plt.plot(i)
    plt.show()

