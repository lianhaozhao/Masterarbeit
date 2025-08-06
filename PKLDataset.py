import os
import pickle
import torch
from torch.utils.data import Dataset


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

class PKLDataset(Dataset):
    def __init__(self, txt_path, transform_type=None):
        """
        Args:
            txt_path (str): Path to a .txt file containing a list of .pkl file paths.
            transform_type (str, optional): Transformation to apply. If None, the raw time series is returned.
        """
        self.txt_path = txt_path
        self.transform_type = transform_type

        # Read all file paths from txt file
        with open(txt_path, "r", encoding="utf-8") as f:
            self.files = [line.strip() for line in f if line.strip()]

        self.files.sort()  # sort for consistent ordering

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get the path of the pkl file
        file_path = self.files[idx]
        filename = os.path.basename(file_path)

        # Extract class label (e.g. 'R05' from 'R05_I123.pkl')
        class_label = filename.split('_')[0]
        label_index = class_to_index[class_label]
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        # Load the data
        with open(file_path, 'rb') as f:
            array_data = pickle.load(f)
        data_tensor = torch.tensor(array_data, dtype=torch.float32)

        if data_tensor.dim() == 2:
            # 假设 array_data.shape == (64, 2800)，选择一个或聚合
            data_tensor = data_tensor[0].unsqueeze(0)  # 取第0个通道
            # 或者平均：
            # data_tensor = data_tensor.mean(dim=0, keepdim=True)
        elif data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(0)  # (2800,) → (1, 2800)

        return data_tensor, label_tensor

if __name__ == '__main__':
    dataset = PKLDataset(txt_path='datasets/source/test/DC_T197_RP.txt')
    print(len(dataset))  # 样本总数
    x, y = dataset[1]  # 获取第一个样本
    print(x.shape,x, y)  # 打印数据和标签