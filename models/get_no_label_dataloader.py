import torch
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader ,TensorDataset






class NoLabelDataset(torch.utils.data.Dataset):
    """
       A wrapper dataset that removes labels from a labeled dataset.

       Args:
           signal_dataset (Dataset): A dataset returning (signal, label) tuples.
       """
    def __init__(self, signal_dataset):
        self.signal_dataset = signal_dataset

    def __len__(self):
        return len(self.signal_dataset)

    def __getitem__(self, idx):
        signal, _ = self.signal_dataset[idx]
        return signal


def get_target_loader(path = None ,batch_size=64 , shuffle=True):
    """
        Loads unlabeled target domain data as a DataLoader for inference or pseudo-labeling.

        This function reads the target dataset from the given txt path,
        removes labels using NoLabelDataset, and returns a DataLoader that yields only input signals.

        Args:
            path (str): Path to the txt file listing .pkl sample paths (target domain).
            batch_size (int): Batch size for the DataLoader.

        Returns:
            DataLoader: A DataLoader yielding batches of input signals without labels.
        """

    dataset = PKLDataset(txt_path=path)
    no_label_dataset = NoLabelDataset(dataset)
    loader = DataLoader(no_label_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader