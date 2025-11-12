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


def get_target_loader(path = None ,batch_size=64 , shuffle=True, drop_last=True):
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
    loader = DataLoader(no_label_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
    return loader


def get_dataloaders(source_path, target_path, batch_size):
    """
            Construct the DataLoaders for the source domain and target domain.

                Parameters:
                    source_path (str): Path to the txt file of the source domain data.
                                       Each line usually contains the sample file path and its label.
                    target_path (str): Path to the txt file of the target domain data.
                                       The target domain usually has no labels.
                    batch_size (int): Number of samples in each batch.

                Returns:
                    tuple:
                        - source_loader: DataLoader of the source domain,
                          which returns batches in the format (x, y).
                        - target_loader: DataLoader of the target domain,
                          which returns x (without labels).


            """
    source_dataset = PKLDataset(txt_path=source_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader
def get_pseudo_dataloaders( target_path, batch_size):

    target_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=False, drop_last=False)
    return  target_loader
def get_source_p_dataloaders(source_path, batch_size = 128):
    source_dataset = PKLDataset(txt_path=source_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    return source_loader