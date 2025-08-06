from pathlib import Path
from tqdm import tqdm
from outliers import outliers
import random
import re
from collections import defaultdict


def is_outlier(material: str, thickness: str, target_value: str, index: str):
    """
    Checks if a specific combination of material, thickness, target value, and index is marked as an outlier.

    Parameters:
        material (str): The type of material being checked.
        thickness (str): The thickness of the material being checked.
        target_value (str): The target value related to the data being checked.
        index (int): The index representing a specific instance of data.

    Returns:
        bool: True if the combination is found in the outliers list, False otherwise.
    """
    return f'{material}_{thickness}_{target_value}_{index}' in outliers

def dataset_split(dataset_path: str, output_path: str, material: str, thickness: str, typ: str):
    """
    Splits a dataset by filtering out outliers and writes the list of valid file paths to a text file.

    Parameters:
        dataset_path (str): The base path to the dataset directory.
        output_path (str): The directory where the output text file will be saved.
        material (str): The material type to filter within the dataset.
        thickness (str): The thickness of the material to filter.
        typ (str): The type/category of the data (e.g., 'AE', 'RP').

    Returns:
        None: The function writes the filtered file paths to a text file and does not return any value.

    Function Details:
        - Collects all `.pkl` files from the specified folder (based on material, thickness, and type).
        - Filters out files that are identified as outliers using the `is_outlier` function.
        - Appends the absolute paths of the remaining valid files to a list.
        - Writes the absolute paths of valid files to a text file named according to the material, thickness, and type in the `output_path` directory.
    """
    dataset_path = Path(dataset_path)

    folder = dataset_path / material / thickness / typ

    file_list = list(folder.glob('*.pkl'))

    C = []

    for file_path in file_list:

        target_value = file_path.stem.split('_')[0]
        i = file_path.stem.split('_')[1]

        if is_outlier(material, thickness, target_value, i):
            continue

        C.append(file_path.absolute())

    s = ""
    for i in tqdm(C, desc=f'write datasetset to txt：{folder}'):
        s += f'{i}\n'
    s = s[:-1]
    with open(Path(output_path) / f'{material}_{thickness}_{typ}.txt', 'w') as f:
        f.write(s)

def write_file():
    """
    create file
    :return: no
    """
    for typ in ['RP', 'AZ', 'AE']:
        dataset_split('./preprocessing', './datasets', 'DC', 'T197', typ)
        dataset_split('./preprocessing', './datasets', 'DC', 'T194', typ)
        dataset_split('./preprocessing', './datasets', 'DC', 'T191', typ)
        dataset_split('./preprocessing', './datasets', 'DC', 'T188', typ)
        dataset_split('./preprocessing', './datasets', 'DC', 'T185', typ)

        dataset_split('./preprocessing', './datasets', 'HC', 'T197', typ)
        dataset_split('./preprocessing', './datasets', 'HC', 'T194', typ)
        dataset_split('./preprocessing', './datasets', 'HC', 'T191', typ)
        dataset_split('./preprocessing', './datasets', 'HC', 'T188', typ)
        dataset_split('./preprocessing', './datasets', 'HC', 'T185', typ)

def split_source_save_all(input_txt, output_train, output_rest, split_ratio=0.8, seed=42):
    """
    Split the dataset paths in the input text file into train (80%) and rest (20%)
    for each Rxx category, and save them into two separate files.
    A fixed random seed ensures reproducibility.

    Parameters:
        input_txt (str): Path to the input text file containing dataset paths.
        output_train (str): Path to the output file for the training set.
        output_rest (str): Path to the output file for the remaining set.
        split_ratio (float): Proportion of training data (default 0.8).
        seed (int): Random seed to ensure reproducibility (default 42).
    """
    random.seed(seed)  # Set random seed for reproducibility

    # Read all paths from the input file
    with open(input_txt, "r", encoding="utf-8") as f:
        all_paths = [line.strip() for line in f if line.strip()]

    # Group paths by Rxx category
    category_dict = defaultdict(list)
    for p in all_paths:
        match = re.search(r"(R\d+)_", Path(p).stem)
        if match:
            category = match.group(1)
            category_dict[category].append(p)

    # Ensure output directories exist
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(output_rest).parent.mkdir(parents=True, exist_ok=True)

    # Split and write paths into train and rest files
    with open(output_train, "w", encoding="utf-8") as f_train, open(output_rest, "w", encoding="utf-8") as f_rest:
        for category, files in category_dict.items():
            sample_size = max(1, int(len(files) * split_ratio))
            train_files = random.sample(files, sample_size)
            rest_files = list(set(files) - set(train_files))

            for path in train_files:
                f_train.write(path + "\n")
            for path in rest_files:
                f_rest.write(path + "\n")

    print(f"Training set saved to {output_train}")
    print(f"Remaining set saved to {output_rest}")

def split_data(input_txt, output_train, split_ratio=0.8, seed=42):
    """
    Split the dataset paths in the input text file into a training set only
    (default 80%) for each Rxx category, and save them into one file.
    A fixed random seed ensures reproducibility.

    Parameters:
        input_txt (str): Path to the input text file containing dataset paths.
        output_train (str): Path to the output file for the training set.
        split_ratio (float): Proportion of training data (default 0.8).
        seed (int): Random seed to ensure reproducibility (default 42).
    """
    random.seed(seed)  # Set random seed for reproducibility

    # Read all paths from the input file
    with open(input_txt, "r", encoding="utf-8") as f:
        all_paths = [line.strip() for line in f if line.strip()]

    # Group paths by Rxx category
    category_dict = defaultdict(list)
    for p in all_paths:
        match = re.search(r"(R\d+)_", Path(p).stem)
        if match:
            category = match.group(1)
            category_dict[category].append(p)

    # Ensure output directory exists
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)

    # Split and write paths into train file
    with open(output_train, "w", encoding="utf-8") as f_train:
        for category, files in category_dict.items():
            sample_size = max(1, int(len(files) * split_ratio))
            train_files = random.sample(files, sample_size)

            for path in train_files:
                f_train.write(path + "\n")

    print(f"Training set saved to {output_train}")

if __name__ == '__main__':
    # write_file()
    input_txt = "datasets/source/val/DC_T197_RP.txt"  # 你的输入文件
    output_train = "datasets/source/validation/DC_T197_RP.txt"
    output_rest = "datasets/source/test/DC_T197_RP.txt"
    split_source_save_all(input_txt, output_train, output_rest, split_ratio=0.5,seed=42)


