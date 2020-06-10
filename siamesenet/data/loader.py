from .omniglot import load_omniglot


def load(data_dir, config, splits):

    """
    data_dir (str): path to the dataset directory
    config(dict) user-defined settings
    splits(list): ['train', 'val','test']

    return dictionary with keys that are strings in splits
    and values as tensorflow dataset object
    """

    if config['data.dataset'] == "omniglot":
        ds = load_omniglot(data_dir, config, splits)
    else:
        raise ValueError(f"Unknown dataset:{config['data.dataset']}")
    return ds
