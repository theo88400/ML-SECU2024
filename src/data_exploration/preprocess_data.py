import pandas as pd

def get_HITL(path="data/HardwareInTheLoop/", small=False):
    """
    Loads the HITL dataset from the given path.
    :param path: Path to the HITL dataset.
    :param smaller: Whether to load a 1/100th of the HITL dataset.
    :returns: Dictionary with the following keys:
        - network_dataset_attack: List of the three attack network datasets. (~ 5M rows each)
        - network_dataset_normal: Normal network dataset. (~ 7M rows)
        - physical_dataset_attack: List of the three attack physical datasets. (~ 3000 rows each)
        - physical_dataset_normal: Normal physical dataset. (~ 3000 rows)
    """

    skiprows = lambda x: x % 100 != 0 if small else None
    res = {}

    # Network dataset
    network_dataset = []
    for i in range(1, 4):
        network_dataset.append(pd.read_csv(path + "Network datatset/csv/attack_" + str(i) + ".csv", skiprows=skiprows))
    
    res["network_dataset_attack"] = network_dataset
    res["network_dataset_normal"] = pd.read_csv(path + "Network datatset/csv/normal.csv", skiprows=skiprows)

    # Physical dataset
    physical_dataset = []
    for i in range(1, 4):
        physical_dataset.append(pd.read_csv(path + "Physical dataset/phy_att_" + str(i) + ".csv", skiprows=skiprows, delimiter=";"))
    
    res["physical_dataset_attack"] = physical_dataset
    res["physical_dataset_normal"] = pd.read_csv(path + "Physical dataset/phy_norm.csv", skiprows=skiprows, delimiter=";")

    return res

def get_SWT(path="data/SecureWaterTreatment/", small=False):
    """
    Loads the SWT dataset from the given path.
    :param path: Path to the SWT dataset.
    :param small: Whether to load a 1/100th of the SWT dataset.
    :returns: List of the four SWT datasets.
              Shapes: (14400, 82), (3600, 82), (7201, 16382), (7201, 61)
    """
    res = []
    skiprows = lambda x: x % 100 != 0 if small else None
    res.append(pd.read_excel(path + "22June2020 (1).xlsx"), skiprows=skiprows)
    res.append(pd.read_excel(path + "22June2020 (2).xlsx"), skiprows=skiprows)
    res.append(pd.read_excel(path + "29June2020 (1).xlsx"), skiprows=skiprows)
    res.append(pd.read_excel(path + "29June2020 (2).xlsx"), skiprows=skiprows)

    return res

def _clean_HITL(hitl_dict: dict, is_network: bool = True):
    """
    Cleans the HITL datasets in a dictionary.
    :param hitl_dict: containing 3 lists of attack datasets and 1 normal dataset.
    :param is_network: Whether the dataset is the network dataset.
    """
    prefix = "network_" if is_network else "physical_"

    dataset_attack = hitl_dict[prefix + "dataset_attack"] # 4 attacks recorded from the network perspective
    dataset_normal = hitl_dict[prefix + "dataset_normal"] # 1 normal recording from the network perspective (no attack)
    for dataset in [*dataset_attack, dataset_normal]:
        dataset.rename(columns=lambda x: x.strip().lower(), inplace=True) # remove whitespace from column names
    
    # Check unmatched dtypes
    normal_dtypes = dataset_normal.dtypes
    attack_dtypes = dataset_attack[0].dtypes
    mask = attack_dtypes == normal_dtypes
    
    # Update to broader dtype to avoid errors
    dataset_normal[normal_dtypes[~mask].index] = dataset_normal[normal_dtypes[~mask].index].astype("float64")

    # Set which line if from which dataset
    for ds in dataset_attack:
        ds["attack"] = 1
    dataset_normal["attack"] = 0

    # Concatenate all datasets
    dataset = pd.concat([*dataset_attack, dataset_normal], ignore_index=True)
    
    # Update a wrong column if physical dataset
    if not is_network:
        # There is a third column named lable_n which seems to be the same as label_n
        ds_label_n = dataset["label_n"]
        ds_lable_n = dataset["lable_n"]

        # Print both column's respective non NaN indexes to see if they are complementary
        physical_label_n_indexes = ds_label_n[~ds_label_n.isna()].index
        physical_lable_n_indexes = ds_lable_n[~ds_lable_n.isna()].index
        assert len(set(physical_label_n_indexes).intersection(set(physical_lable_n_indexes))) == 0 # Both columns are complementary

        # Merge the two columns
        dataset["label_n"] = ds_label_n.fillna(ds_lable_n)

        assert dataset["label_n"].isna().sum() == 0

        dataset = dataset.drop(columns=["lable_n"])
    
    # Convert time to timestamp
    dataset["time"] = pd.to_datetime(dataset["time"], format="mixed")
    dataset["time"] = dataset["time"].apply(lambda x: x.timestamp())

    return dataset

def clean_HITL(hitl_dict: dict):
    """
    Returns the cleaned hitl datasets.
    :param hitld_dict: the HITL dict from get_HITL
    :returns: tuple (network_dataset, physical_dataset)
    """
    network_dataset = _clean_HITL(hitl_dict, is_network=True)
    physical_dataset = _clean_HITL(hitl_dict, is_network=False)
    return network_dataset, physical_dataset