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
        physical_dataset.append(pd.read_csv(path + "Physical dataset/phy_att_" + str(i) + ".csv", skiprows=skiprows))
    
    res["physical_dataset_attack"] = physical_dataset
    res["physical_dataset_normal"] = pd.read_csv(path + "Physical dataset/phy_norm.csv", skiprows=skiprows)

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