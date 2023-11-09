# ML-SECU - Final Project

> Datasets can be found on the moodle page `Machine Learning pour la Cybersécurité`

Please rename both datasets folders like below to ensure correct loading:
- `HardwareInTheLoop`
- `SecureWaterTreatment`

## Authors

- Arnaud Baradat
- Tom Genlis
- Théo Ripoll
- Quentin Fisch

## Architecture

- src/
    - data_exploration/ : all our notebooks (don't mind the bad architecture)
        - preprocess_data.py : utils to load the datasets
        - exploration.ipynb : EDA of the datasets
        - network.ipynb : models analysis and training on the network dataset
        - physical.ipynb : models analysis and training on the physical dataset
        - demo.ipynb : smaller notebooks for the live demo (subset of the previous notebooks)
- data/
    - *.pdf: PDFs of the two datasets
    - **Locally**: HardwareInTheLoop/
    - **Locally**: SecureWaterTreatment/
