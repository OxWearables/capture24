# Capture24 Dataset and Baselines

This repository contains code for [Capture24: Activity recognition on a large activity tracker data collected in the wild](https://openreview.net/forum?id=RUzBgTQSSvq). Source code for training and evaluating models for activity recognition under different labelling schemes are available here. 

<p align="center">
<img src="wrist_accelerometer.jpg" width="300"/>
</p>



## Data
The raw data of Capture-24 can be downloaded [here](https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11). We have prepared `tutorial.ipynb` and `tutorial.py` for data exploration on the raw dataset. 

## Raw data
There are a total of 153 files: 

- `annotation-label-dictionary.csv`: A table containing the correspondence of fine-grained to coarse activity label under 6 different annotation schemes. 
- `metadata.csv`: A table containing the age group and gender of each user. 
- 151 `P{USER_ID}.csv.gz` files: Each file contains data from one user. The columns are ['time', 'x', 'y', 'z', 'annotation'], which refers to the timestamp of each record, the triaxial acceleration values measured in `g`, and the fine-grained annotation string (which can be mapped to coarse-grained class labels using the `annotation-label-dictionary.csv`)

### Processing the raw data
After the raw data is downloaded and saved in `RAW_DATA_PATH`, run the following script for segmenting the data into sliding windows to prepare the processed dataset for running machine learning models. The default parameters are for creating window of 10-second sizes. 
```python prepare_data.py --datadir {RAW_DATA_PATH}```

## Training and evaluation 

### Deep Learning models
We use `hydra` to configure all hyperparameters for running the experiments, the configuration files are located in `config/`.

Use the following to run CNN models with selected hyperparameters (this will train and evaluate both CNN and CNN+HMM):
```
python main.py model.is_cnnlstm=false
```

Use the following to run RNN models with selected hyperparameters (this will train and evaluate both RNN and RNN+HMM):
```
python main.py
```

### Random Forests models 
Use the following to run RF models with selected hyperparameters (this will train and evaluate both RF and RF+HMM):
```
python rf.py
```


## References

Additional information about the dataset can be found [here](https://github.com/activityMonitoring/capture24_neurips/tree/master/data_info).

Papers that used the Capture-24 dataset:
- [Reallocating time from machine-learned sleep, sedentary behaviour or
light physical activity to moderate-to-vigorous physical activity is
associated with lower cardiovascular disease
risk](https://www.medrxiv.org/content/10.1101/2020.11.10.20227769v2.full?versioned=true)
(Walmsley2020 labels)
- [GWAS identifies 14 loci for device-measured
physical activity and sleep
duration](https://www.nature.com/articles/s41467-018-07743-4)
(Doherty2018 labels)
- [Statistical machine learning of sleep and physical activity phenotypes
from sensor data in 96,220 UK Biobank
participants](https://www.nature.com/articles/s41598-018-26174-1)
(Willetts2018 labels)
