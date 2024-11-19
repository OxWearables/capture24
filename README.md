# CAPTURE-24 Benchamrk

The purpose of the CAPTURE-24 dataset is to serve as a training dataset for developing Human Activity Recognition (HAR) classifiers. 
These classifiers enable the identification of activities performed by individuals wearing smartwatch accelerometers during unmonitored free-living, such as in the UK Biobank physical activity monitoring study. 
Our goal is to ensure the use of the best possible model for accurate activity identification, with generalizable performance across different individuals and behaviors.
For more information on this dataset, you can go to the [webpage](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) or [github repo](https://github.com/OxWearables/capture24).

The purpose of this repository is report a benchmark for trained HAR models on the CAPTURE-24 dataset. 
This can be done by either running through the [Benchmark.ipynb](Benchmark.ipynb) notebook, or by running through the set of command line scripts listed in the [Implementation](#implementation) section below. 

## Methods

The flow diagram below illustrates the pipeline for the HAR model, developed and trained to identify activities from raw accelerometer data.

![Flow diagram of baseline HAR model](ModelFlow.png)
*Flow diagram of the baseline HAR model*

The numbered stages of this analysis present various possibilities and considerations, which are explained below.

### Stage 1: Participant accelerometery
<img src="wrist_accelerometer.jpg" width="300">

The CAPTURE-24 dataset contains wrist-worn accelerometer data collected from 151 participants. To obtain ground truth annotations, the participants also wore a body camera during daytime, and used sleep diaries to register their sleep times. 
Each participant was recorded for roughly 24 hours. The accelerometer was an Axivity AX3 wrist watch (image above) that mearures acceleration in all three axes (x, y, z) at a sampling rate of 100Hz. 
The body camera was a Vicon Autographer with a sampling rate of 1 picture every 20 seconds. 
Note that the camera images are not part of the data release — only the raw acceleration trace with text annotations are provided.

The data file containing data for all participant can be downloaded from: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001, or by using the code provided. Note that this file is over 6.5GB in size, so may take some time to download.

### Stage 2: Data Preparation

#### Windowing

The full accelerometer signal is divided into 10-second windows with no overlap. 
The activity label for each window is determined based on the most frequent label observed within that time window.
The raw activity label was chosen from the [2011 Compendium of Physical Activity](https://sites.google.com/site/compendiumofphysicalactivities/home) and comprises of over 200 unique labels. 
For the purposes of this benchmark however, we look to characterise groups of similar activities, defined by either Willetts or Walmsley.
The dictionary for this mapping of raw annotations to activity labels are found in the downloaded `capture24/annotation-label-dictionary.csv` file. 

#### Feature Extraction

For deep learning models used in this analysis, the feature extraction is learned, eliminating the need for handcrafted features. 
However, for non-deep learning models, handcrafted signal features have been extracted from each data window. 
The list of these signal features can be found in [feature-list.md](feature-list.md).

### Stage 3: Windowed Activity Classification

ML models are chosen and trained to classify activity labels based on the prepared accelerometer data. 
To ensure balanced training, sampling or weighting of loss functions is implemented, preventing the model from being biased solely by the frequency of certain activity labels. 
Note that for stages 3 and 4, a portion of the training data may need to be split into training and validation sets. 
This occurs when a model in either stage 3 or 4 requires a validation dataset. 
Group shuffle split is used to ensure no leakage of participant data.

The table below shows the models used in activity classification along with the percentage of the validation set used:

| Model         | Validation Set Used |
|---------------|---------------------|
| Random forest | 0%                  |
| XGBoost       | 0%                  |
| CNN           | 20%                 |
| RNN           | 20%                 |

### Stage 4: Predicted Label Smoothing

The trained activity classification model outlined prior assumes that windows of data from the same participant are independent and identically distributed (i.i.d.). 
However, this assumption does not hold in reality, as the likelihood of a given activity level depends on previous activity levels. 
To incorporate some temporal information in label prediction and introduce smoothing effects, we can employ a model to learn this dependency. 
Several approaches were considered:

| Model                     | Validation Set Used |
|---------------------------|---------------------|
| No smoothing              | 0%                  |
| Hidden Markov Model (HMM) | 20%*                |

\* For random forest, out of bag decisions can be used, hence does not require a validation set for HMM training.

### Stage 5: Model Evaluation

In this stage, we proceed to evaluate the performance of each model based on a carefully designed process that ensures reliable results.

To begin, we split the dataset into two distinct groups:

* The **Training Group**, consisting of the first 101 participants (P001 to P101), is utilized to train the models on various activities and their corresponding accelerometer data.

* The **Test Group**, comprising the remaining 50 participants (P102 to P151), serves as an independent set to assess how well the trained models generalize to new, unseen data.

The evaluation process incorporates the use of participant-based group bootstrapping. This approach involves the following steps, repeated for a total of 100 iterations:

1. **Random Participant Selection:** In each iteration, we randomly select 50 participants from the Test Group. Since we allow replacement, the same participant's data may be included in multiple iterations, enabling us to capture the variance and diversity within the data.

2. **Model Evaluation:** Using the data from the selected participants, we evaluate the performance of each model. This evaluation provides us with performance scores, which we use to gauge the effectiveness of the models on the unseen data.

After running the 100 iterations, we obtain a distribution of performance scores for each metric. From these distributions, we calculate a 95% confidence interval for each performance metric. The confidence intervals allow us to estimate the range within which the true performance of the models likely lies with 95% confidence.

The performance metrics we consider in this evaluation include:
* Balanced accuracy
* Macro F1 score
* Cohen's kappa score
* Matthews correlation coefficient

## Implementation

### Environment setup
To ensure the installation of all necessary packages used in this repository;

```bash
# create conda environment
$ conda env create -f environment.yml
# activate environment
$ conda activate benchmark
```

### Data preparation

The CAPTURE-24 data must first be prepared.
This means it must be downloaded, extracted, and divided into 10s windows.
To do so, run:

```bash
$ python prepare_data.py -d data -a Walmsley2020,WillettsSpecific2018 --winsec 10 --n_jobs 8
```

This also produces windowed annotations for all activity label definitions that we wish to benchmark. 

These must match to columns in the `capture24/annotation-label-dictionary.csv` file.

When parsed as an argument in the `prepare_data.py` script, all annotations should be separated with a comma (',').

### Model optimisation

Before evaluating our model, we must first tune our hyperparameters to have optimal performance. 
This must be done separately for each model.
Note that optimisation can take a long time.

For the random forest:

```bash
$ python rf.py -d prepared_data -a Walmsley2020 -o optimised_params
```

For XGBoost:

```bash
$ python xgb.py -d prepared_data -a Walmsley2020 -o optimised_params
```

Note this model currently expects hyperparameter tuning to be done per model and per annotation, hence an optimised model for Walmsley2020 will not default to use these parameters for the same model, but for WillettsSpecific2018 labels.

### Model evaluation
To train and evaluate each model and annotation set, run the following:

```bash
$ python benchmark.py -a WillettsSpecific2018,Walmsley2020 -m rf,xgb,rf_hmm,xgb_hmm
```

When parsed as an argument in the `benchmark.py` script, all annotations and models should be separated with a comma (',').

## References
Dataset description and benchmark paper: 
<!-- ```tex
@misc{chan2024capture24,
      title={CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition}, 
      author={Shing Chan and Hang Yuan and Catherine Tong and Aidan Acquah and Abram Schonfeldt and Jonathan Gershuny and Aiden Doherty},
      year={2024},
      eprint={2402.19229},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
``` -->
Chan, S., Hang, Y., Tong, C. et al. CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition. Sci Data 11, 1135 (2024). https://doi.org/10.1038/s41597-024-03960-3

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
