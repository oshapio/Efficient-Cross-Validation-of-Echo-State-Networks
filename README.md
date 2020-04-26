# Efficient Implementations of Echo State Network Cross-Validation
Echo State Networks (ESNs) as a prime example of Reservoir Computing (RC) models are known for their fast and precise one-shot learning of time series. But they often need good hyper-parameter tuning for the best performance. For this good validation is key, but usually a single validation split is used. 

This repository contains the code used for the experiments in [add reference, years, authors]

## 1. Summary of best complexities of the different CV methods

<p align="center"><img align="center" src="https://i.imgur.com/M0ZfzBH.png"></p>

## 2. Precision Experiments

Some details of implementation and computational savings depend on what type of task we are learning. Let us distinguish three types of temporal machine learning tasks:

1. <b>Generative</b> tasks, where the computed output <i><b>y</b>(n)</i> comes back as (part of) the input <i><b>u</b>(n + k)</i>. This is often pattern generation or multi-step timeseries prediction in a generative mode.

2. <b>Output</b> tasks, where the computed output time series <i><b>y</b>(n)</i> does not comeback as part of input. This is often detection or recognition in time series, or deducing a signal from other contemporary signals.

3. <b>Classification</b> tasks, of separate (pre-cut) finite sequences, where a class <i><b>y</b></i> is assigned to each sequence <i><b>u</b>(n)</i>. 

For the latter type of tasks we usually store only an averaged or a fixed number of states <i><b>x</b>(n)</i> for every sequence in the state matrix <b>X</b>. It is similar to a non-temporal classification task.

For each of the learning task we apply our approach to verify its effectiveness. The code for each task is similar, yet divided into separate files to make application to each task clear. As such, each script can be easily alterted to support other datasets with minimal change. 

<!-- <p align="center"><img align="center" src="https://i.imgur.com/tObv3JZ.png"></p> -->

## 2.1. Generative Mode

### Usage
Exact experiments reported in the paper can be replicated by running:

`./run_generative_experiments.sh`

This will start a job with 4 threads and search for optimal hyper-parameters with 5 different initializations. 
After the search ends, all of the results will be saved in `precision_experiments/generative_mode/results` folder.
Results can be viewed by running [show_results.py](https://github.com/oshapio/testing/blob/master/precision_experiments/generative_mode/show_results.py).

Alternatively, individual experiments, for example, can be run by executing:

`python precision_experiments/generative_mode/generative_exp.py --dataset gasoline --valid_method standard --final_model retrain`

Full range of options can be seen by adding `--help` argument.

### Results
<p align="center"><img align="center" src="https://i.imgur.com/75CIBqz.png"></p>

## 2.2. Time Series Output

### Usage
Same structure was followed as in `Generative Mode` experiment, see how individual experiments were executed in `run_output_exps.sh`.
### Results
<p align="center"><img align="center" src="https://i.imgur.com/EYRsk2P.png"></p>


## 2.3. Time Series Classification

### Usage
Same structure was followed as in `Generative Mode` experiment, see how individual experiments were executed in `run_classification_exps.sh`.
### Results
<p align="center"><img align="center" src="https://i.imgur.com/sWjUu2v.png"></p>


## 3. Speed Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/hierarchical_probabilistic_unet/HPU_Net.ipynb)

A notebook ([ESN_speed_experiment.ipynb](/ESN_speed_experiment.ipynb)) is available to replicate the speed experiment. Alternatively, experiment can be run locally by clicking the `Open in Colab` button.

<p align="center"><img align="center" src="https://i.imgur.com/A1Iyro8.png"></p>


## Contact

Please contact Arnas Uselis (auselis gmx com) if any issues regarding the code arise.
