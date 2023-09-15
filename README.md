# Uncertainty-Quantification-in-Model-Based-DL

This repository includes the source code used in our paper:

Yehonatan Dahan, Guy Revach, Jindrich Dunik, and Nir Shlezinger. "[Uncertainty Quantification in Deep Learning Based Kalman Filters](https://arxiv.org/abs/2309.03058)." (2023).


## Abstract

Various algorithms combine deep neural networks (DNNs) and Kalman filters (KFs) to learn from data to track in complex dynamics. Unlike classic KFs, DNN-based systems do not naturally provide the error covariance alongside their estimate, which is of great importance in some applications, e.g., navigation. To bridge this gap, in this work we study error covariance extraction in DNN-aided KFs. We examine three main approaches that are distinguished by the ability to associate internal features with meaningful KF quantities such as the Kalman gain (KG) and prior covariance. We identify the differences between these approaches in their requirements and their effect on the training of the system. Our numerical study demonstrates that the above approaches allow DNN-aided KFs to extract error covariance, with most accurate error prediction provided by model-based/data-driven designs.


## Overview

This repository consists of following Python scripts:
* `Main.py` the interface for applying both training and test for the different State-Spaces presented in our paper.
* `config.ini` configuration for running `Main.py` script. Further details about the parameters could be found in `config.md` [file](https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL/blob/main/config.md).
* `GSSFiltering/dnn.py` defines deep neural network (dnn) architectures: KalmanNet and Split-KalmanNet.
* `GSSFiltering/filtering.py` handles the filtering algorithms for the dnns and the extended kalman filter.
* `GSSFiltering/model.py` defines the State-Space model's parameters.
* `GSSFiltering/tester.py` handles the testing method.
* `GSSFiltering/trainer.py` handles the training method.


## Requirements

All required packages are listed in [requirement.txt](https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL/blob/develop/requirements.txt) file.


## Getting Started

To simply run the code, define the desired configuration in `config.ini` and execute `Main.py`.


#### Notes
+The Recurrent Kalman Network (RKN) comparison in the paper was done with respect to [RKN](https://github.com/ALRhub/rkn_share).  
+This code is based on the [Split-KalmanNet](https://github.com/geonchoi/Split-KalmanNet) code.  
