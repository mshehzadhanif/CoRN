**Competitive Residual Neural Network**
---------------------------------------

Hanif MS, Bilal M. Competitive residual neural network for image classification. ICT Express. 2020 Mar 1;6(1):28-37.

Find the full article [here](https://www.sciencedirect.com/science/article/pii/S2405959519300694).

Abstract
--------
We propose a novel residual network called competitive residual network (CoRN) for image classification. The proposed network is composed of residual units which are made up of two identical blocks each containing convolutional filters, batch normalization and a maxout unit. Contrary to the ResNet with rectified linear units (ReLUs) in the residual units, the maxout activation function in our residual units not only enables the competition among the convolutional filters but also reduces the dimensionality of the convolutional layer. Our experimental study includes the performance analysis of several deep and wide variants of our proposed network on CIFAR-10, CIFAR-100 and SVHN benchmark datasets. The proposed network outperforms the original ResNet by a sufficiently large margin and test errors on the benchmark datasets are comparable to the recent published works in the domain. Using the ensemble network, we achieve a test error of 3.85% on CIFAR-10, 18.17% on CIFAR-100 and 1.59% on SVHN.

Evaluation Code: 
----------------

eval_model.py (usage: python eval_model.py [dataset_name] [path_to_model])

preprocessing.py (for CIFAR-10 and CIFAR-100 mean/std preprocessing) 

Requirements
------------

tensorflow 1.4 (tf.keras)

numpy 1.13 or newer

h5py 2.7.1 or newer

scipy 0.19.1 or newer

six 1.11.0 or newer



