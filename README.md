# XGboost Notebooks

This repo has a bank of notebooks related to XGboost

### 1. Notebooks

### 2. Additional Info about XGboost algorithm, data input, pre-requisites..

#### 2.1 Pre-requisite
Install a virtual environment, use python 1.6+ and run: `pip install xgboost`

Python pre-built binary capability for each platform:
![xgboost-platforms](images/xgboost-platforms.png)

#### 2.2 XGboost Code Snippet
Code snippet to quickly try out XGBoost on the demo dataset on a binary classification task.
```python
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
```

#### 2.3 Text Input Format of DMatrix

XGBoost currently supports two text formats for ingesting data: `LIBSVM` and `CSV`. The rest of this document will describe the LIBSVM format.

Please be careful that, **XGBoost does not understand file extensions, nor try to guess the file format**, as there is no universal agreement upon file extension of LIBSVM or CSV. Instead it employs URI format for specifying the precise input file type. For example if you provide a csv file ./data.train.csv as input, XGBoost will blindly use the default LIBSVM parser to digest it and generate a parser error. Instead, users need to provide an URI in the form of `train.csv?format=csv`. For external memory input, the URI should of a form similar to `train.csv?format=csv#dtrain.cache`

For training or predicting, XGBoost takes an instance file with the format as below:

train.txt
```
1 101:1.2 102:0.03
0 1:2.1 10001:300 10002:400
0 0:1.3 1:0.3
1 0:0.01 1:0.3
0 0:0.2 1:0.3
```

Each line represent a single instance, and in the first line ‘1’ is the instance label, ‘101’ and ‘102’ are feature indices, ‘1.2’ and ‘0.03’ are feature values. In the binary classification case, ‘1’ is used to indicate positive samples, and ‘0’ is used to indicate negative samples. We also support probability values in [0,1] as label, to indicate the probability of the instance being positive.




#### 2.4 What is XGboost?
**[XGBoost](https://xgboost.readthedocs.io/en/latest/)** stands for e`X`treme `G`radient `Boost`ing.

It is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.

**XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, **GBM**) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.


It is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by **combining an ensemble of estimates from a set of simpler and weaker models**. The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune. You can use XGBoost for regression, classification (binary and multiclass), and ranking problems.

- **Model Features**:
The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:

  - Gradient Boosting algorithm also called gradient boosting machine including the learning rate.
  - Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.
  - Regularized Gradient Boosting with both L1 and L2 regularization.


 - **System Features**:
The library provides a system for use in a range of computing environments, not least:

  - Parallelization of tree construction using all of your CPU cores during training.
  - Distributed Computing for training very large models using a cluster of machines.
  - Out-of-Core Computing for very large datasets that don’t fit into memory.
  - Cache Optimization of data structures and algorithm to make best use of hardware.

 - **Algorithm Features**:
The implementation of the algorithm was engineered for efficiency of compute time and memory resources. A design goal was to make the best use of available resources to train the model. Some key algorithm implementation features include:

  - Sparse Aware implementation with automatic handling of missing data values.
  - Block Structure to support the parallelization of tree construction.
  - Continued Training so that you can further boost an already fitted model on new data.
XGBoost is free open source software available for use under the permissive Apache-2 license.

- **Why Use XGBoost?**
The two reasons to use XGBoost are also the two goals of the project:

  - Execution Speed.
  - Model Performance.

XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. tree-basedHowever, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. Please see the chart below for the evolution of tree-based algorithms over the years.

Note - Ensemble learning methods can be performed in two ways:
- **Bagging** (parallel ensemble)
- **Boosting** (sequential ensemble)

![tree-based algos](images/tree-based-algos.jpeg)

### 3. Resources
 - [XGboost documentation](https://xgboost.readthedocs.io/en/latest/)
 - [Text Input Format of DMatrix](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
 - [XGBoost Tutorials (includes Distributed with Dask)](https://xgboost.readthedocs.io/en/latest/tutorials/index.html)
 - [XGBoost Demo Notebooks](https://github.com/dmlc/xgboost/tree/master/demo)
 - [Kaggle from A to Z with XGBoost (Tutorial)](https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial#)
 - [On-Premise Machine Learning with XGBoost Explained](https://towardsdatascience.com/on-premise-machine-learning-with-xgboost-explained-5adfdfcfec77)
 - [Avoid Overfitting By Early Stopping With XGBoost In Python](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/)
