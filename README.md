# PKH_Neural_Network
PKH Neural Network Project 

## Introduction 

The movement of treated macrophages to a tumor region impacts the efficacy of their immune response. Here, we introduce four models that predict the ability of a macrophage to infiltrate a tumor region given different environmental conditions, including physical space and chemoattraction.

## DataSet 

- What are we looking at? Picture? 


## Softwares
matplotlib - Python API for plotting and data visualization

scikit-learn - Python machine leaarning library supporting several algorithms

tensorflow - Python machine learning library for training neural networks

scikeras - Makes it possible to use Keras/tensorflow with scikit-learn


## Instructions

On the Command Line: 

Git Clone - 

```python
git clone https://github.com/Mpatel41/PKH_Neural_Network.git
```

Navigate to this directory 

```python 
cd PKH_Neural_Network
```
Install all the packages required to run the script 

```python
python dependencies_installation.py
```

Run the following script for one of the models - 

```python
python SVM_Classification.py Classification_Dataset.csv
```
```python
python SVM_Regression.py Regression_Dataset.csv
```
```python
python Classification_Neural_Network.py Classification_Dataset.csv
```
```python
python Regression_Neural_Network.py Regression_Dataset.csv
```


## Output 

Accuracy -
Applied to both the SVM and Neural Network classification models. Accuracy is used to measure the overall correctness of the model in predicting the flow of migratory cells (whether it enters the tumor region or not).

F1 Score -
Applied to both the SVM and Neural Network classification models. This is useful in assessing the precision and performance of the model especially when the dataset might have a distribution imbalance.The F1 score ensures that the model is both accurately identifying the patterns of migration for the macrophages.

Mean Squared Error -
Applied to both the SVM and Neural Network classification models. It is useful in quantifying the magnitude of the error in prediction.

