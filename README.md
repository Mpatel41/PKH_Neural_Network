# PKH Neural Network
PKH Neural Network Project by Tessa Patton, Fiore Cassettari, Kayla Jarm, and Mansi Patel under the guidance of Dr. Heather Wheeler and Dr. Peter Kekenes-Huskey. 

## Introduction 

The movement of treated macrophages to a tumor region impacts the efficacy of their immune response. Here, we introduce four models that predict the ability of a macrophage to infiltrate a tumor region given different environmental conditions, including physical space and chemoattraction.

***For more information about the Introduction, Dataset, and project workflow please look at the Wiki page.***

----------------------------------------------------------------------------------------------------------------------------------------------

## Dependencies Used 
- os
- Subprocess
- Sys
- Pandas - 2.1.4
- Numpy - 1.26.3
- Scikeras -  0.13.0
- Keras - 3.2.1
- Scikit learn - 1.4.2
- Tensorflow - 2.16.1

------------------------------------------------------------------------------------------------------------------------------------------------

## Softwares 

Scikit-learn - Python machine leaarning library supporting several algorithms
https://scikit-learn.org/stable/index.html

Tensorflow - Python machine learning library for training neural networks
https://www.tensorflow.org/

Scikeras - Makes it possible to use Keras/tensorflow with scikit-learn
https://pypi.org/project/scikeras/

Keras - machine learning library
https://keras.io/

Pandas - data analysis and manipulation library
https://pandas.pydata.org/docs/user_guide/index.html

Numpy - data manipulation https://numpy.org/doc/

---------------------------------------------------------------------------------------------------------------------------------------------------

## Description of the Scripts 

dependencies_installation.py -> Install Keras, Tensorflow, Scikeras, and Scikit-Learn

preprocessing_data.py -> Data cleanup, Dropping the unwanted columns and creating labels. Outputs 2 .csv files that are the classification and regression datasets. No need to run this code as the datasets have been provided in this GitHub 

Classification_Neural_Network.py -> Classification Neural Network Model that 2 hidden layers. Outputs Accuracy and F1 score to Results.log

Regression_Neural_Network.py -> Regression Neural Network Model that 2 hidden layers. Outputs MSE score to Results.log

SVM_Classification.py -> SVM Classification Model. Outputs Accuracy and F1 score to Results.log

SVM_Regression.py -> SVM Regression Model. Outputs MSE score to Results.log

---------------------------------------------------------------------------------------------------------------------------------------------------

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
This script will take about 2-3 minutes to run. If this script gives any error when installing any dependencies, use ```pip install [dependency]```

This script installs the major machine learning dependencies. If you are missing any other than the ones listed in this script please use ``` pip install ```




If you want to run all the scripts of Neural Network Classification, Neural Network Regression, SVM Classification, and SVM Regression, run these 2 commands on the command line


```python
python SVM_Classification.py Classification_Dataset.csv ; python SVM_Regression.py Regression_Dataset.csv ; python Classification_Neural_Network.py Classification_Dataset.csv ; python Regression_Neural_Network.py Regression_Dataset.csv
```

This script should run within 5 minutes


---------------------------------



If you want to run one of them or run them seperately, run the code corresponsing to the model you want to run.  

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

**Classification Metrics**

Accuracy -
Applied to both the SVM and Neural Network classification models. Accuracy is used to measure the overall correctness of the model in predicting the flow of migratory cells (whether it enters the tumor region or not).

F1 Score -
Applied to both the SVM and Neural Network classification models. This is useful in assessing the precision and performance of the model especially when the dataset might have a distribution imbalance.The F1 score ensures that the model is both accurately identifying the patterns of migration for the macrophages.

**Regression Metrics**

Mean Squared Error -
Applied to both the SVM and Neural Network classification models. It is useful in quantifying the magnitude of the error in prediction.

----------------

**The output file is named Results.log that can be accessed in the PKH_Neural_Network Directory. It contains the accuracy and f1 scores for the Neural Network Classification and SVM classification model. Additionally, it contains the MSE score for Neural Network Regression and SVM Regression model.**

```python
less -S Results.log
```



---------------------------------------------------------------------------------------------------------------------------------------------------
## Acknowlegements

Thank you to Dr. Heather Wheeler, Dr. Peter Kekenes-Huskey, and COMP 383 class members for making this project possible 


