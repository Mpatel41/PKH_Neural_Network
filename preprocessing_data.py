#Manipulating the Data 

#Import necessary libraries 
import pandas as pd

#Read in the Data
dataset = pd.read_csv("out(in).csv")

#Manipulate the Dataset
#Create Labels for the output to be used in classification 
#Organize which columns to be included in the dataset
dataset['Labels'] = dataset['totflux_A_1000']
dataset.loc[dataset['Labels'] <= 0, 'Labels' ] = 0 
dataset.loc[dataset['Labels'] > 0, 'Labels' ] = 1
classification_dataset = dataset.drop(columns=['totflux_A_1000','Deff_membrane','Unnamed: 0','job_id'])
regression_dataset = dataset.drop(columns=['Labels','Deff_membrane','Unnamed: 0','job_id'])

#Save the made dataset files for classification and regression to CSV files 
classification_dataset.to_csv('Classification_Dataset.csv',index = False)
regression_dataset.to_csv('Regression_Dataset.csv',index = False)
