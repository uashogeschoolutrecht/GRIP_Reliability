# GRIP_Human_Activity_Patterns


The GRIP (beweeGsensoren voor mensen met chRonIsche Pijn) project, aims to use movement sensors (accelerometers) in order to model the complex activity patterns of people with Chronic Pain (CP). 
Several challenges arise in this project, for example lack of enough labeled data that allows for a Machine Learning (ML) algorithm to properly learn about the activity patterns of people with CP. 

The Data Science Pool (DSP) team collaborates with researchers from the [Lifestyle and Health](https://www.internationalhu.com/research/lifestyle-and-health) Department at the University of Applied Sciences in this effort.

In this repo we keep a centralized Sandbox, where we experiment with open source datasets in order to gather further understanding about the technicalities of working with sensor data, in our case accelerometers.

Many parts of the code are simple jupyter notebooks where we perform the experiments; we use openly available datasets and code for our tests.


## Data processing of lab data

We are going to process the data obtained from the laboratory so that it can be used as suitable input for training the neural network. The processing detials are explianed in docs/ML_GRIP_Tech_Documentation.pdf. 

### Data cleaning

In order to get clean data for each participant, we can run main.py in src folder. The cleaned data will be stored in data\processedData\cleanData. 

### Data combining

Then we can choose which subjects we want to combine. If run src/combine.py, the script will combine all the cleaned data in data\processedData\cleanData. The combined data can be directly used for our neural network. The combined data is saved in \data\processedData\combinedData.

## Machine learning training, validation and test

We are going to do the machine learning model training, validation and test in direction models. Subjects that you want to retain until final testing can be indicated in test_subjects.

### Cross validation

For cross validation, we can run models_main.py. The parameters can be changed in config/config_file.yaml and config/config_model.yaml. We can change number of folds, epochs, etc. 

### Final test

For final test, we can run final_test.py. 

### Generating plots

For loading the machine learning model to generate plots, we can run load_model.py with pain scores or without.

The data structure of the project can be found in [Research Drive](https://hu.data.surfsara.nl/index.php/apps/files/?dir=/23016683_Data_Science_Pool%20(Projectfolder)/Data%20Structure%20of%20GRIP_HAP_Development&fileid=43744009).







    
