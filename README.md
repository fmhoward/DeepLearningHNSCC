# DeepLearningHNSCC
Deep Learning Survival Models for HNSCC

## Overview
Enclosed is code used for generating models that predict survival after adjuvant therapy for resected head and neck squamous cell carcinoma. Aside from standard dependencies, this code will require installing a modified version of PySurvival (original work by Stephane Fotso): https://github.com/fmhoward/pysurvival

Key functions are as follows:

```
#Generates a limited CSV dataset with only desired features from the larger NCDB dataset
#The input dataset is modified from the standard NCDB dataset to include the following columns
#MTS_LN_DIS - a numeric measurement of number of lymph nodes dissected
#MTS_LN_POS - a numeric measurement of number of positive lymph nodes dissected at surgery
#MTS_TX_PKG_TIME - measurement of time in days from surgery to completion of adjuvant therapy
#MTS_SIMPLE_PRIMARY - a simple wording of primary tumor site, including OC, OPX, HPX, and Larynx
#MTS_SIMPLE_PRIMARY_2 - a simple wording of tumor subsite, including  'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'HP', 'BOT', 'SP', 'PW', 'Ton', 'Glot', 'SubGlot', 'SGL', 'HP PC', 'HP PS', 'HP PW'
#MTS_SURG - a categorical variable that describes the type of surgery undergone; all patients undergoing definitve surgical resection should have 'Surgical resection' in this category
#MTS_RT1 - a categorical variable to describe whether a patient received radiotherapy, should be listed as 'Radiotherapy'
#MTS_RACE - listing race categories
#MTS_HISPANIC - lists 'Hispanic' for patients who are hispanic
#MTS_CDCC - categories of CDCC score, including '0-1', '2', and '>=3'
#MTS_SIMPLE_AGE - category of age, including '<=50y', and '>50y'
#MTS_IMMUNO - category to indicate whether patient received immunotherapy (in which case, listed as 'Immunotherapy' in this column)
#
#filename - the location of the input CSV file
#defResect - specifies to include patients who underwent definitive resection (True), or only those undergoing definitive RT (False)
#verbose - if true, will print additional dataset characteristics
generateData(filename, defResect = True, verbose = False)


#Run to find hyperparameters for an RSF model
#All hyperparameter search functions selects 80% of the dataset for all training purposes
#This 80% is then further divided into 5 equal parts
#A set of random hyperparameters is then chosen
#The model is trained leaving out one of the 5 parts of the 80% of training data, and tested on the part that was left out
#If the model has a higher concordance index for the left out part of the sample, averaged over this 5-fold cross validation, the parameters of the model are then saved
#This is repeated with another set of hyperparameters indefintiely, and outputs performance of each tested set of hyperparameters to the console
hyperparameters_RSF()

#Run to find hyperparameters for a DeepSurv model
hyperparameters_DS()

#Run to find hyperparameters for an N-MLTR model
hyperparameters_NMLTR()

#Will generate a new set of three models (DeepSurv, N-MLTR, and RSF), trained with the optimal hyperparameters described in our paper, and provide performance characteristics.
#Models are trained on same 80% of data that hyperparameter search was performed in, and tested in remaining 20% of data
trainModels()

#Will use the provided models included in this repository, or previously trained models on a local computer, to generate performance characteristics in the test 
#Will also provide performance characteristics for the EORTC 22931 and RTOG 95-01 trial inclusion criteria for deciding to administer chemotherapy vs radiotherapy
testModels()
```
