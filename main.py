import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lifelines.utils import k_fold_cross_validation
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.metrics import concordance_index_chunk


#------------------------------------------------------------------------
#
#  DATABASE PARSING OPERATIONS
#
#------------------------------------------------------------------------

#Returns dataframe of row s from data with simple imputation
def addRow(s, data, name=None):
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart = data[[s]]
    dpart.loc[:,s] = pd.to_numeric(dpart.loc[:,s], errors='coerce')
    if name is None:
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = [s])
    else:
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns=[name])
    return dpart

#Generates a array with each column being a categorical assessment of involvement of each nodal site
def nodePositiveArray(data, impute=True):
    dpart = pd.concat((
        nodePositive(data, "I", "CS_SITESPECIFIC_FACTOR_3", impute),
        nodePositive(data, "II", "CS_SITESPECIFIC_FACTOR_3", impute),
        nodePositive(data, "III", "CS_SITESPECIFIC_FACTOR_3", impute),
        nodePositive(data, "IV", "CS_SITESPECIFIC_FACTOR_4", impute),
        nodePositive(data, "V", "CS_SITESPECIFIC_FACTOR_4", impute),
        nodePositive(data, "Retropharyngeal", "CS_SITESPECIFIC_FACTOR_4", impute),
        nodePositive(data, "VI", "CS_SITESPECIFIC_FACTOR_5", impute),
        nodePositive(data, "VII", "CS_SITESPECIFIC_FACTOR_5", impute),
        nodePositive(data, "Facial", "CS_SITESPECIFIC_FACTOR_5", impute),
        nodePositive(data, "Parotid", "CS_SITESPECIFIC_FACTOR_6", impute),
        nodePositive(data, "Suboccipital", "CS_SITESPECIFIC_FACTOR_6", impute),
        nodePositive(data, "Parapharyngeal", "CS_SITESPECIFIC_FACTOR_6", impute)
            ), axis = 1)
    return dpart

#Generates an array with 4 columns; RT_Dose is a numerical representation of radiation dose; and 3 categories to indicate RT dose of 50-59, 60-69, and >70 Gy RT
def totalRTDose(data):
    dpart = data[['RAD_REGIONAL_DOSE_CGY', 'RAD_BOOST_DOSE_CGY']]
    dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'] = dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'].replace('Not administered', 0)
    dpart.loc[:,'RAD_BOOST_DOSE_CGY'] = dpart.loc[:,'RAD_BOOST_DOSE_CGY'].replace('Not administered', 0)
    dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'] = pd.to_numeric(dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'], errors='coerce')
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'] = pd.DataFrame(imp.fit_transform(dpart[['RAD_REGIONAL_DOSE_CGY']]), columns = ['RAD_REGIONAL_DOSE_CGY'])
    dpart.loc[:,'RAD_BOOST_DOSE_CGY'] = pd.to_numeric(dpart.loc[:,'RAD_BOOST_DOSE_CGY'], errors='coerce')
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart.loc[:,'RAD_BOOST_DOSE_CGY'] = pd.DataFrame(imp.fit_transform(dpart[['RAD_BOOST_DOSE_CGY']]), columns = ['RAD_BOOST_DOSE_CGY'])
    dpart.loc[:,'RT_Dose'] = dpart.loc[:,'RAD_REGIONAL_DOSE_CGY'] + dpart.loc[:,'RAD_BOOST_DOSE_CGY']
    dpart.loc[:,'50-59 Gy 1'] = (dpart.loc[:,'RT_Dose'] > 4999) & (dpart.loc[:,'RT_Dose'] < 6000)
    dpart.loc[:,'RT_50_59_Gy'] = dpart.loc[:,'50-59 Gy 1'].astype(int)
    dpart.loc[:,'60-69 Gy 1'] = (dpart.loc[:,'RT_Dose'] > 5999) & (dpart.loc[:,'RT_Dose'] < 7000)
    dpart.loc[:,'RT_60_69_Gy'] = dpart.loc[:,'60-69 Gy 1'].astype(int)
    dpart.loc[:,'>70 Gy 1'] = dpart.loc[:,'RT_Dose'] >= 7000
    dpart.loc[:,'RT_70_Gy'] = dpart.loc[:,'>70 Gy 1'].astype(int)
    return dpart[['RT_Dose', 'RT_50_59_Gy', 'RT_60_69_Gy', 'RT_70_Gy']]

#Generates a categorical value, if time from surgery to completion of adjuvant therapy is > 100 days then 1, else 0
def txPackageTime(data, cutoff, impute=True):
    dpart = data[['MTS_TX_PKG_TIME']]
    dpart.loc[:, 'MTS_TX_PKG_TIME'] = pd.to_numeric(dpart.loc[:, 'MTS_TX_PKG_TIME'], errors='coerce')
    if impute:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['MTS_TX_PKG_TIME'])
    if cutoff:
        dpart['MTS_TX_PKG_TIME'] = dpart['MTS_TX_PKG_TIME'].apply(lambda i: 1 if i > 100 else 0)
    return dpart

#Generates numeric value for travel distance
def travelDistance(data):
    dpart = data[['MTS_DISTANCE']]
    dpart = dpart.replace("<10 miles", 5)
    dpart = dpart.replace("10-19 miles", 15)
    dpart = dpart.replace("20-29 miles", 25)
    dpart = dpart.replace("30-50 miles", 40)
    dpart = dpart.replace("50-100 miles", 75)
    dpart = dpart.replace(">100 miles", 150)
    dpart.loc[:, 'MTS_DISTANCE'] = pd.to_numeric(dpart.loc[:,'MTS_DISTANCE'], errors='coerce')
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['Distance'])
    return dpart

#Groups treatment years into 2009-2012 and 2013-2016 categories (if both are 0 then indicates treatment from before 2009)
def yearGroup(data, cutoff):
    dpart = data[['YEAR_OF_DIAGNOSIS']]
    dpart.loc[:, 'YEAR_OF_DIAGNOSIS'] = pd.to_numeric(dpart.loc[:,'YEAR_OF_DIAGNOSIS'], errors='coerce')
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['YEAR_OF_DIAGNOSIS'])
    if cutoff:
        dpart = pd.concat([dpart, dpart], axis = 1)
        dpart.columns = ['Y2009_2012','Y2013_2016']
        dpart.loc[:, ("Y2009_2012")] = dpart.loc[:, ("Y2009_2012")].apply(lambda i: 0 if i < 2009 else (0 if i > 2012 else 1))
        dpart.loc[:, ("Y2013_2016")] = dpart.loc[:, ("Y2013_2016")].apply(lambda i: 0 if i < 2013 else 1)
    return dpart

#Parses nodal involvement for each level of cervical node involvement
def nodePositive(data, s, column, impute=True):
    dpart = data[[column]]
    dpart.loc[:, (column)] = dpart.loc[:, (column)].str.strip()
    import re
    var = np.nan
    if impute:
        var = 0
    if column == "CS_SITESPECIFIC_FACTOR_6":
        sbrief = s[1:]
        dpart.loc[:, (column)] = dpart.loc[:, (column)].apply(
            lambda i: 1 if re.search(sbrief, i) else 0 if re.search("nvolve", i) else var if (re.search("Not documented", i) is not None) | (re.search("Unknown", i) is not  None) | (i == "") else var)
    elif len(s) > 3:
        sbrief = s[1:]
        dpart.loc[:, (column)] = dpart.loc[:, (column)].apply(lambda i: 1 if re.search(sbrief + "[ |,|a].*involved", i) else 0 if re.search("nvolve", i) else var if (re.search("Not documented", i) is not  None) | (re.search("Unknown", i) is not  None) | (i == "") else var)
    else:
        dpart.loc[:, (column)] = dpart.loc[:, (column)].apply(lambda i: 1 if re.search(" " + s + "[ |,|a].*involved", i)  else 0 if re.search("nvolve", i)  else var if (re.search("Not documented", i) is not  None) | (re.search("Unknown", i) is not  None) | (i == "") else var)
    dpart.columns = [s]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    dpart = pd.DataFrame(imp.fit_transform(dpart), columns=dpart.columns)
    return dpart

#Add a categorical value 'name' for column 's' in 'data', where the category is 1 if the value is in 'input', otherwise it is 0
def addDummy(s, input, name, data):
    dpart = data[[s]]
    dpart.loc[:, (s)] = dpart.loc[:, (s)].str.strip()
    dpart.loc[:, (s)] = dpart.loc[:, (s)].apply(lambda i: 1 if i in input else 0)
    dpart.columns = [name]
    #print(dpart)
    return dpart

#Adds a column for HPV status, which is 1 for patients with either high risk or low risk positive HPV, 0 if negative, and empty otherwise (for future imputation)
def addHPV(data):
    input = ['HPV positive, high risk', 'HPV positive, low risk', 'HPV positive, NOS, risk and type(s) not stated']
    dpart = data[['MTS_CS_HIGH_RISK_HPV']]
    dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')] = dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')].str.strip()
    dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')] = dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')].apply(lambda i: 1 if i in input else (0 if i == 'Human papilloma virus (HPV) negative for high-risk and low-risk types' else np.nan))
    dpart.columns = ['HPV']
    #print(dpart)
    return dpart

#Adds a seperate column for HPV status to allow for assessing accuracy of imputation
def addValidationHPV(data):
    input = ['HPV positive, high risk', 'HPV positive, low risk', 'HPV positive, NOS, risk and type(s) not stated']
    dpart = data[['MTS_CS_HIGH_RISK_HPV']]
    dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')] = dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')].str.strip()
    dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')] = dpart.loc[:, ('MTS_CS_HIGH_RISK_HPV')].apply(lambda i: 1 if i in input else (0 if i == 'Human papilloma virus (HPV) negative for high-risk and low-risk types' else -1))
    dpart.columns = ['HPV2']
    #print(dpart)
    return dpart

#Adds a categorical value which is 0 if the input data was blank
def addDummyTx(s, name, data):
    dpart = data[[s]]
    dpart.loc[:, (s)] = dpart.loc[:, (s)].str.strip()
    dpart.loc[:, (s)] = dpart.loc[:, (s)].apply(lambda i: 0 if i=='' else 1)
    dpart.columns = [name]
    return dpart

#Creates a numerical column for Charlson Deyo score
def CDCCNumeric(data):
    dpart = data[['CDCC_TOTAL_BEST']]
    dpart = dpart.replace(">=3", 3)
    dpart['CDCC_TOTAL_BEST'] = pd.to_numeric(dpart['CDCC_TOTAL_BEST'], errors='coerce')
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['CDCC_TOTAL_BEST'])
    return dpart

#Creates a numerical column for depth of invasion
def Depth(data, impute=True):
    dpart = data[['CS_SITESPECIFIC_FACTOR_11']]
    dpart = dpart.replace("<=2mm", 2)
    dpart = dpart.replace("0.21-0.5 mm", 4)
    dpart = dpart.replace("5.1-10mm", 7)
    dpart = dpart.replace("10.1-20 mm", 15)
    dpart = dpart.replace("20.1-40mm", 30)
    dpart = dpart.replace("40.1-97.9 mm", 75)
    dpart = dpart.replace("98.0 mm or larger", 98)
    dpart = dpart.replace("Microinvasion", 1)
    dpart = dpart.replace("Not applicable:  In situ carcinoma", 0)
    dpart = dpart.replace("No tumor found", 0)
    dpart['CS_SITESPECIFIC_FACTOR_11'] = pd.to_numeric(dpart['CS_SITESPECIFIC_FACTOR_11'], errors='coerce')
    if impute:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['CS_SITESPECIFIC_FACTOR_11'])
    return dpart

#Creates dataframe with a numerical or four categorical (1, 2-4, 5-9, and 10+) columns for number of lymph nodes involved (cat = true for categorical)
def NumLNPos(data, cat, impute=True):
    dpart = data[['MTS_LN_POS']]
    dpart = dpart.replace("All nodes examined are negative", 0)
    dpart = dpart.replace("90 or more nodes are positive", 90)
    dpart['MTS_LN_POS'] = pd.to_numeric(dpart['MTS_LN_POS'], errors='coerce')
    if impute:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['MTS_LN_POS'])
    if cat:
        dpart['LN1'] = dpart['MTS_LN_POS'].apply(lambda i: 1 if (i > 0) & (i < 2) else 0)
        dpart['LN2_4'] = dpart['MTS_LN_POS'].apply(lambda i: 1 if (i > 1) & (i < 5) else 0)
        dpart['LN5_9'] = dpart['MTS_LN_POS'].apply(lambda i: 1 if (i > 4) & (i < 10) else 0)
        dpart['LN10'] = dpart['MTS_LN_POS'].apply(lambda i: 1 if i >= 10 else 0)
        return dpart[['LN1', 'LN2_4', 'LN5_9', 'LN10']]
    return dpart

#Creates dataframe with a numerical or two categorical (18+ vs <18 nodes dissected) columns for number of lymph nodes dissected (cat = true for categorical)
def NumLNDis(data, cat, impute = True):
    dpart = data[['MTS_LN_DIS']]
    dpart = dpart.replace("No nodes were examined", 0)
    dpart = dpart.replace("90 or more nodes were examined", 90)
    dpart['MTS_LN_DIS'] = pd.to_numeric(dpart['MTS_LN_DIS'], errors='coerce')
    if impute:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['MTS_LN_DIS'])
    if cat:
        dpart['MTS_LN_DIS_CAT'] = dpart['MTS_LN_DIS'].apply(lambda i: 1 if i > 17 else 0)
        return dpart[['MTS_LN_DIS_CAT']]
    return dpart

#Creates a dataframe with categorical columns representing individual tumor and nodal stage (taking path stage if available, otherwise using clin stage)
def addCategoricalTN(data):
    dpart = data[['TNM_CLIN_T', 'TNM_PATH_T', 'TNM_CLIN_N', 'TNM_PATH_N']]
    dpart.loc[:, 'TNM_CLIN_T'] = dpart.loc[:, 'TNM_CLIN_T'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else 0))))
    dpart.loc[:, 'TNM_PATH_T'] = dpart.loc[:, 'TNM_PATH_T'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else 0))))
    dpart.loc[:, 'TNM_CLIN_N'] = dpart.loc[:, 'TNM_CLIN_N'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else 0))))
    dpart.loc[:, 'TNM_PATH_N'] = dpart.loc[:, 'TNM_PATH_N'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else 0))))
    dpart.loc[:,'T'] = np.maximum(dpart.loc[:,'TNM_PATH_T']*10, dpart.loc[:,'TNM_CLIN_T'])
    dpart.loc[:,'T'] = dpart.loc[:,'T'].apply(lambda i: i if i < 10 else int(i/10))
    dpart.loc[:,'T0'] = dpart.loc[:,'T'].apply(lambda i: 1 if i == 0 else 0)
    dpart.loc[:,'T1'] = dpart.loc[:,'T'].apply(lambda i: 1 if i == 1 else 0)
    dpart.loc[:,'T2'] = dpart.loc[:,'T'].apply(lambda i: 1 if i == 2 else 0)
    dpart.loc[:,'T3'] = dpart.loc[:,'T'].apply(lambda i: 1 if i == 3 else 0)
    dpart.loc[:,'T4'] = dpart.loc[:,'T'].apply(lambda i: 1 if i == 4 else 0)
    dpart.loc[:,'N'] = np.maximum(dpart.loc[:,'TNM_PATH_N'] * 10, dpart.loc[:,'TNM_CLIN_N'])
    dpart.loc[:,'N'] = dpart.loc[:,'N'].apply(lambda i: i if i < 10 else i / 10)
    dpart.loc[:,'N0'] = dpart.loc[:,'N'].apply(lambda i: 1 if i == 0 else 0)
    dpart.loc[:,'N1'] = dpart.loc[:,'N'].apply(lambda i: 1 if i == 1 else 0)
    dpart.loc[:,'N2'] = dpart.loc[:,'N'].apply(lambda i: 1 if i == 2 else 0)
    dpart.loc[:,'N3'] = dpart.loc[:,'N'].apply(lambda i: 1 if i == 3 else 0)
    dpart.loc[:,'T1,2'] = (dpart.loc[:,'T1'] == 1) | (dpart.loc[:,'T2'] == 1)
    dpart.loc[:,'T3,4'] = (dpart.loc[:,'T3'] == 1) | (dpart.loc[:,'T4'] == 1)
    dpart.loc[:,'N0,1'] = (dpart.loc[:,'N1'] == 1) | (dpart.loc[:,'N0'] == 1)
    dpart.loc[:,'N2_3'] = (dpart.loc[:,'N3'] == 1) | (dpart.loc[:,'N2'] == 1)
    return dpart[['T0', 'T1', 'T2', 'T3', 'T4', 'N0', 'N1', 'N2', 'N3', 'T1,2', 'T3,4', 'N0,1', 'N2_3']]

#Numerical dataframe with the highest of clinical and pathologic tumor stages
def addWorstTStage(data):
    dpart = data[['TNM_CLIN_T', 'TNM_PATH_T']]
    dpart['TNM_CLIN_T'] = dpart['TNM_CLIN_T'].apply(
        lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else -1))))
    dpart['TNM_PATH_T'] = dpart['TNM_PATH_T'].apply(
        lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else -1))))
    dpart2 = np.maximum(dpart['TNM_CLIN_T'], dpart['TNM_PATH_T'])
    return pd.DataFrame(dpart2, columns = ['MaxT'], dtype=np.float32)

#Numerical dataframe with the highest of clinical and pathologic nodal stages
def addWorstNStage(data):
    dpart = data[['TNM_CLIN_N', 'TNM_PATH_N']]
    dpart.loc[:, 'TNM_CLIN_N'] = dpart.loc[:, 'TNM_CLIN_N'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else (0 if '0' in i else -1)))))
    dpart.loc[:, 'TNM_PATH_N'] = dpart.loc[:, 'TNM_PATH_N'].apply(lambda i: 4 if '4' in i else (3 if '3' in i else (2 if '2' in i else (1 if '1' in i else (0 if '0' in i else -1)))))
    dpart2 = np.maximum(dpart['TNM_CLIN_N'], dpart['TNM_PATH_N'])
    return pd.DataFrame(dpart2, columns = ['MaxN'], dtype=np.float32)

#Numeric dataframe column representing tumor size in mm
def sizeArray(data, impute=True):
    dpart = data[['TUMOR_SIZE']]
    dpart = dpart.replace("< 1 cm", 5)
    dpart = dpart.replace("> 1 cm, < 2 cm", 15)
    dpart = dpart.replace("> 2 cm, < 3 cm", 25)
    dpart = dpart.replace("> 3 cm, < 4 cm", 35)
    dpart = dpart.replace("> 4 cm, < 5 cm", 45)
    dpart = dpart.replace("> 5 cm, < 6 cm", 55)
    dpart = dpart.replace("> 6 cm, < 7 cm", 65)
    dpart = dpart.replace("> 7 cm, < 8 cm", 75)
    dpart = dpart.replace("> 8 cm, < 9 cm", 85)
    dpart = dpart.replace("989 millimeters or larger", 989)
    dpart = dpart.replace("No mass or tumor found", 0)
    dpart = dpart.replace("Microscopic focus or foci only", 1)
    dpart['TUMOR_SIZE'] = pd.to_numeric(dpart['TUMOR_SIZE'], errors='coerce')
    if impute:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        dpart = pd.DataFrame(imp.fit_transform(dpart), columns = ['TUMOR_SIZE'])
    return dpart

#Dataframe representing if treatment s1 occured more than 30 days after treatment s2
def txSequence(s1, s2, name, data):
    import pandas as pd
    dpart = pd.to_numeric(data[s1].replace(' ',0)) - pd.to_numeric(data[s2].replace(' ',10000))
    dpart = dpart.rename(name)
    dpart = dpart.apply(lambda i: 1 if i > 30 else 0)
    return dpart


#------------------------------------------------------------------------
#
#  MODEL TREATMENT RECOMMENDATIONS FROM RISK
#
#------------------------------------------------------------------------

#For a RSF model, generates the difference in hazard ratio between if X[col] = 0 and if X[col] = 1; generates prediction for each row in X
def treatmentRecForestRisk(model, X, col, printChunk = False):
    x_trt = np.copy(X)
    # Calculate risk of observations treatment i
    x_trt[:, col] = 0
    if len(X) > 500:
        h_i = model.predict_risk_chunk(x_trt, 3, printChunk)
    else:
        h_i = model.predict_risk(x_trt, 3)
    # Risk of observations in treatment j
    x_trt[:, col] = 1
    if len(X) > 500:
        h_j = model.predict_risk_chunk(x_trt, 3, printChunk)
    else:
        h_j = model.predict_risk(x_trt, 3)
    rec_ij = h_i - h_j
    rec_ij = pd.DataFrame(rec_ij, columns=["RecRx"])
    return rec_ij

#For a N-MTLR model, generates the difference in hazard ratio at time 'timeToCompare' between if X[col] = 0 and if X[col] = 1; generates prediction for each row in X
def treatmentRecNN(model, X, col, timeToCompare):
    x_trt = np.copy(X)
    # Calculate risk of observations treatment i
    x_trt[:, col] = 0
    h_i = model.predict(x_trt, timeToCompare)
    # Risk of observations in treatment j
    x_trt[:, col] = 1;
    h_j = model.predict(x_trt, timeToCompare)
    h_i = np.array(h_i)
    h_j = np.array(h_j)
    h_i = h_i[2, :]
    h_j = h_j[2, :]
    rec_ij = h_j - h_i
    rec_ij = pd.DataFrame(rec_ij, columns=["RecRx"])
    return rec_ij

#For a DeepSurv or N-MLTR model, generates the difference in hazard ratio  between if X[col] = 0 and if X[col] = 1; generates prediction for each row in X
def treatmentRecNNRisk(model, X, col):
    x_trt = np.copy(X)
    # Calculate risk of observations treatment i
    x_trt[:, col] = 0
    h_i = model.predict_risk(x_trt)
    # Risk of observations in treatment j
    x_trt[:, col] = 1;
    h_j = model.predict_risk(x_trt)
    rec_ij = h_i - h_j
    rec_ij = pd.DataFrame(rec_ij, columns=["RecRx"])
    return rec_ij


#------------------------------------------------------------------------
#
#  PLOTTING FUNCTIONS
#
#------------------------------------------------------------------------


#Generates a heatmap of feature importance
#feature_filename - location of csv file with feature importance

def plotFeatureImportance(feature_filename):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    mpl.rcParams['figure.dpi'] = 300
    a = pd.read_csv(feature_filename)
    a = a.sort_values(by=['Average'], ascending=False)
    fig, ax = plt.subplots(figsize=(8, 10))

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    im = ax.imshow(a[['DeepSurv', 'N-MTLR', 'RSF', 'Average']],
                   cmap=truncate_colormap(cm.get_cmap("hot"), minval=0.4, maxval=1), norm=colors.SymLogNorm(1e-3))
    # im.set_clim(-5, -2)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Feature Permutation Importance (% Decrease in Concordance Index)", rotation=-90, va="bottom")
    cbar.ax.set_aspect(50)
    cbar.set_ticks([0.01, 0.001, 0, -0.001])
    cbar.set_ticklabels(["1", "0.1", "0", "-0.1"])

    ax.set_yticks(np.arange(len(a)))
    ax.set_xticks(np.arange(len(a.columns) - 1))
    ax.set_xticklabels(['DeepSurv', 'N-MTLR', 'RSF', 'Average'])
    ax.set_yticklabels(a["Index"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig("variable.eps")
    plt.show()

#Plots an array of survival functions; one for patients recommended for CRT or RT, and one for patients in total
#rec_ij - an array of treatment recommendations for each model
#X - model features for each patient
#T - time to event for each patient
#D - death status for each patient
#name - an array of names for each plot
#ratio - can be used to set the cutoff for chemo vs RT

def plotSurvivalArray(rec_ij, X, T, D, name, ratio = 0):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(len(rec_ij), 2, figsize=(20, len(rec_ij)*5))
    i = 0

    #plt.set_size_inches(20, len(rec_ij)*5)
    for rec in rec_ij:
        fullSet = pd.concat([rec, X, T, D], axis = 1)
        if(ratio > 0):
            cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
        else:
            cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
        ratio = 0
        cutoff = cutoff[0] * ratio
        from lifelines import KaplanMeierFitter
        fullSetNN = fullSet[(fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)]
        fullSetNC = fullSet[(fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)]
        fullSetCN = fullSet[(fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0)]
        fullSetCC = fullSet[(fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1)]
        kmf2 = KaplanMeierFitter()
        kmf = KaplanMeierFitter()

        kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Received CRT')
        kmf2.survival_function_.plot(ax=axs[i, 0], color='C1')
        kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received RT')
        kmf.survival_function_.plot(ax=axs[i, 0], color = 'C0')
        axs[i, 0].set_xlim([0, 100])
        axs[i, 0].set_ylim([0.2, 1])
        axs[i, 0].set_title("Subgroup Recommended for RT - " + name[i])
        from lifelines.statistics import logrank_test
        results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
        from lifelines.utils import median_survival_times
        if results.p_value < 0.001:
            pval = "< 0.001"
        else:
            pval = "= " + str(round(results.p_value, 3))
        axs[i, 0].text(2, 0.35, "Log Rank Test p value " + pval)
        mst = str(median_survival_times(kmf.confidence_interval_)).split("              ")
        print(str(median_survival_times(kmf.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i, 0].text(2, 0.31, "MST (months), Received RT: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        mst = str(median_survival_times(kmf2.confidence_interval_)).split("              ")
        print(str(median_survival_times(kmf2.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i, 0].text(2, 0.27, "MST (months), Received CRT: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        axs[i, 0].set(xlabel='Time (months)', ylabel='Overall Survival')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf2, kmf, ax=axs[i, 0], labels =['CRT', 'RT'], special=True)

        kmf2.fit(fullSetCC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetCC[['Dead']], label='Received CRT')
        kmf2.survival_function_.plot(ax=axs[i, 1], color='C1')
        kmf.fit(fullSetCN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetCN[['Dead']], label='Received RT')
        kmf.survival_function_.plot(ax=axs[i, 1], color='C0')
        axs[i, 1].set_xlim([0, 100])
        axs[i, 1].set_ylim([0.2, 1])
        axs[i, 1].set_title("Subgroup Recommended for CRT - " + name[i])
        from lifelines.statistics import logrank_test
        results = logrank_test(fullSetCN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetCC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetCN[['Dead']], fullSetCC[['Dead']])
        if results.p_value < 0.001:
            pval = "< 0.001"
        else:
            pval = "= " + str(round(results.p_value, 3))
        axs[i, 1].text(2, 0.35, "Log Rank Test p value " + pval)
        mst = str(median_survival_times(kmf.confidence_interval_)).split("              ")
        print(str(median_survival_times(kmf.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i, 1].text(2, 0.31, "MST (months), Received RT: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        mst = str(median_survival_times(kmf2.confidence_interval_)).split("              ")
        print(str(median_survival_times(kmf2.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i, 1].text(2, 0.27, "MST (months), Received CRT: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        axs[i, 1].set(xlabel='Time (months)', ylabel='Overall Survival')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf2, kmf, ax=axs[i, 1], labels=['CRT','RT'], special=True)
        i = i + 1
    labi = 0
    labels = ['    A',  '    B', '    C', '    D', '    E', '    F']
    for ax in axs.flat:
        ax.tick_params(which='both', labelbottom=True, labelleft=True)
        ax.get_xaxis().get_label().set_visible(True)
        ax.get_yaxis().get_label().set_visible(True)
        ax.text(-0.075, 1.05, labels[labi], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        labi = labi + 1
    #plt.tight_layout(pad =2)
    plt.tight_layout()
    try:
        plt.savefig('RTvsCRT.eps', format='eps', dpi=fig.dpi)
    except:
        print("Error!")
    plt.show()




    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(len(rec_ij), 1, figsize=(10, len(rec_ij)*5))
    i = 0
    for rec in rec_ij:
        fullSet = pd.concat([rec, X, T, D], axis = 1)
        if(ratio > 0):
            cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
        else:
            cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
        ratio = 0
        cutoff = cutoff[0] * ratio
        from lifelines import KaplanMeierFitter
        fullSetNN = fullSet[
            ((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
        fullSetNC = fullSet[
            ((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]

        kmf2 = KaplanMeierFitter()
        kmf = KaplanMeierFitter()

        kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Received Recommended Treatment')
        kmf2.survival_function_.plot(ax=axs[i], color='C1')
        kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Did Not Receive Recommended Treatment')
        kmf.survival_function_.plot(ax=axs[i], color = 'C0')
        axs[i].set_xlim([0, 100])
        axs[i].set_ylim([0.2, 1])
        axs[i].set_title("Treatment Recommendations - " + name[i])
        from lifelines.statistics import logrank_test
        results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
        from lifelines.utils import median_survival_times
        if results.p_value < 0.001:
            pval = "< 0.001"
        else:
            pval = "= " + str(round(results.p_value, 3))
        axs[i].text(2, 0.35, "Log Rank Test p value " + pval)
        mst = str(median_survival_times(kmf.confidence_interval_)).split("                                  ")
        print(str(median_survival_times(kmf.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i].text(2, 0.31, "MST (months), Received Recommended Treatment: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        mst = str(median_survival_times(kmf2.confidence_interval_)).split("                                  ")
        print(str(median_survival_times(kmf2.confidence_interval_)))
        ub = mst[1].strip()
        lb = mst[2].strip()
        if ub == "inf":
            ub = "not reached"
        if lb == "inf":
            lb = "not reached"
        axs[i].text(2, 0.27, "MST (months), Did Not Receive Recommended Treatment: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
        axs[i].set(xlabel='Time (months)', ylabel='Overall Survival')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf2, kmf, ax=axs[i], labels =['Rec', 'Anti-Rec'], special=True)

        i = i + 1
    labi = 0
    labels = ['A',  'B', 'C', 'D', 'E', 'F']
    for ax in axs.flat:
        ax.tick_params(which='both', labelbottom=True, labelleft=True)
        ax.get_xaxis().get_label().set_visible(True)
        ax.get_yaxis().get_label().set_visible(True)
        ax.text(-0.1, 1.05, labels[labi], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        labi = labi + 1

    plt.tight_layout()
    try:
        plt.savefig('TreatmentRecs.eps', format='eps', dpi=fig.dpi)
    except:
        print("Error!")
    plt.show()

    return

#Generate sets of plots for a single model including - received CRT vs RT but recommended RT; received CRT vs RT but recommended CRT
#rec_ij - treatment recommendation for a single model
#X - model features for each patient
#T - time to event for each patient
#D - death status for each patient
#name - model name
#ratio - can be used to set the cutoff for chemo vs RT

def plotSurvival(rec_ij, X, T, D, name, ratio):
    fullSet = pd.concat([rec_ij, X, T, D], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    print(fullSet.describe().to_string())
    from lifelines import KaplanMeierFitter
    fullSetNN = fullSet[(fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)]
    fullSetNC = fullSet[(fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)]
    fullSetCN = fullSet[(fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0)]
    fullSetCC = fullSet[(fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1)]

    kmf2 = KaplanMeierFitter()
    kmf = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Received CRT')
    kmf2.survival_function_.plot(ax=ax, color='C1')
    kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received RT')
    kmf.survival_function_.plot(ax=ax, color = 'C0')
    ax.set_xlim([0, 100])
    plt.title("Subgroup Recommended for RT - " + name)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    from lifelines.utils import median_survival_times
    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    lrrt = results.p_value
    plt.figtext(0.12, 0.35, "Log Rank Test p value " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("              ")
    #print(str(median_survival_times(kmf.confidence_interval_)))
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.12, 0.31, "MST (months), Received RT: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("              ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.12, 0.27, "MST (months), Received CRT: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels =['CRT', 'RT'])
    plt.show()

    ax = plt.subplot(111)
    kmf2.fit(fullSetCC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetCC[['Dead']], label='Received CRT')
    kmf2.survival_function_.plot(ax=ax, color='C1')
    kmf.fit(fullSetCN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetCN[['Dead']], label='Received RT')
    kmf.survival_function_.plot(ax=ax, color='C0')
    ax.set_xlim([0, 100])
    plt.title("Subgroup Recommended for CRT - " + name)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetCN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetCC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetCN[['Dead']], fullSetCC[['Dead']])
    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    lrcrt = results.p_value
    plt.figtext(0.12, 0.35, "Log Rank Test p value " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("              ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.12, 0.31, "MST (months), Received RT: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("              ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.12, 0.27, "MST (months), Received CRT: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels=['CRT','RT'])
    plt.show()
    return rec_ij, lrrt, lrcrt

#Generate plot for single model- received recommended treatment vs received other treatment
#rec_ij - treatment recommendation for a single model
#X - model features for each patient
#T - time to event for each patient
#D - death status for each patient
#name - model name
#ratio - can be used to set the cutoff for chemo vs RT

def plotSurvivalConcordant(rec_ij, X, T, D, name, ratio):
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    D.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, D], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    #cutoffLB = np.percentile(fullSet[fullSet.RecRx < 0][['RecRx']], 100 - ratio)
    #cutoffUB = np.percentile(fullSet[fullSet.RecRx > 0][['RecRx']], ratio)
    #if(ratio == 0):
    #   cutoffLB = 0
    #    cutoffUB = 0
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]
    #fourPlots(fullSet)

    #print(CPHStat(fullSetNN, fullSetNC, False, True))
    kmf = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received Recommended Treatment')
    kmf.survival_function_.plot(ax=ax, color='C1')
    kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Did Not Receive Recommended Treatment')
    kmf2.survival_function_.plot(ax=ax, color='C0')
    ax.set_xlim([0, 100])
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    from lifelines.utils import median_survival_times

    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    plt.figtext(0.15, 0.35, "Log Rank Test p value  " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.31, "MST (months), Received Recommended Treatment: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.27, "MST (months), Did Not Receive Recommended Treatment: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")

    plt.title("Treatment recommendations - " + name)
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels=['Rec', 'Anti-Rec'])
    plt.show()
    #print("HR CPH Stat: " + name)
    #print(CPHStat(fullSetNN, fullSetNC))
    #print("Median Survivals")
    #print(kmf.median_survival_time_)
    from lifelines.utils import median_survival_times
    #print(median_survival_times(kmf.confidence_interval_))
    #print(kmf2.median_survival_time_)
    #print(median_survival_times(kmf2.confidence_interval_))
    #print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
#    print("Log Rank Test Results")
#    print(results.print_summary())

    #print("RMST difference: " + str(rmst1 - rmst2))
    return CPHStat(fullSetNN, fullSetNC)



#Returns two subsets of data; those who received chemo and those who did not
def treatmentGroupsChemoAll(X,T,D):
    fullSet = pd.concat([X, T, D], axis=1)
    fullSet["RecRx"] = 1#(fullSet["N0"] == 0 ) | (fullSet["T3"] == 1 ) | (fullSet["T4"] == 1 )
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    return fullSetNN, fullSetNC

#Returns patients in two groups - one group is patients who received CRT if they had positive margins / ECE and RT otherwise; other group is remainder of patients (received RT for +margin/ECE or CRT with neg margin and neg ECE)
def treatmentGroupsECE(X,T,D):
    fullSet = pd.concat([X, T, D], axis=1)
    fullSet["RecRx"] = (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (fullSet["Micro_Margins"] == 1) | (
                fullSet["Gross_Margins"] == 1)
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    return fullSetNN, fullSetNC

#Returns two groups of patients - those who received CRT if they met EORTC 22931 inclusion criteria and RT if they didnt; and other group is all other patients (received RT only despite meeting EORTC 22931 inclusion criteria, and CRT despite not meeting EORTC inclusion criteria)
def treatmentGroupsEORTC(X,T,D):
    fullSet = pd.concat([X, T, D], axis=1)
    fullSet["RecRx"] = (fullSet["T4"] == 1) | (
                (fullSet["T3"] == 1) & ((fullSet["N0"] != 1) | (fullSet["Larynx"] != 1))) | (fullSet["N2_3"] == 1) | (
                                   (fullSet["OC_OPX"] == 1) & ((fullSet["IV"] == 1) | (fullSet["V"] == 1))) | (
                                   fullSet["LVI"] == 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (
                                   fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    return fullSetNN, fullSetNC

#Returns two groups of patients - those who received CRT if they met RTOG 95-01 inclusion criteria and RT if they didnt; and other group is all other patients (received RT only despite meeting RTOG 95-01 inclusion criteria, and CRT despite not meeting RTOG inclusion criteria)
def treatmentGroupsRTOG(X,T,D):
    fullSet = pd.concat([X, T, D], axis=1)
    fullSet["RecRx"] = (fullSet["MTS_LN_POS"] > 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (
                fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    return fullSetNN, fullSetNC

#Plots survival of two groups of patients - those who received CRT if they met EORTC 22931 inclusion criteria and RT if they didnt; and other group is all other patients (received RT only despite meeting EORTC 22931 inclusion criteria, and CRT despite not meeting EORTC inclusion criteria)
def plotSurvivalEORTC(X, T, D):
    fullSet = pd.concat([X, T, D], axis = 1)
    fullSet["RecRx"] =  (fullSet["T4"] == 1) | ((fullSet["T3"] == 1) & ((fullSet["N0"] != 1) | (fullSet["Larynx"] != 1))) | (fullSet["N2_3"] == 1) | ((fullSet["OC_OPX"] == 1) & ((fullSet["IV"] == 1) | (fullSet["V"] == 1))) |  (fullSet["LVI"] == 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    from lifelines import KaplanMeierFitter
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    #print(fullSet[fullSet.RecRx == 0].describe().to_string())
    #print(fullSet[fullSet.RecRx == 1].describe().to_string())
    kmf = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received Recommended Treatment')
    kmf.survival_function_.plot(ax=ax, color='C1')
    kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Did Not Receive Recommended Treatment')
    kmf2.survival_function_.plot(ax=ax, color='C0')
    ax.set_xlim([0, 100])
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    from lifelines.utils import median_survival_times

    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    plt.figtext(0.15, 0.35, "Log Rank Test p value  " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.31, "MST (months), Received Recommended Treatment: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.27, "MST (months), Did Not Receive Recommended Treatment: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")
    plt.title("Treatment recommendations - EORTC")
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels =['Rec', 'Anti-Rec'])
    plt.show()
    from lifelines.utils import median_survival_times
    #print(kmf.median_survival_time_)
    #print(median_survival_times(kmf.confidence_interval_))
    #print(kmf2.median_survival_time_)
    #print(median_survival_times(kmf2.confidence_interval_))
    #print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    #print(results.print_summary())
    CPHStat(fullSetNN, fullSetNC)

#Plots survival of two groups of patients - those who received CRT if they met RTOG 95-01 inclusion criteria and RT if they didnt; and other group is all other patients (received RT only despite meeting RTOG 95-01 inclusion criteria, and CRT despite not meeting RTOG inclusion criteria)
def plotSurvivalRTOG(X, T, D):
    fullSet = pd.concat([X, T, D], axis = 1)
    fullSet["RecRx"] = (fullSet["MTS_LN_POS"] > 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    from lifelines import KaplanMeierFitter
    fullSetNN = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 0)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx == 0) & (fullSet.Chemo == 1)) | ((fullSet.RecRx == 1) & (fullSet.Chemo == 0))]
    #print(fullSet[fullSet.RecRx == 0].describe().to_string())
    #print(fullSet[fullSet.RecRx == 1].describe().to_string())
    kmf = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received Recommended Treatment')
    kmf.survival_function_.plot(ax=ax, color='C1')
    kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Did Not Receive Recommended Treatment')
    kmf2.survival_function_.plot(ax=ax, color='C0')

    ax.set_xlim([0, 100])
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    from lifelines.utils import median_survival_times

    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    plt.figtext(0.15, 0.35, "Log Rank Test p value  " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.31, "MST (months), Received Recommended Treatment: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.27, "MST (months), Did Not Receive Recommended Treatment: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")
    plt.title("Treatment recommendations - RTOG")
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels =['Rec', 'Anti-Rec'])
    plt.show()
    from lifelines.utils import median_survival_times
    #print(kmf.median_survival_time_)
    #print(median_survival_times(kmf.confidence_interval_))
    #print(kmf2.median_survival_time_)
    #print(median_survival_times(kmf2.confidence_interval_))
    #print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    #print(results.print_summary())
    CPHStat(fullSetNN, fullSetNC)

def plotSurvivalChemo(X, T, D):
    fullSet = pd.concat([X, T, D], axis = 1)
    fullSet["RecRx"] = 1
    from lifelines import KaplanMeierFitter
    fullSetNN = fullSet[(fullSet.Chemo == 1)]
    fullSetNC = fullSet[(fullSet.Chemo == 0)]
    #print(fullSet[fullSet.RecRx == 0].describe().to_string())
    #print(fullSet[fullSet.RecRx == 1].describe().to_string())
    kmf = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Received Recommended Treatment')
    kmf.survival_function_.plot(ax=ax, color='C1')
    kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Did Not Receive Recommended Treatment')
    kmf2.survival_function_.plot(ax=ax, color='C0')

    ax.set_xlim([0, 100])
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    from lifelines.utils import median_survival_times

    if results.p_value < 0.001:
        pval = "< 0.001"
    else:
        pval = "= " + str(round(results.p_value, 3))
    plt.figtext(0.15, 0.35, "Log Rank Test p value  " + pval)
    mst = str(median_survival_times(kmf.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.31, "MST (months), Received ChemoRT: " + str(kmf.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    mst = str(median_survival_times(kmf2.confidence_interval_)).split("                                  ")
    ub = mst[1].strip()
    lb = mst[2].strip()
    plt.figtext(0.15, 0.27, "MST (months), Received RT: " + str(kmf2.median_survival_time_) + " 95% CI: (" + lb + ", " + ub + ")")
    plt.xlabel("Time (months)")
    plt.ylabel("Overall Survival")
    plt.title("Treatment recommendations - Chemo for all")
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf2, kmf, ax=ax, labels =['Chemo', 'RT'])
    plt.show()
    from lifelines.utils import median_survival_times
    #print(kmf.median_survival_time_)
    #print(median_survival_times(kmf.confidence_interval_))
    #print(kmf2.median_survival_time_)
    #print(median_survival_times(kmf2.confidence_interval_))
    #print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    from lifelines.statistics import logrank_test
    results = logrank_test(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], fullSetNN[['Dead']], fullSetNC[['Dead']])
    #print(results.print_summary())
    CPHStat(fullSetNN, fullSetNC)

def modelPlots(model, X, T, E, name, ratio = 0, rec_ij = None):
    a, b, c = plotSurvival(rec_ij, X, T, E, name, ratio)
    return plotSurvivalConcordant(rec_ij, X, T, E, name, ratio), b, c

import shap
def plotShap(model = None, vd = None, modelType = "NNCPH"):
    vd = shap.sample(vd['x'], 100)
    def modelWrapper(X):
        if modelType == "RSF":
            rec_ij = treatmentRecForestRisk(model, X, 0)
        else:
            rec_ij = treatmentRecNNRisk(model, X, 0)
        return rec_ij

    explainer = shap.KernelExplainer(modelWrapper, vd)
    shap_values = explainer.shap_values(vd)
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, vd, max_display=45, show = False)
    plt.savefig("summary_plot"+modelType+".eps", format='eps', dpi=300)
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, vd, plot_type="bar", max_display=45, show = False)
    plt.savefig("summary_plot_bar"+modelType+".eps", format='eps', dpi=300)
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, vd, plot_type="violin", max_display=45, show = False)
    plt.savefig("summary_plot_violin"+modelType+".eps", format='eps', dpi=300)
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------
#
#  STATISTICS FUNCTIONS
#
#------------------------------------------------------------------------


#Returns the CPH HR between groups D1 and D2. PM = True for propensity weighting using 'scores' column; PS = True for also printing summary in console; MS = True for printing median survival times
def CPHStat(D1, D2, PM = False, PS = False, MS = False):
    D1["Group"] = 1
    D2["Group"] = 0
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    if PM:
        DFR = pd.concat([D1[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group', 'scores']],
                         D2[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group', 'scores']]])
        cph.fit(DFR, 'DX_LASTCONTACT_DEATH_MONTHS', event_col='Dead', weights_col='scores', robust=True)
    else:
        DFR = pd.concat([D1[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group']],
                         D2[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group']]])
        cph.fit(DFR,  'DX_LASTCONTACT_DEATH_MONTHS', event_col='Dead')
    if PS:
        cph.print_summary()
    if MS:
        kmf = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf.fit(D1[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=D1[['Dead']],
                label='Recommended Tx = Received Tx')
        kmf2.fit(D2[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=D2[['Dead']],
                 label='Recommended Tx != Received Tx')
        print("Median Survivals")
        print(kmf.median_survival_time_)
        from lifelines.utils import median_survival_times
        print(median_survival_times(kmf.confidence_interval_))
        print(kmf2.median_survival_time_)
        print(median_survival_times(kmf2.confidence_interval_))
        print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    #print(np.exp(cph.params_['Group']))
    return np.exp(cph.params_['Group'])

#Returns propensity matched CPH HR
def CPHStatPM(D1, D2):
    D1["Group"] = 1
    D2["Group"] = 0
    DFR = pd.concat([D1[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group', 'scores']], D2[['DX_LASTCONTACT_DEATH_MONTHS', 'Dead', 'Group', 'scores']]])
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(DFR,  'DX_LASTCONTACT_DEATH_MONTHS', event_col='Dead', weights_col='scores')
    #print(np.exp(cph.params_['Group']))
    return np.exp(cph.params_['Group'])


#For an RSF model: Finds the difference in HRs that would be associated with the biggest difference between groups receiving recommended treatment and groups not receiving recommended treatment
def bestRatioForest(model, X, T, E, rec_ij):
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    import scipy.optimize as opt
    minres = opt.brute(lambda x: optimizeFunction(x, fullSet), ((-1,1),), Ns=100, full_output=True, finish = opt.fmin)
    #print(minres)
    return minres[0][0]

#For a NN model: Finds the difference in HRs that would be associated with the biggest difference between groups receiving recommended treatment and groups not receiving recommended treatment
def bestRatio(model, X, T, E):
    rec_ij = treatmentRecNNRisk(model, X, 0)
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    import scipy.optimize as opt
    minres = opt.brute(lambda x: optimizeFunction(x, fullSet), ((-1,1),), Ns=100, full_output=True, finish = opt.fmin)
    #print(minres)
    return minres[0][0]

#Returns the CPH stat comparing two groups, those receiving the recommended treatment and those who did not, where an Rx is recommended only if the HR benefit is greater than 'ratio'
def optimizeFunction(ratio, fullSet):
    ratio = ratio[0]
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0

    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]
    return CPHStat(fullSetNN, fullSetNC)

#Print out summary statistics about model performance and patients recommended to receive one treatment vs another for a NN model
def treatmentDifferencesNN(model, data, columns, testVar, ratio = 0):
    data = data.reset_index(drop = True)
    rec_ij = treatmentRecNNRisk(model, data[columns], 0)
    return treatmentDifferences(model, data, columns, rec_ij, testVar, ratio)

#Print out summary statistics about model performance and patients recommended to receive one treatment vs another for an RSF model
def treatmentDifferencesSF(model, data, columns):
    print("Getting rec_ij")
    data = data[(data["ECE_micro"] == 0) & (data["ECE_macro"] == 0) & (data["Micro_Margins"] == 0) & (data["Gross_Margins"] == 0)]
    data = data.reset_index(drop=True)
    rec_ij = treatmentRecForestRisk(model, data[columns], 0, True)
    return treatmentDifferences(model, data, columns, rec_ij)

#Print out summary statistics about model performance and patients recommended to receive one treatment vs another
def treatmentDifferences(model, data, columns, rec_ij, testVar = "Chemo", ratio = 0):
    fullSet = pd.concat([rec_ij, data], axis = 1)
    #fullSet = fullSet[(fullSet["ECE_micro"] == 0) & (fullSet["ECE_macro"] == 0) & (fullSet["Micro_Margins"] == 0) & (fullSet["Gross_Margins"] == 0)]
    fullSet["EORTC"] = (fullSet["T4"] == 1) | (
                (fullSet["T3"] == 1) & ((fullSet["N0"] != 1) | (fullSet["Larynx"] != 1))) | (fullSet["N2_3"] == 1) | (
                                   (fullSet["OC_OPX"] == 1) & ((fullSet["IV"] == 1) | (fullSet["V"] == 1))) | (
                                   fullSet["LVI"] == 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (
                                   fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    fullSet["EORTC"] = fullSet["EORTC"].astype(int)
    fullSet["RTOG"] = (fullSet["MTS_LN_POS"] > 1) | (fullSet["ECE_micro"] == 1) | (fullSet["ECE_macro"] == 1) | (
                fullSet["Micro_Margins"] == 1) | (fullSet["Gross_Margins"] == 1)
    fullSet["RTOG"] = fullSet["RTOG"].astype(int)
    print(fullSet.describe().to_string())
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    print("Recommended for RT alone")
    print((fullSet[fullSet.RecRx < cutoff]).reset_index(drop=True).describe().to_string())
    print("Recommended for CRT")
    a = len(fullSet[fullSet.RecRx >= cutoff].reset_index(drop=True))
    return a

#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a NN model
def hazardRatioNN(model, X, T, E, ratio = 0, PS = False, MS = False, testVar = "Chemo"):
    X = X.reset_index(drop=True)
    T = T.reset_index(drop=True)
    E = E.reset_index(drop=True)
    rec_ij = treatmentRecNNRisk(model, X, 0)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    print(fullSet[fullSet.RecRx < 0].shape)
    print(fullSet[fullSet.RecRx >= 0].shape)
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 0))]
    if MS:
        kmf = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']], label='Recommended Tx = Received Tx')
        kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']], label='Recommended Tx != Received Tx')
        print("Median Survivals")
        print(kmf.median_survival_time_)
        from lifelines.utils import median_survival_times
        print(median_survival_times(kmf.confidence_interval_))
        print(kmf2.median_survival_time_)
        print(median_survival_times(kmf2.confidence_interval_))
        print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    return CPHStat(fullSetNN, fullSetNC, False, PS)

#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a NN model, with propensity matching
def hazardRatioPMNN(model, X, T, E, trainCols, ratio = 0, PS = False, testVar = "Chemo"):
    X = X.reset_index(drop=True)
    T = T.reset_index(drop=True)
    E = E.reset_index(drop=True)
    rec_ij = treatmentRecNNRisk(model, X[trainCols], 0)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 0))]
    return CPHStat(fullSetNN, fullSetNC, True, PS)

def hazardRatio(model, X, T, E, rec_ij, ratio = 0, testVar = "Chemo", propMatch = False, printSummary = False, printSurvival = False):
    X = X.reset_index(drop=True)
    T = T.reset_index(drop=True)
    E = E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet[testVar] == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet[testVar] == 0))]
    return CPHStat(fullSetNN, fullSetNC, PM = propMatch, PS = printSummary, MS = printSurvival)
#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a RSF model, with propensity matching [requires pre-computing the recommended Rx which is computational expensive]

def hazardRatioPMSF(model, X, T, E, rec_ij, ratio = 0):
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]
    return CPHStatPM(fullSetNN, fullSetNC)

#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a RSF model [does not require precomputing the recommended Rx]
def hazardRatioPMSFRIJ(model, X, T, E, trainCols, ratio = 0, PS = False):
    rec_ij = treatmentRecForestRisk(model, X[trainCols], 0, True)
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]
    return CPHStat(fullSetNN, fullSetNC, True, PS)

#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a RSF model, and prints median survivals [does not require precomputing the recommended Rx]
def hazardRatioSFRIJ(model, X, T, E, ratio, PS = False, MS = False):
    rec_ij = treatmentRecForestRisk(model, X, 0, True)
    return hazardRatioSF(model, X, T, E, ratio, rec_ij, PS, MS)

#Return the CPH HR comparing patients receiving recommended Rx vs alternative Rx for a RSF model [requires pre-computing the recommended Rx which is computational expensive]
def hazardRatioSF(model, X, T, E, ratio, rec_ij, PS = False, MS = True):
    X.reset_index(drop=True)
    T.reset_index(drop=True)
    E.reset_index(drop=True)
    rec_ij.reset_index(drop=True)
    fullSet = pd.concat([rec_ij, X, T, E], axis = 1)
    if(ratio > 0):
        cutoff = np.mean(fullSet[fullSet.RecRx < 0][['RecRx']])
    else:
        cutoff = -1*np.mean(fullSet[fullSet.RecRx > 0][['RecRx']])
    cutoff = cutoff[0] * ratio
    if np.isnan(cutoff):
        cutoff = 0
    print(fullSet[fullSet.RecRx < cutoff].shape)
    print(fullSet[fullSet.RecRx >= cutoff].shape)
    fullSetNN = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 0)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 1))]
    fullSetNC = fullSet[((fullSet.RecRx < cutoff) & (fullSet.Chemo == 1)) | ((fullSet.RecRx >= cutoff) & (fullSet.Chemo == 0))]

    if MS:
        kmf = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf.fit(fullSetNN[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNN[['Dead']],
                label='Recommended Tx = Received Tx')
        kmf2.fit(fullSetNC[['DX_LASTCONTACT_DEATH_MONTHS']], event_observed=fullSetNC[['Dead']],
                 label='Recommended Tx != Received Tx')
        print("Median Survivals")
        print(kmf.median_survival_time_)
        from lifelines.utils import median_survival_times
        print(median_survival_times(kmf.confidence_interval_))
        print(kmf2.median_survival_time_)
        print(median_survival_times(kmf2.confidence_interval_))
        print(kmf.median_survival_time_ - kmf2.median_survival_time_)
    return CPHStat(fullSetNN, fullSetNC, False, PS)

#Generate inverse probability of treatment weighting scores between two groups of patients; adds a 'scores' column to the dataframe
def scoresTwoGroup(G0, G1, trainCols, excludeCols, testVar = "Chemo"):
    G1["Gr"] = 1
    G0["Gr"] = 0
    total_data = pd.concat([G1, G0])
    total_data = G1
    sublist = trainCols.copy()
    sublist.append("Dead")
    sublist.append("DX_LASTCONTACT_DEATH_MONTHS")
    sublist2 = sublist.copy()
    sublist.append("Gr")
    sublist2.append("scores")
    total_data = total_data[sublist]
    test = total_data[total_data[testVar] == 1]
    control = total_data[total_data[testVar] == 0]
    from pymatch.Matcher import Matcher
    excludeA = ["DX_LASTCONTACT_DEATH_MONTHS", "Dead", "Gr"] + excludeCols
    print(test.columns == control.columns)
    print(testVar)
    m = Matcher(test, control, yvar=testVar, exclude=excludeA)

    np.random.seed(20170925)
    m.fit_scores(balance=True, nmodels=10)
    m.predict_scores()
    m.data["scores"] = m.data[testVar] / m.data["scores"] + (1 - m.data[testVar]) / (1 - m.data["scores"])
    total_data = m.data
    total_data.reset_index(drop=True)
    G1 = total_data[total_data.Gr == 1]
    G1 = G1[sublist2]
    return G1


#Generate hazard ratios for a standard model with inverse probability of treatment weighting
def hazardGroupStandard(dataS, trainCols, RS):
    excludeBase = ["MTS_LN_POS", "N0", "N1", "N2_3", "T1", "T2", "OC_OPX"]
    excludeE = ["Micro_Margins", "Gross_Margins", "ECE_micro", "ECE_macro"]

    dataS = dataS[(dataS["ECE_micro"] == 0) & (dataS["ECE_macro"] == 0) & (dataS["Micro_Margins"] == 0) & (dataS["Gross_Margins"] == 0)]

    sublist = trainCols.copy()
    sublist.append("Dead")
    sublist.append("DX_LASTCONTACT_DEATH_MONTHS")

    train_dataSet, test_data, y_train, y_test = train_test_split(dataS, dataS, test_size=0.2, random_state=RS)
    test_data1 = test_data# test_data[(test_data["ECE_micro"] == 0) & (test_data["ECE_macro"] == 0) & (test_data["Micro_Margins"] == 0) & (test_data["Gross_Margins"] == 0)]
    test_data1 = test_data1.reset_index(drop=True)
    train_dataSet1 = train_dataSet
    #train_dataSet1 = train_dataSet[(train_dataSet["ECE_micro"] == 0) & (train_dataSet["ECE_macro"] == 0) & (train_dataSet["Micro_Margins"] == 0) & (train_dataSet["Gross_Margins"] == 0)]
    test_data1 = scoresTwoGroup(train_dataSet1, test_data1, trainCols, excludeBase + excludeE)
    hazardStandard(test_data1, trainCols, excludeBase + excludeE)

#Generate hazard ratios for treatment according to standard decision rules (Chemo only for +Margins/ECE; chemo only for patients meeting RTOG 9501 or EORTC 22931 inclusion criteria)
def hazardStandard(test_data1, trainCols, excludeCols):
    print(test_data1.describe().to_string())
    print(test_data1[test_data1.Chemo == 0].describe().to_string())
    print(test_data1[test_data1.Chemo == 1].describe().to_string())

    vdW1 = {
        'x': test_data1.loc[:, trainCols],
        'e': np.squeeze(test_data1[["Dead"]]),
        't': np.squeeze(test_data1[["DX_LASTCONTACT_DEATH_MONTHS"]])
    }
    sublist2 = trainCols.copy()
    sublist2.append("scores")
    vdPMW1 = {
        'x': test_data1.loc[:, sublist2].reset_index(drop=True),
        'e': pd.DataFrame(np.squeeze(test_data1[["Dead"]]), columns = ["Dead"]).reset_index(drop=True),
        't': pd.DataFrame(np.squeeze(test_data1[["DX_LASTCONTACT_DEATH_MONTHS"]]), columns = ["DX_LASTCONTACT_DEATH_MONTHS"]).reset_index(drop=True)
    }

    a, b = treatmentGroupsECE(vdW1['x'], vdW1['t'], vdW1['e'])
    print("ECE/Margns: " + str(CPHStat(a, b, False, True)))
    a, b = treatmentGroupsRTOG(vdW1['x'], vdW1['t'], vdW1['e'])
    print("RTOG: " + str(CPHStat(a, b, False, True)))
    a, b = treatmentGroupsEORTC(vdW1['x'], vdW1['t'], vdW1['e'])
    print("EORTC: " + str(CPHStat(a, b, False, True)))
    a, b = treatmentGroupsChemoAll(vdW1['x'], vdW1['t'], vdW1['e'])
    print("Chemo: " + str(CPHStat(a, b, False, True)))


    a, b = treatmentGroupsECE(vdPMW1['x'], vdPMW1['t'], vdPMW1['e'])
    print(a.describe().to_string())
    print(b.describe().to_string())
    print("ECE/Margns, PM: " + str(CPHStat(a, b, True, True)))
    a, b = treatmentGroupsRTOG(vdPMW1['x'], vdPMW1['t'], vdPMW1['e'])
    print("RTOG, PM: " + str(CPHStat(a, b, True, True)))
    a, b = treatmentGroupsEORTC(vdPMW1['x'], vdPMW1['t'], vdPMW1['e'])
    print("EORTC, PM: " + str(CPHStat(a, b, True, True)))
    a, b = treatmentGroupsChemoAll(vdPMW1['x'], vdPMW1['t'], vdPMW1['e'])
    print("Chemo: " + str(CPHStat(a, b, True, True)))


# Estimate importance of each variable for a neural network model, where importance = concordance index when feature is included versus concordance index when feature is excluded
def variableImportance(model, X, T, E):
    ctmp = concordance_index(model, X, T, E)
    for column in X:
        tmpX = X.copy()
        tmpX[column] = np.random.permutation(tmpX[column])
        ctmp2 = concordance_index(model, tmpX, T, E)
        print(column + "     " + str(ctmp - ctmp2))

# Estimate importance of each variable for a neural network model, where importance = hazard ratio when feature is included versus hazard ratio when feature is excluded
def variableImportanceHR(model, X, T, E):
    hrtmp = hazardRatioNN(model, X, T, E, 0, False, False, "Chemo")
    for column in X:
        tmpX = X.copy()
        tmpX[column] = np.random.permutation(tmpX[column])
        hrtmp2 = hazardRatioNN(model, tmpX, T, E, 0, False, False, "Chemo")
        print(column + "     " + str(hrtmp2 - hrtmp))

# Estimate importance of each variable for a RSF model, where importance = hazard ratio when feature is included versus hazard ratio when feature is excluded
def variableImportanceChunk(model, X, T, E):
    ctmp = concordance_index_chunk(model, X, T, E)
    for column in X:
        tmpX = X.copy()
        tmpX[column] = np.random.permutation(tmpX[column])
        ctmp2 = concordance_index_chunk(model, tmpX, T, E)
        print(column + "     " + str(ctmp - ctmp2))

# Estimate importance of each variable for a RSF model, where importance = hazard ratio when feature is included versus hazard ratio when feature is excluded
def variableImportanceHRChunk(model, X, T, E):
    hrtmp = hazardRatioSFRIJ(model, X, T, E, 0, False)
    for column in X:
        tmpX = X.copy()
        tmpX[column] = np.random.permutation(tmpX[column])
        hrtmp2 = hazardRatioSFRIJ(model, tmpX, T, E, 0, False, False)
        print(column + "     " + str(hrtmp2 - hrtmp))


#------------------------------------------------------------------------
#
#  HYPERPARAMETER SEARCHES
#
#------------------------------------------------------------------------


#Select best hyperparameters for a DeepSurv model
#If CI = True, then select based on concordnace index; otherwise select based on HR between two groups
#This training first selects 80% of the dataset for all training purposes
#This 80% is then further divided into 5 equal parts
#A set of random hyperparameters is then chosen
#The model is trained leaving out one of the 5 parts of the 80% of training data, and tested on the part that was left out
#If the model has a higher concordance index or lower hazard ratio (depending on CIndex), the parameters of the model are then saved
def findNNCPHLRSingle2(dataS, trainCols, CI = True, testVar = "Chemo", RS = 0):
    structure = [{'activation': 'ReLU', 'num_units': 100}, {'activation': 'Tanh', 'num_units': 100}]
    activations = ["Atan", "BentIdentity", "BipolarSigmoid", "CosReLU", "ELU", "Gaussian", "Hardtanh", "InverseSqrt", "LeakyReLU", "LeCunTanh", "LogLog", "LogSigmoid", "ReLU", "SELU", "Sigmoid", "Sinc", "SinReLU","Softmax", "Softplus", "Softsign", "Swish", "Tanh"]

    units = [8, 16, 32, 64, 128]
    lr = [1e-4, 1e-5]
    num_epochs = [2000, 4000, 6000, 8000]
    dropout = [0.1, 0.2, 0.3]
    l2_reg = [1e-2, 1e-3, 1e-4]
    l2_smooth = [1e-2, 1e-3, 1e-4]
    batch_normalization = [False, True]
    num_layers = [1,2,3,4,5]
    bins = [10, 20, 50, 100]
    cindex = -1
    if not CI:
        cindex = 1
    train_dataSet, test_data, y_train, y_test = train_test_split(dataS, dataS, test_size=0.2, random_state=RS)
    shuffledTrain = train_dataSet.sample(frac=1)
    resultTrain = np.array_split(shuffledTrain, 5)
    tdC = {}
    vdC = {}
    for i in range(0,5):
        valid_data = resultTrain[i].reset_index(drop = True)
        valid_data = valid_data[
            (valid_data["ECE_micro"] == 0) & (valid_data["ECE_macro"] == 0) & (valid_data["Micro_Margins"] == 0) & (
                        valid_data["Gross_Margins"] == 0)].reset_index(drop=True)
        train_data = pd.concat(resultTrain[:i] + resultTrain[(i+1):])
        tdC[i] = {
        'x': train_data.loc[:, trainCols],
        'e': np.squeeze(train_data[["Dead"]]),
        't': np.squeeze(train_data[["DX_LASTCONTACT_DEATH_MONTHS"]]) }
        vdC[i] = {
            'x': valid_data.loc[:, trainCols],
            'e': np.squeeze(valid_data[["Dead"]]),
            't': np.squeeze(valid_data[["DX_LASTCONTACT_DEATH_MONTHS"]])
        }

    import random
    while True:
        binA = random.choice(bins)
        lrA = random.choice(lr)
        epochA = random.choice(num_epochs)
        dropoutA = random.choice(dropout)
        l2A = random.choice(l2_reg)
        l2sA = random.choice(l2_smooth)
        batchA = random.choice(batch_normalization)
        layerS = random.choice(num_layers)

        if layerS == 1:
            structure = [{"activation": "Atan", "num_units": 1}]
        if layerS == 2:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1}]
        if layerS == 3:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1}]
        if layerS == 4:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1}]
        if layerS == 5:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1}]
        for i in range(0, layerS):
            activationA = random.choice(activations)
            unitA = random.choice(units)
            structure[i] = {"activation": activationA, "num_units": unitA}
        print(structure)
        print(str(lrA) + " " + str(epochA) + " " + str(dropoutA) + " " + str(l2A) + " " + str(l2sA) + " " + str(batchA) + " " + str(binA))
        ctotal = 0
        hrtotal = 0
        for i in range(0, 5):
            cph = NonLinearCoxPHModel(structure, auto_scaler = True)
            cph.fit(tdC[i]['x'], tdC[i]['t'], tdC[i]['e'], verbose=False, lr=lrA, num_epochs=epochA, dropout=dropoutA, l2_reg=l2A, batch_normalization=batchA)
            hra = hazardRatioNN(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'], 0, False, False, testVar)
            if CI:
                ctmp = concordance_index(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'])
                ctotal += ctmp
                print("Concordance " + str(i) + ": " + str(ctmp) + " " + str(hra))
            else:
                ctotal += hra
                print("Hazard ratio: " + str(hra))
            hrtotal += hra/5
            if CI:
                if ctotal + (5-i)/50 < ((i+1)*cindex) and i > 0:
                    break
            else:
                if ctotal > (i + 1) * 0.95:
                    ctotal = 5
                    hrtotal *= 5 / (i + 1)
                    break
                if ctotal - (5 - i) / 15 > ((i + 1) * cindex) and i > 0:
                    ctotal = 5
                    hrtotal *= 5 / (i + 1)
                    break
        ctotal = ctotal/5
        hravg = hrtotal
        if CI:
            print("Average concordance: " + str(ctotal) + ", average HR, LR: " + str(hravg) )
        else:
            print("Average HR: " + str(hravg))
        if (((CI) and (ctotal > cindex)) or ((not CI) and (ctotal < cindex))):
            print("Model better (non-corrected ratios) " + str(ctotal) + " " + str(cindex))
            cindex = ctotal

#Select best hyperparameters for an N-MTLR model
#If CI = True, then select based on concordnace index; otherwise select based on HR between two groups
#This training first selects 80% of the dataset for all training purposes
#This 80% is then further divided into 5 equal parts
#A set of random hyperparameters is then chosen
#The model is trained leaving out one of the 5 parts of the 80% of training data, and tested on the part that was left out
#If the model has a higher concordance index or lower hazard ratio (depending on CIndex), the parameters of the model are then saved
def findNNMTMLRSingle2(dataS, trainCols, CI = True, testVar = "Chemo", RS = 0):
    structure = [{'activation': 'ReLU', 'num_units': 100}, {'activation': 'Tanh', 'num_units': 100}]
    activations = ["Atan", "BentIdentity", "BipolarSigmoid", "CosReLU", "ELU", "Gaussian", "Hardtanh", "InverseSqrt", "LeakyReLU", "LeCunTanh", "LogLog", "LogSigmoid", "ReLU", "SELU", "Sigmoid", "Sinc", "SinReLU","Softmax", "Softplus", "Softsign", "Swish", "Tanh"]

    units = [8, 16, 32, 64, 128]
    lr = [1e-4, 1e-5]
    num_epochs = [2000, 4000, 6000, 8000]
    dropout = [0.1, 0.2, 0.3]
    l2_reg = [1e-2, 1e-3, 1e-4]
    l2_smooth = [1e-2, 1e-3, 1e-4]
    batch_normalization = [False, True]
    num_layers = [1,2,3,4,5]
    bins = [10, 20, 50, 100]
    cindex = -1
    if not CI:
        cindex = 1
    train_dataSet, test_data, y_train, y_test = train_test_split(dataS, dataS, test_size=0.2, random_state=RS)
    shuffledTrain = train_dataSet.sample(frac=1)
    resultTrain = np.array_split(shuffledTrain, 5)
    tdC = {}
    vdC = {}
    for i in range(0,5):
        valid_data = resultTrain[i].reset_index(drop = True)
        valid_data = valid_data[
            (valid_data["ECE_micro"] == 0) & (valid_data["ECE_macro"] == 0) & (valid_data["Micro_Margins"] == 0) & (
                        valid_data["Gross_Margins"] == 0)].reset_index(drop=True)
        train_data = pd.concat(resultTrain[:i] + resultTrain[(i+1):])
        tdC[i] = {
        'x': train_data.loc[:, trainCols],
        'e': np.squeeze(train_data[["Dead"]]),
        't': np.squeeze(train_data[["DX_LASTCONTACT_DEATH_MONTHS"]]) }
        vdC[i] = {
            'x': valid_data.loc[:, trainCols],
            'e': np.squeeze(valid_data[["Dead"]]),
            't': np.squeeze(valid_data[["DX_LASTCONTACT_DEATH_MONTHS"]])
        }

    import random
    while True:
        binA = random.choice(bins)
        lrA = random.choice(lr)
        epochA = random.choice(num_epochs)
        dropoutA = random.choice(dropout)
        l2A = random.choice(l2_reg)
        l2sA = random.choice(l2_smooth)
        batchA = random.choice(batch_normalization)
        layerS = random.choice(num_layers)

        if layerS == 1:
            structure = [{"activation": "Atan", "num_units": 1}]
        if layerS == 2:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1}]
        if layerS == 3:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1}]
        if layerS == 4:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1}]
        if layerS == 5:
            structure = [{"activation": "Atan", "num_units": 1}, {"activation": "Atan", "num_units": 1},
                         {"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1},{"activation": "Atan", "num_units": 1}]
        for i in range(0, layerS):
            activationA = random.choice(activations)
            unitA = random.choice(units)
            structure[i] = {"activation": activationA, "num_units": unitA}
        print(structure)
        print(str(lrA) + " " + str(epochA) + " " + str(dropoutA) + " " + str(l2A) + " " + str(l2sA) + " " + str(batchA) + " " + str(binA))
        ctotal = 0
        hrtotal = 0
        for i in range(0, 5):
            structure = [{'activation': 'Sinc', 'num_units': 64},
                                     {'activation': 'Sigmoid', 'num_units': 32},
                                     {'activation': 'LogLog', 'num_units': 16},
                                     {'activation': 'Atan', 'num_units': 8}]

            cph = NeuralMultiTaskModel(structure, auto_scaler = True, bins=binA)
            try:
                cph.fit(tdC[i]['x'], tdC[i]['t'], tdC[i]['e'], verbose=False, lr=lrA, num_epochs=epochA, dropout=dropoutA, l2_reg=l2A, l2_smooth = l2sA, batch_normalization=batchA)
                hra = hazardRatioNN(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'])
                if CI:
                    ctmp = concordance_index(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'])
                    ctotal += ctmp
                    print("Concordance " + str(i) + ": " + str(ctmp) + " " + str(hra))
                else:
                    ctotal += hra
                    print("Hazard ratio: " + str(hra))
                hrtotal += hra/5
                if CI:
                    if ctotal + (5-i)/50 < ((i+1)*cindex) and i > 0:
                        break
                else:
                    if ctotal > (i+1)*0.95:
                        ctotal = 5
                        hrtotal *= 5 / (i+1)
                        break
                    if ctotal - (5-i)/15 > ((i+1)*cindex) and i > 0:
                        ctotal = 5
                        hrtotal *= 5 / (i+1)
                        break
            except:
                break
        ctotal = ctotal/5
        hravg = hrtotal
        if CI:
            print("Average concordance: " + str(ctotal) + ", average HR, LR: " + str(hravg) )
        else:
            print("Average HR: " + str(hravg))

        if (((CI) and (ctotal > cindex)) or ((not CI) and (ctotal < cindex))):
            print("Model better (non-corrected ratios) " + str(ctotal) + " " + str(cindex))
            cindex = ctotal

#Select best hyperparameters for an RSF model
#If CI = True, then select based on concordnace index; otherwise select based on HR between two groups
#This training first selects 80% of the dataset for all training purposes
#This 80% is then further divided into 5 equal parts
#A set of random hyperparameters is then chosen
#The model is trained leaving out one of the 5 parts of the 80% of training data, and tested on the part that was left out
#If the model has a higher concordance index or lower hazard ratio (depending on CIndex), the parameters of the model are then saved
def findRSFLRSingle2CV(dataS, trainCols, CI = True, testVar = "Chemo", RS = 0):
    max_features = [0.1, 0.2, 'sqrt', 'log2', 'all']
    min_node_size = [5, 10, 20, 40, 80]
    alpha = [0.01, 0.05, 0.1]
    minprop = [0.05, 0.1, 0.2 ]
    max_depth = [10, 20, 40]
    sample_size_pct = [0.2, 0.4, 0.6, 0.8]
    importance_mode = ['impurity', 'impurity_corrected', 'permutation', 'normalized_permutation']
    num_trees = [20, 40, 60, 80, 100]
    cindex = -1
    if not CI:
        cindex = 1
    train_dataSet, test_data, y_train, y_test = train_test_split(dataS, dataS, test_size=0.2, random_state=RS)
    shuffledTrain = train_dataSet.sample(frac=1)
    resultTrain = np.array_split(shuffledTrain, 5)
    tdC = {}
    vdC = {}
    for i in range(0,5):
        valid_data = resultTrain[i].reset_index(drop = True)
        valid_data = valid_data[
            (valid_data["ECE_micro"] == 0) & (valid_data["ECE_macro"] == 0) & (valid_data["Micro_Margins"] == 0) & (
                        valid_data["Gross_Margins"] == 0)].reset_index(drop=True)
        train_data = pd.concat(resultTrain[:i] + resultTrain[(i+1):])
        tdC[i] = {
        'x': train_data.loc[:, trainCols],
        'e': np.squeeze(train_data[["Dead"]]),
        't': np.squeeze(train_data[["DX_LASTCONTACT_DEATH_MONTHS"]]) }
        vdC[i] = {
            'x': valid_data.loc[:, trainCols],
            'e': np.squeeze(valid_data[["Dead"]]),
            't': np.squeeze(valid_data[["DX_LASTCONTACT_DEATH_MONTHS"]])
        }

    import random
    while True:
        max_featuresA = random.choice(max_features)
        min_node_sizeA = random.choice(min_node_size)
        alphaA = random.choice(alpha)
        minpropA = random.choice(minprop)
        sample_size_pctA = random.choice(sample_size_pct)
        importance_modeA = random.choice(importance_mode)
        num_treesA = random.choice(num_trees)
        max_depthA = random.choice(max_depth)
        ctotal = 0
        hrtotal = 0
        print(str(num_treesA) + " " + str(max_featuresA) + " " + str(max_depthA) + " " + str(
            min_node_sizeA) + " " + str(alphaA) + " " + str(minpropA) + " " + str(sample_size_pctA) + " " + str(
            importance_modeA))
        for i in range(0, 3):
            try:
                cph = RandomSurvivalForestModel(num_trees=num_treesA)
                cph.fit(tdC[i]['x'], tdC[i]['t'], tdC[i]['e'], max_features=max_featuresA, max_depth=max_depthA,
                        min_node_size=min_node_sizeA, num_threads=4, sample_size_pct=sample_size_pctA,
                        importance_mode=importance_modeA, save_memory=True)
                recij = treatmentRecForestRisk(cph, vdC[i]['x'], 0)
                hra = hazardRatioSF(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'], 0, recij, False, False)
                if CI:
                    ctmp = concordance_index(cph, vdC[i]['x'], vdC[i]['t'], vdC[i]['e'])
                    ctotal += ctmp
                    print("Concordance " + str(i) + ": " + str(ctmp) + " " + str(hra))
                else:
                    ctotal += hra
                    print("Hazard ratio: " + str(hra))
                hrtotal += hra / 3
                if CI:
                    if ctotal + (3 - i) / 30 < ((i + 1) * cindex) and i > 0:
                        break
                else:
                    if ctotal > (i + 1) * 0.95:
                        ctotal = 3
                        hrtotal *= 3 / (i + 1)
                        break
                    if ctotal - (3 - i) / 9 > ((i + 1) * cindex) and i > 0:
                        ctotal = 3
                        hrtotal *= 3 / (i + 1)
                        break
            except:
                break
        ctotal = ctotal / 3
        hravg = hrtotal
        if CI:
            print("Average concordance: " + str(ctotal) + ", average HR, LR: " + str(hravg))
        else:
            print("Average HR: " + str(hravg))
        if (((CI) and (ctotal > cindex)) or ((not CI) and (ctotal < cindex) and (ctotal > 0))):
                print("Model better (non-corrected ratios) " + str(ctotal) + " " + str(cindex))
                cindex = ctotal


#------------------------------------------------------------------------
#
#  MODEL AND DATABASE LOADING AND GENERATION
#
#------------------------------------------------------------------------


def checkModel(model = None, vd = None, testVar = "Chemo", modelType = "NNCPH", IPTW=True, appendString ="", ratio = 0.3):
    #vd['x'].loc[vd['x'].LIFE > 15, "LIFE"] = 15
    modelString = "DeepSurv"
    if modelType == "NNMTLR":
        modelString = "N-MTLR"
    if modelType == "RSF":
        modelString = "RSF"
    if modelType == "RSF":
        rec_ij = treatmentRecForestRisk(model, vd['x'], 0)
    else:
        rec_ij = treatmentRecNNRisk(model, vd['x'], 0)
    b = 0
    c = 0
    if IPTW:
        a, b, c = modelPlots(model, vd['x'], vd['t'], vd['e'], modelString, ratio = ratio, rec_ij = rec_ij)
    #plotSurvivalArray([rec_ij, rec_ij], vd['x'], vd['t'], vd['e'], modelString, 0)
    sumtext = "Hazard ratio, " + modelString + " model" + appendString
    print(sumtext + ":")
    print("__________________________________________________________")
    hr = hazardRatio(model, vd['x'], vd['t'], vd['e'], rec_ij, ratio, testVar, propMatch = False, printSummary = False, printSurvival = False)
    print(hazardRatio(model, vd['x'], vd['t'], vd['e'], rec_ij, ratio, testVar, propMatch = False, printSummary = True, printSurvival = False))
    if IPTW:
        print()
        print()
        print(sumtext + ", with IPTW:")
        print("__________________________________________________________")
        print(hazardRatio(model, vd['xPM'], vd['t'], vd['e'], rec_ij, ratio, testVar, propMatch = True, printSummary = True, printSurvival = False))
    return rec_ij, hr, b, c

def getModel(load = True, testVar = "Chemo", modelType = "NNCPH", td = None, hp = None):
    model = None
    if load:
        if modelType == "NNCPH":
            model = NonLinearCoxPHModel(structure = hp['structure'], auto_scaler=True)
            model.load("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\cphModelFinal" + testVar + ".zip")
        if modelType == "NNMTLR":
            model = NeuralMultiTaskModel(structure = hp['structure'], bins = hp['bins'], auto_scaler=True)
            model.load("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\nnmtmModelFinal" + testVar + ".zip")
        if modelType == "RSF":
            model = RandomSurvivalForestModel(num_trees = hp['num_trees'])
            model.load("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\rsfModelFinal" + testVar + ".zip")
    else:
        if modelType == "NNCPH":
            model = NonLinearCoxPHModel(structure = hp['structure'], auto_scaler=True)
            model.fit(td['x'], td['t'], td['e'], verbose=False, lr=hp['lr'], num_epochs = hp['num_epochs'], dropout = hp['dropout'], l2_reg= hp['l2_reg'], batch_normalization=hp['batch_normalization'])
            model.save("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\cphModelFinal" + testVar + ".zip")
        if modelType == "NNMTLR":
            model = NeuralMultiTaskModel(structure = hp['structure'], bins = hp['bins'], auto_scaler=True)
            model.fit(td['x'], td['t'], td['e'], verbose=False,  lr=hp['lr'], num_epochs = hp['num_epochs'], dropout = hp['dropout'], l2_reg= hp['l2_reg'], l2_smooth=hp['l2_smooth'], batch_normalization = hp['batch_normalization'])
            model.save("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\nnmtmModelFinal" + testVar + ".zip")
        if modelType == "RSF":
            model = RandomSurvivalForestModel(num_trees=hp['num_trees'])
            model.fit(td['x'], td['t'], td['e'], max_features=hp['max_features'], max_depth=hp['max_depth'], min_node_size=hp['min_node_size'], num_threads=3, sample_size_pct=hp['sample_size_pct'], importance_mode=hp['importance_mode'], save_memory=True)
            model.save("C:\\Users\\fhowa_000\\Pycharm Workspace\\hnscc\\rsfModelFinal" + testVar + ".zip")
    return model

def getDataset(dataS, trainCols, testVar = "Chemo", RS = 0, validation = False, HPV = 0, age = 0):
    train_dataSet, test_data, y_train, y_test = train_test_split(dataS, dataS, test_size=0.2, random_state=RS)
    train_dataSet = train_dataSet.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    if not validation:
        tdW = {
            'x': train_dataSet.loc[:, trainCols],
            'e': np.squeeze(train_dataSet[["Dead"]]),
            't': np.squeeze(train_dataSet[["DX_LASTCONTACT_DEATH_MONTHS"]])
        }
        return tdW
    else:
        import contextlib, os
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            from pymatch.Matcher import Matcher
            sublist = trainCols.copy()
            sublist.append("Dead")
            sublist.append("DX_LASTCONTACT_DEATH_MONTHS")
            sublist.append("AGE")
            test = test_data[test_data[testVar] == 1]
            control = test_data[test_data[testVar] == 0]
            test = test[sublist]
            control = control[sublist]
            excludeList = ["DX_LASTCONTACT_DEATH_MONTHS", "Dead", "ECE_micro", "ECE_macro", "Micro_Margins", "Gross_Margins", "Multiagent"]
            m = Matcher(test, control, yvar=testVar, exclude=excludeList)
            np.random.seed(20170925)
            m.fit_scores(balance=True, nmodels=10)
            m.predict_scores()
            m.data["scores"] = m.data[testVar] / m.data["scores"] + (1 - m.data[testVar]) / (1 - m.data["scores"])
            test_dataPM = m.data.reset_index(drop=True)
        sublist2 = trainCols.copy()
        sublist2.append("scores")
        if (HPV == 1):
            test_dataPM = test_dataPM[(test_dataPM.HPV_OP == 1)].reset_index(drop=True)
        if (HPV == 2):
            test_dataPM = test_dataPM[(test_dataPM.HPV_OP == 0) & (test_dataPM.HPV_OTHER == 0)].reset_index(drop=True)
        if (age == 1):
            test_dataPM = test_dataPM[test_dataPM.AGE > 71].reset_index(drop=True)
        if (age == 2):
            test_dataPM = test_dataPM[test_dataPM.AGE < 70].reset_index(drop=True)
        test_dataPM = test_dataPM.reset_index(drop=True)
        #print(test_dataPM.describe().to_string())
        vdW = {
            'x': test_dataPM.loc[:, trainCols],
            'xPM': test_dataPM.loc[:, sublist2],
            'e': np.squeeze(test_dataPM[["Dead"]]),
            't': np.squeeze(test_dataPM[["DX_LASTCONTACT_DEATH_MONTHS"]])
        }
        return vdW


## Generate estimates of the HR for treatment according to EORTC and RTOG trial inclusion criteria, with and without propensity matching
def checkStandard(vdW1, trainCols, RS, trainSetRisk = 0, testSetRisk = 0, testVar = "Chemo"):
    print()
    print()
    print("Hazard ratio, RTOG 95-01 Model:")
    print("__________________________________________________________")
    a, b = treatmentGroupsRTOG(vdW1['x'], vdW1['t'], vdW1['e'])
    CPHStat(a, b, False, True, False)
    print()
    print()
    print("Hazard ratio, EORTC 22931 Model:")
    print("__________________________________________________________")
    a, b = treatmentGroupsEORTC(vdW1['x'], vdW1['t'], vdW1['e'])
    CPHStat(a, b, False, True, False)
    plotSurvivalRTOG(vdW1['x'], vdW1['t'], vdW1['e'])
    plotSurvivalEORTC(vdW1['x'], vdW1['t'], vdW1['e'])
    print()
    print()
    print("Hazard ratio, RTOG 95-01 Model, with IPTW:")
    print("__________________________________________________________")
    a, b = treatmentGroupsRTOG(vdW1['xPM'], vdW1['t'], vdW1['e'])
    CPHStat(a, b, True, True, False)
    print()
    print()
    print("Hazard ratio, EORTC 22931 Model, with IPTW:")
    print("__________________________________________________________")
    a, b = treatmentGroupsEORTC(vdW1['xPM'], vdW1['t'], vdW1['e'])
    CPHStat(a, b, True, True, False)

# Generate a limited dataset of desired features
# defResect = True - only include patients with definitive surgical resection; = False - exclude patients with definitive surgical resection
def generateData(filename, defResect = True, verbose = False, missing = False):
    cols = [
        "TNM_PATH_T",
        "TNM_PATH_N",
        "TNM_PATH_M",
        "TNM_CLIN_T",
        "TNM_CLIN_N",
        "TNM_CLIN_M",
        "CS_SITESPECIFIC_FACTOR_3",
        "CS_SITESPECIFIC_FACTOR_4",
        "CS_SITESPECIFIC_FACTOR_5",
        "CS_SITESPECIFIC_FACTOR_6",
        "CS_SITESPECIFIC_FACTOR_9",
        "CS_SITESPECIFIC_FACTOR_11",
        "MTS_LN_DIS",
        "RX_SUMM_SCOPE_REG_LN_SUR",
        "MTS_SURG",
        "MTS_SIMPLE_PRIMARY",
        "MTS_SIMPLE_PRIMARY_2",
        "HISTOLOGY",
        "DX_DEFSURG_STARTED_DAYS",
        "DX_RAD_STARTED_DAYS",
        "DX_CHEMO_STARTED_DAYS",
        "AGE",
        "PUF_VITAL_STATUS",
        "DX_LASTCONTACT_DEATH_MONTHS",
        "ANALYTIC_STAGE_GROUP",
        "MTS_SIMPLE_AGE",
        "SEX",
        "MTS_CDCC",
        "FACILITY_TYPE_CD",
        "MTS_RACE",
        "MTS_HISPANIC",
        "LYMPH_VASCULAR_INVASION",
        "RX_SUMM_SURGICAL_MARGINS",
        "MTS_CS_HIGH_RISK_HPV",
        "GRADE",
        "RX_SUMM_CHEMO",
        "MTS_RT1",
        "TUMOR_SIZE",
        "MTS_CS11_DEPTH",
        "MTS_LN_POS",
        "MTS_LN_DIS",
        "MTS_LN_DIS",
        "MTS_IMMUNO",
        "YEAR_OF_DIAGNOSIS",
        "MTS_TX_PKG_TIME",
        "RAD_REGIONAL_DOSE_CGY",
        "RAD_BOOST_DOSE_CGY",
        "MTS_DISTANCE",
        "MTS_ChemoRT"
    ]
    data = pd.read_csv('hnscc.csv', error_bad_lines=False, index_col=False, dtype='unicode', usecols=cols)
    np.set_printoptions(threshold=sys.maxsize)

    if verbose:
        print("Total number of cases: " + str(len(data.index)))

    data = data[data.MTS_SIMPLE_PRIMARY.str.contains("OC") | data.MTS_SIMPLE_PRIMARY.str.contains(
        "OPX") | data.MTS_SIMPLE_PRIMARY.str.contains("HPX") | data.MTS_SIMPLE_PRIMARY.str.contains(
        "Larynx")]  # | data.MTS_SIMPLE_PRIMARY.str.contains("HPX") | data.MTS_SIMPLE_PRIMARY.str.contains("Larynx")

    if verbose:
        print("OC, OPX, HPX, and Larynx sites: " + str(len(data.index)))

    #Squamous histology
    data = data[data.HISTOLOGY.str.contains("quamous")]
    if verbose:
        print("Squamous Histology: " + str(len(data.index)))

    data = data[(data.TNM_CLIN_M != '   c1') & (data.TNM_PATH_M != '   p1')]

    if verbose:
        print("No metastatic disease: " + str(len(data.index)))

    # Baseline Inclusion criteria
    # data = data[(data.RX_SUMM_SCOPE_REG_LN_SUR=='Regional lymph node surgery')]
    if defResect:
        data = data[data.MTS_SURG == 'Surgical resection']
        if verbose:
            print("Definitve Resection: " + str(len(data.index)))

        data = data[(data.RX_SUMM_SCOPE_REG_LN_SUR == 'Regional lymph node surgery') | (
                    data.TNM_CLIN_N == '   c0')]  # Consider not including this
        # should we include the non-neck dissected N0
        if verbose:
            print("cN0 neck or regional lymph node surgery: " + str(len(data.index)))

    else:
        data = data[data.MTS_SURG != 'Surgical resection']

    data = data[data.MTS_RT1 == "Radiotherapy"]
    if verbose:
        print("Received radiotherapy: " + str(len(data.index)))

    data = data.reset_index()

    dataS = data[['AGE']]

    dataS = dataS.replace(' ', np.nan)
    # imp = SimpleImputer(missing_values = np.nan, strategy='mean')
    # dataS = pd.DataFrame(imp.fit_transform(dataS), columns = ['AGE'])

    maleLife = [76.04, 75.52, 74.55, 73.58, 72.59, 71.6, 70.62, 69.63, 68.64, 67.64, 66.65, 65.66, 64.66, 63.67, 62.68,
                61.7, 60.73, 59.76, 58.81, 57.86, 56.91, 55.98, 55.05, 54.13, 53.22, 52.3, 51.38, 50.47, 49.55, 48.63,
                47.72, 46.8, 45.89, 44.97, 44.06, 43.15, 42.23, 41.32, 40.41, 39.5, 38.59, 37.69, 36.78, 35.88, 34.98,
                34.08, 33.19, 32.3, 31.43, 30.55, 29.69, 28.84, 27.99, 27.16, 26.34, 25.52, 24.72, 23.93, 23.15, 22.37,
                21.61, 20.85, 20.11, 19.37, 18.65, 17.92, 17.2, 16.49, 15.78, 15.09, 14.4, 13.73, 13.07, 12.43, 11.8,
                11.18, 10.58, 10, 9.43, 8.88, 8.34, 7.82, 7.32, 6.84, 6.38, 5.94, 5.52, 5.12, 4.75, 4.4, 4.08, 3.78,
                3.5, 3.25, 3.03, 2.83, 2.66, 2.51, 2.37, 2.25, 2.13, 2.02, 1.91, 1.81, 1.71, 1.61, 1.52, 1.43, 1.35,
                1.27, 1.19, 1.11, 1.04, 0.97, 0.91, 0.84, 0.78, 0.73, 0.67, 0.62]
    femaleLife = [80.99, 80.43, 79.46, 78.48, 77.49, 76.5, 75.51, 74.52, 73.53, 72.54, 71.54, 70.55, 69.56, 68.56,
                  67.57, 66.58, 65.6, 64.62, 63.63, 62.66, 61.68, 60.71, 59.73, 58.76, 57.8, 56.83, 55.86, 54.9, 53.93,
                  52.97, 52.01, 51.05, 50.09, 49.14, 48.19, 47.23, 46.28, 45.34, 44.39, 43.45, 42.5, 41.56, 40.62,
                  39.69, 38.76, 37.83, 36.9, 35.98, 35.07, 34.16, 33.26, 32.36, 31.48, 30.59, 29.72, 28.85, 27.99,
                  27.13, 26.28, 25.44, 24.6, 23.76, 22.94, 22.12, 21.3, 20.49, 19.69, 18.89, 18.11, 17.33, 16.57, 15.82,
                  15.09, 14.37, 13.66, 12.97, 12.29, 11.62, 10.98, 10.35, 9.74, 9.15, 8.58, 8.04, 7.51, 7.01, 6.53,
                  6.07, 5.64, 5.23, 4.85, 4.5, 4.18, 3.88, 3.61, 3.37, 3.16, 2.96, 2.79, 2.63, 2.48, 2.33, 2.19, 2.06,
                  1.93, 1.81, 1.69, 1.58, 1.47, 1.37, 1.27, 1.18, 1.09, 1.01, 0.93, 0.86, 0.79, 0.73, 0.67, 0.62]

    dataS = pd.concat([dataS, addDummy('MTS_RACE', ['Black'], 'Black', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_RACE', ['White'], 'White', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_HISPANIC', ['Hispanic'], 'Hispanic', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('SEX', ['Male'], 'Male', data)], axis=1)
    dataS = pd.concat([dataS, addRow("DX_DEFSURG_STARTED_DAYS", data)], axis=1)
    dataS = pd.concat([dataS, addRow("DX_RAD_STARTED_DAYS", data)], axis=1)
    dataS = pd.concat([dataS, addRow("DX_CHEMO_STARTED_DAYS", data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_CHEMO', ['Chemotherapy recommended, unknown if administered', 'Unknown if recommended or administered'], 'Unknown_chemo', data)], axis=1)

    dataS["DX_RAD_STARTED_DAYS"] = dataS["DX_RAD_STARTED_DAYS"] - dataS["DX_DEFSURG_STARTED_DAYS"]
    dataS["DX_CHEMO_STARTED_DAYS"] = dataS["DX_CHEMO_STARTED_DAYS"] - dataS["DX_DEFSURG_STARTED_DAYS"]
    dataT = dataS[['AGE', 'Male']]
    ages = dataT['AGE'].values.astype(int)
    sexes = dataT['Male'].values.astype(int)
    maleLife = np.array(maleLife)
    femaleLife = np.array(femaleLife)
    dataT.loc[:, 'AGE'] = np.where(sexes == 0, maleLife[ages], femaleLife[ages])
    dataT.columns = ['LIFE', 'Male']
    dataS = pd.concat([dataS, dataT[['LIFE']]], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_AGE', ['<=50y'], 'Young', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('FACILITY_TYPE_CD', ['Academic/Research Program'], 'Academic', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_CDCC', ['0-1'], 'Low_Comorbidity', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SURG', ['Debulking'], 'Debulking', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_SCOPE_REG_LN_SUR', ['Regional lymph node surgery'], 'RegionalLND', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_CDCC', ['>=3'], 'High_Comorbidity', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY', ['OC', 'OPX'], 'OC_OPX', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_T', ['c1', 'c1a', 'c1b'], 'cT1', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_T', ['c2', 'c2a', 'c2b'], 'cT2', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_T', ['c3', 'c3a', 'c3b'], 'cT3', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_T', ['c4', 'c4a', 'c4b'], 'cT4', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_N', ['c1', 'c1a', 'c1b'], 'cN1', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_N', ['c2', 'c2a', 'c2b'], 'cN2', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_CLIN_N', ['c3', 'c3a', 'c3b'], 'cN3', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('LYMPH_VASCULAR_INVASION', ['Present'], 'LVI', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('CS_SITESPECIFIC_FACTOR_9',
                                       ['Regional lymph node(s) involved pathologically microscopic ECE'],
                                       'ECE_micro', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('CS_SITESPECIFIC_FACTOR_9',
                                       ['Regional lymph node(s) involved pathologically macroscopic ECE'],
                                       'ECE_macro', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_SURGICAL_MARGINS',
                                       ['Microscopic residual tumor  Cannot be seen by the naked eye',
                                        'Residual tumor, NOS  Involvement is indicated, but not otherwise specified'],
                                       'Micro_Margins', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_SURGICAL_MARGINS', [
        'Macroscopic residual tumor, Gross tumor of the primary site which is visible to the naked eye'],
                                       'Gross_Margins', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_CHEMO', ['Multiagent chemotherapy'], 'Multiagent', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('RX_SUMM_CHEMO', ['Multiagent chemotherapy', 'Single-agent chemotherapy','Chemotherapy administered, type and number of agents not documented'], 'Chemo', data)], axis=1)
    #dataS = pd.concat([dataS, addDummyTx('DX_CHEMO_STARTED_DAYS', 'Chemo', data)], axis=1)
    dataS = pd.concat([dataS, txSequence('DX_RAD_STARTED_DAYS', 'DX_CHEMO_STARTED_DAYS', 'Induction', data)], axis=1)
    # dataS = pd.concat([dataS, addDummy('MTS_CS_HIGH_RISK_HPV', ['HPV positive, high risk', 'HPV positive, low risk', 'HPV positive, NOS, risk and type(s) not stated'], 'HPV2')], axis = 1)


    dataS = pd.concat([dataS, addDummy('GRADE', ['Well differentiated, differentiated, NOS'], 'Well_Differentiated', data)],
                      axis=1)
    dataS = pd.concat([dataS, addDummy('GRADE', [
        'Moderately differentiated, moderately well differentiated, intermediate differentiation'],
                                       'Moderately_Differentiated', data)], axis=1)
    dataS = pd.concat(
        [dataS, addDummy('GRADE', ['Poorly differentiated', 'Undifferentiated, anaplastic'], 'Poorly_Differentiated', data)],
        axis=1)
    dataS = pd.concat([dataS, addDummy('GRADE', ['Undifferentiated, anaplastic'], 'Anaplastic', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_T', ['p1', 'p1a', 'p1b'], 'pT1', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_T', ['p2', 'p2a', 'p2b'], 'pT2', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_T', ['p3', 'p3a', 'p3b'], 'pT3', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_T', ['p4', 'p4a', 'p4b'], 'pT4', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_N', ['p1', 'p1a', 'p1b'], 'pN1', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_N', ['p2', 'p2a', 'p2b'], 'pN2', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('TNM_PATH_N', ['p3', 'p3a', 'p3b'], 'pN3', data)], axis=1)
    dataS = pd.concat([dataS, addWorstNStage(data)], axis=1)
    dataS = pd.concat([dataS, addWorstTStage(data)], axis=1)
    dataS = pd.concat([dataS, sizeArray(data, True)], axis=1)
    dataS = pd.concat([dataS, Depth(data, True)], axis=1)
    dataS = pd.concat([dataS, NumLNDis(data, True, True)], axis=1)
    dataS = pd.concat([dataS, NumLNPos(data, True, True)], axis=1)
    dataS = pd.concat([dataS, NumLNDis(data, False, True)], axis=1)
    dataS = pd.concat([dataS, NumLNPos(data, False, True)], axis=1)
    dataS = pd.concat([dataS, txPackageTime(data, True)], axis=1)
    dataS = pd.concat([dataS, yearGroup(data, True)], axis=1)
    dataS = pd.concat([dataS, totalRTDose(data)], axis=1)
    dataS = pd.concat([dataS, nodePositiveArray(data, False)], axis=1)
    dataS = pd.concat([dataS, travelDistance(data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_IMMUNO', ['Immunotherapy'], 'Immunotherapy', data)], axis=1)


    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataS = pd.DataFrame(imp.fit_transform(dataS), columns=dataS.columns)
    dataS = pd.concat([dataS, addHPV(data)], axis=1)
    # All Locations list:
    # 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'HP', 'BOT', 'SP', 'PW', 'Ton', 'Glot', 'SubGlot', 'SGL','HP PC', 'HP PS', 'HP PW'

    # Oral Cavity cancer locations 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'HP'
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-Tongue'], 'Tongue', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-FOM'], 'FOM', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-RMT'], 'RMT', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-BucMuc'], 'BucMuc', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-AlvRid'], 'AlvRid', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OC-HP'], 'HP', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY', ['OC'], 'OC', data)], axis=1)

    # Oral Cavity cancer locations 'BOT', 'SP', 'PW', 'Ton'
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OPX-BOT'], 'BOT', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OPX-SP'], 'SP', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OPX-PW'], 'PW', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['OPX-Ton'], 'Ton', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY', ['OPX'], 'OPX', data)], axis=1)

    # Oral Cavity cancer locations 'Glot', 'SubGlot', 'SGL'
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['L-Glot'], 'Glot', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['L-SubGlot'], 'SubGlot', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['L-SGL'], 'SGL', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY', ['Larynx'], 'Larynx', data)], axis=1)

    # Oral Cavity cancer locations 'HP PC', 'HP PS', 'HP PW'
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['HPX-PC'], 'HP_PC', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['HPX-PS'], 'HP_PS', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY_2', ['HPX-PW'], 'HP_PW', data)], axis=1)
    dataS = pd.concat([dataS, addDummy('MTS_SIMPLE_PRIMARY', ['HPX'], 'HPX', data)], axis=1)
    dataS = pd.concat([dataS, addCategoricalTN(data)], axis=1)

    if missing:
        data = pd.concat([data, addRow("DX_DEFSURG_STARTED_DAYS", data, "SURG_START")], axis=1)
        data = pd.concat([data, addRow("DX_RAD_STARTED_DAYS", data, "RT_START")], axis=1)
        data['CS_SITESPECIFIC_FACTOR_11'] = Depth(data, False)

        data = pd.concat([data, addDummy('TNM_CLIN_T', ['c1', 'c1a', 'c1b'], 'cT1', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_T', ['c2', 'c2a', 'c2b'], 'cT2', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_T', ['c3', 'c3a', 'c3b'], 'cT3', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_T', ['c4', 'c4a', 'c4b'], 'cT4', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_N', ['c1', 'c1a', 'c1b'], 'cN1', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_N', ['c2', 'c2a', 'c2b'], 'cN2', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_CLIN_N', ['c3', 'c3a', 'c3b'], 'cN3', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_T', ['p1', 'p1a', 'p1b'], 'pT1', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_T', ['p2', 'p2a', 'p2b'], 'pT2', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_T', ['p3', 'p3a', 'p3b'], 'pT3', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_T', ['p4', 'p4a', 'p4b'], 'pT4', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_N', ['p1', 'p1a', 'p1b'], 'pN1', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_N', ['p2', 'p2a', 'p2b'], 'pN2', data)], axis=1)
        data = pd.concat([data, addDummy('TNM_PATH_N', ['p3', 'p3a', 'p3b'], 'pN3', data)], axis=1)
        data = pd.concat([data, addDummy('RX_SUMM_CHEMO', ['Multiagent chemotherapy', 'Single-agent chemotherapy','Chemotherapy administered, type and number of agents not documented'], 'Chemo', data)], axis=1)
        data = pd.concat([data, addWorstNStage(data)], axis=1)
        data = pd.concat([data, addWorstTStage(data)], axis=1)
        data = pd.concat([data, addDummy('MTS_IMMUNO', ['Immunotherapy'], 'Immunotherapy', data)], axis=1)
        data = pd.concat([data, totalRTDose(data)], axis=1)
        data = pd.concat([data, addHPV(data)], axis=1)
        data = pd.concat([data, addDummy('CS_SITESPECIFIC_FACTOR_9',
                                           ['Regional lymph node(s) involved pathologically microscopic ECE'],
                                           'ECE_micro', data)], axis=1)
        data = pd.concat([data, addDummy('CS_SITESPECIFIC_FACTOR_9',
                                           ['Regional lymph node(s) involved pathologically macroscopic ECE'],
                                           'ECE_macro', data)], axis=1)
        data = pd.concat([data, addDummy('RX_SUMM_SURGICAL_MARGINS',
                                           ['Microscopic residual tumor  Cannot be seen by the naked eye',
                                            'Residual tumor, NOS  Involvement is indicated, but not otherwise specified'],
                                           'Micro_Margins', data)], axis=1)
        data = pd.concat([data, addDummy('RX_SUMM_SURGICAL_MARGINS', [
            'Macroscopic residual tumor, Gross tumor of the primary site which is visible to the naked eye'],
                                           'Gross_Margins', data)], axis=1)
        data = pd.concat([data, addDummy('MTS_SIMPLE_PRIMARY', ['OC'], 'OC', data)], axis=1)
        data["RT_START"] = data["RT_START"] - data["SURG_START"]
        data["RT_START"] = pd.DataFrame(imp.fit_transform(data[["RT_START"]]), columns=["RT_START"])
        data["RT_Dose"] = pd.DataFrame(imp.fit_transform(data[["RT_Dose"]]), columns=["RT_Dose"])

        data = data[data.RT_Dose >= 5000]
        data = data[data.RT_START >= 0]
        data = data[data.MaxT > -1]
        data = data[data.MaxN > -1]
        data = data[data.Immunotherapy == 0]
        data = data[(data.ECE_micro == 0) & (data.ECE_macro == 0) & (data.Gross_Margins == 0) & (data.Micro_Margins == 0)]
        data = data[(data['RX_SUMM_CHEMO'] != 'Chemotherapy recommended, unknown if administered') & (data['RX_SUMM_CHEMO'] != 'Unknown if recommended or administered')]
        print("Oral cavity")
        print(data[data.OC == 1].describe().to_string())
        print("Oral Cavity Chemo")
        print(data[(data.OC == 1) & (data.Chemo == 1)].describe().to_string())
        print("Oral Cavity RT")
        print(data[(data.OC == 1) & (data.Chemo == 0)].describe().to_string())

        data.to_csv("checkMissing.csv")
        return










    # print(dataS.describe().to_string())
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import ExtraTreesRegressor
    print(dataS.describe().to_string())
    imp = IterativeImputer(random_state=0, verbose=2, estimator=ExtraTreesRegressor(n_estimators=100, random_state=0))
    imp.fit(dataS)



    if verbose:
        #assess accuracy of regressor
        td1, td2 = train_test_split(dataS, test_size=0.2, random_state=1)

        td2["HPV_True"] = td2["HPV"]
        td2["HPV_OP_True"] = np.nan
        td2.loc[(td2.OPX == 1), "HPV_OP_True"] = td2["HPV"]

        td2["HPV_OTHER_True"] =  np.nan
        td2.loc[(td2.OPX == 0), "HPV_OTHER_True"] = td2["HPV"]

        td2["HPV"] = np.nan
        imp2 = IterativeImputer(random_state=0, verbose=2, estimator=ExtraTreesRegressor(n_estimators=100, random_state=0))
        imp2.fit(td1)
        tempt = imp2.transform(td2[td1.columns])
        td3 = pd.DataFrame(tempt, columns=td1.columns)
        td3 = pd.concat([td3.reset_index(), td2[["HPV_True", "HPV_OP_True", "HPV_OTHER_True"]].reset_index()], axis = 1)
        from sklearn.metrics import roc_curve
        from matplotlib import pyplot
        from sklearn.metrics import roc_auc_score
        td3 = td3[(td3.HPV_True == 0) | (td3.HPV_True == 1)]
        ns_fpr, ns_tpr, _ = roc_curve(td3.HPV_True, td3.HPV)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.show()
        auc = roc_auc_score(td3.HPV_True, td3.HPV)
        print('AUC: %.3f' % auc)

    #dataS["HPV"].values[:] = np.nan
    #print(dataS.describe().to_string())
    #dataS[dataS.HPV == np.nan] = 0

    dataS = pd.DataFrame(imp.transform(dataS), columns=dataS.columns)
    # dataS.loc[dataS.HPV > 0.5, "HPV"] = 1
    # dataS.loc[dataS.HPV <= 0.5, "HPV"] = 0
    # print(dataS.describe().to_string())
    # dataS = pd.concat([dataS, addValidationHPV(data)], axis = 1)
    # print(dataS[dataS.HPV2 == 1].describe().to_string())
    # print(dataS[dataS.HPV2 == 0].describe().to_string())

    dataS["HPV_OP"] = 0
    dataS.loc[(dataS.OPX == 1), "HPV_OP"] = dataS["HPV"]

    dataS["HPV_OTHER"] = 0
    dataS.loc[(dataS.OPX == 0), "HPV_OTHER"] = dataS["HPV"]

    dataS["HPV_OP_True"] = np.nan
    dataS.loc[(dataS.OPX == 1), "HPV_OP_True"] = dataS["HPV"]

    dataS["HPV_OTHER_True"] =  np.nan
    dataS.loc[(dataS.OPX == 0), "HPV_OTHER_True"] = dataS["HPV"]

    # for i in range(100):
    #    dataS["HPV3"] = dataS["HPV"]*100 > i
    #    print(i)
    #    print(dataS[(dataS.HPV3 == 1) & (dataS.HPV2 == 1)]["HPV"].count())
    #    print(dataS[(dataS.HPV3 == 0) & (dataS.HPV2 == 0)]["HPV"].count())
    #    print(dataS[(dataS.HPV3 == 0) & (dataS.HPV2 == 0)]["HPV"].count() + dataS[(dataS.HPV3 == 1) & (dataS.HPV2 == 1)]["HPV"].count())


    dataS = pd.concat([dataS, addRow("DX_LASTCONTACT_DEATH_MONTHS", data)], axis=1)
    dataS['Dead'] = data.PUF_VITAL_STATUS == 'Dead'
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataS = pd.DataFrame(imp.fit_transform(dataS), columns=dataS.columns)
    dataS = pd.concat([dataS, data[["MTS_SIMPLE_PRIMARY", "TNM_PATH_T", "TNM_PATH_N"]]], axis=1)

    dataS = dataS[dataS.RT_Dose >= 5000]
    if verbose:
        print("RT Dose curative " + str(len(dataS.index)))

    dataS = dataS[dataS.DX_RAD_STARTED_DAYS >= 0]
    if verbose:
        print("RT treatment started >= 0 days: " + str(len(dataS.index)))
    dataS = dataS[dataS.MaxT > -1]
    dataS = dataS[dataS.MaxN > -1]
    if verbose:
        print("Either clinical or pathologic T and N stages recorded: " + str(len(dataS.index)))

    dataS = dataS[dataS.Immunotherapy == 0]
    if verbose:
        print("Did not receive immunotherapy: " + str(len(dataS.index)))

    dataS = dataS[dataS['Unknown_chemo'] == 0]
    if verbose:
        print("Unknown if chemotherapy administered: " + str(len(dataS.index)))

    dataTS = dataS[(dataS.ECE_micro == 0) & (dataS.ECE_macro == 0) & (dataS.Gross_Margins == 0) & (dataS.Micro_Margins == 0)]
    if verbose:
        print("Margin negative / ECE negative: " + str(len(dataTS.index)))



    dataTS = dataTS[dataTS.Chemo == 1]
    if verbose:
        print("Received Chemotherapy: " + str(len(dataTS.index)))

    dataS = dataS[dataS.HPV != np.nan]
    dataS.to_csv(filename + ".csv")
    dataS = dataS[
        (dataS.Micro_Margins == 0) & (dataS.Gross_Margins == 0) & (dataS.ECE_micro == 0) & (dataS.ECE_macro == 0)]
    if verbose:
        print(dataS.describe().to_string())
        print("Received Chemo")
        print(dataS[dataS.Chemo == 1].describe().to_string())
        print("Received RT")
        print(dataS[dataS.Chemo == 0].describe().to_string())
        print("HPV OP +")
        print(dataS[dataS.HPV_OP == 1].describe().to_string())
        print("HPV OP -")
        print(dataS[dataS.HPV_OP == 0].describe().to_string())
        print("HPV Other +")
        print(dataS[dataS.HPV_OTHER == 1].describe().to_string())
        print("HPV Other -")
        print(dataS[dataS.HPV_OTHER == 0].describe().to_string())
        print("Age > 70")
        print(dataS[dataS.AGE > 70].describe().to_string())
        print("Age < 70")
        print(dataS[dataS.AGE <= 70].describe().to_string())
        print("OPX")
        print(dataS[dataS.OPX == 1].describe().to_string())
        print(dataS[(dataS.OPX == 1) & (dataS.Chemo == 1)].describe().to_string())
        print(dataS[(dataS.OPX == 1) & (dataS.Chemo == 0)].describe().to_string())
        print("Non-OPX")
        print(dataS[dataS.OPX == 0].describe().to_string())
        print(dataS[(dataS.OPX == 0) & (dataS.Chemo == 1)].describe().to_string())
        print(dataS[(dataS.OPX == 0) & (dataS.Chemo == 0)].describe().to_string())


#------------------------------------------------------------------------
#
#  END USER FUNCTIONS FOR TESTING AND REPLICATION
#
#------------------------------------------------------------------------

#Run to find hyperparameters for an RSF model
def hyperparameters_RSF():
    pd.set_option('mode.chained_assignment', None)
    includeArray3 = ["Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "LN1", "LN2_4", "LN5_9", "LN10"]
    dataS = pd.read_csv('limited.csv')
    findRSFLRSingle2CV(dataS, includeArray3, CI = False, testVar = "Chemo", RS = 0)

#Run to find hyperparameters for a DeepSurv model
def hyperparameters_DS():
    pd.set_option('mode.chained_assignment', None)
    includeArray3 = ["Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "LN1", "LN2_4", "LN5_9", "LN10"]
    dataS = pd.read_csv('limited.csv')
    findNNCPHLRSingle2(dataS, includeArray3, CI = False, testVar = "Chemo", RS = 0)

#Run to find hyperparameters for an N-MLTR model
def hyperparameters_NMLTR():
    pd.set_option('mode.chained_assignment', None)
    includeArray3 = ["Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "LN1", "LN2_4", "LN5_9", "LN10"]
    dataS = pd.read_csv('limited.csv')
    findNNMTMLRSingle2(dataS, includeArray3, CI = False, testVar = "Chemo", RS = 0)

#Train 3 model types using optimal hyperparameters
def trainModels():
    RS = 0
    pd.set_option('mode.chained_assignment', None)
    includeArray3 = ["Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "LN1", "LN2_4", "LN5_9", "LN10"]
    hpNNCPH = {
        'structure': [{'activation': 'LeCunTanh', 'num_units': 64}, {'activation': 'Sinc', 'num_units': 128}, {'activation': 'LogSigmoid', 'num_units': 32}, {'activation': 'InverseSqrt', 'num_units': 128}, {'activation': 'Swish', 'num_units': 16}],
        'lr': 1e-05,
        'num_epochs': 6000,
        'dropout': 0.1,
        'l2_reg': 0.0001,
        'batch_normalization': False
    }
    hpNNMTLR = {
        'structure': [{'activation': 'Sinc', 'num_units': 32}, {'activation': 'Sigmoid', 'num_units': 32}, {'activation': 'LogLog', 'num_units': 8}, {'activation': 'Atan', 'num_units': 8}],
        'lr': 0.0001,
        'num_epochs': 4000,
        'dropout': 0.1,
        'l2_reg': 0.001,
        'batch_normalization': False,
        'l2_smooth': 0.001,
        'bins': 100
    }
    hpRSF = {
        'num_trees': 80,
        'max_features': 0.1,
        'max_depth': 40,
        'min_node_size': 80,
        'sample_size_pct': 0.6,
        'importance_mode': 'permutation'
    }
    dataS = pd.read_csv('limited.csv')
    tdW = getDataset(dataS, includeArray3, testVar="Chemo", validation=False, RS=0, HPV=0, age=0)
    nncph = getModel(False, "Chemo", modelType="NNCPH", td=tdW, hp=hpNNCPH)
    nnmtm = getModel(False, "Chemo", modelType="NNMTLR", td=tdW, hp=hpNNMTLR)
    rsf = getModel(False, "Chemo", modelType="RSF", td=tdW, hp=hpRSF)

def testModels():
    RS = 0
    pd.set_option('mode.chained_assignment', None)
    includeArray3 = ["Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "LN1", "LN2_4", "LN5_9", "LN10"]
    includeArray3DD = ["MTS_LN_POS", "N0", "N2_3", "T1", "T2", "OC_OPX", "Chemo", "Multiagent", "T3", "T4", "N1", "N2", "N3", 'I', 'II', 'III', 'IV', 'V',
                     'Retropharyngeal', 'Parapharyngeal', 'Larynx', 'HPX', 'Y2009_2012', 'Y2013_2016', 'RT_Dose', 'Tongue', 'FOM', 'RMT', 'BucMuc', 'AlvRid', 'BOT', 'Ton',
                     "CS_SITESPECIFIC_FACTOR_11", "TUMOR_SIZE", "MTS_LN_DIS_CAT", "LIFE", "Black", "Male", "Academic",
                     "Low_Comorbidity", "LVI", "HPV_OP",
                     "HPV_OTHER", "Well_Differentiated", "Moderately_Differentiated", "Poorly_Differentiated",
                     "MTS_TX_PKG_TIME", "ECE_micro", "ECE_macro","Micro_Margins", "Gross_Margins", "LN1", "LN2_4", "LN5_9", "LN10"]
    dataS = pd.read_csv('limited.csv')
    dataS = dataS[(dataS.Micro_Margins == 0) & (dataS.Gross_Margins == 0) & (dataS.ECE_micro == 0) & (dataS.ECE_macro == 0)].reset_index(drop=True)
    hpNNCPH = {
        'structure': [{'activation': 'LeCunTanh', 'num_units': 64}, {'activation': 'Sinc', 'num_units': 128}, {'activation': 'LogSigmoid', 'num_units': 32}, {'activation': 'InverseSqrt', 'num_units': 128}, {'activation': 'Swish', 'num_units': 16}],
        'lr': 1e-05,
        'num_epochs': 6000,
        'dropout': 0.1,
        'l2_reg': 0.0001,
        'batch_normalization': False
    }
    hpNNMTLR = {
        'structure': [{'activation': 'Sinc', 'num_units': 32}, {'activation': 'Sigmoid', 'num_units': 32}, {'activation': 'LogLog', 'num_units': 8}, {'activation': 'Atan', 'num_units': 8}],
        'lr': 0.0001,
        'num_epochs': 4000,
        'dropout': 0.1,
        'l2_reg': 0.001,
        'batch_normalization': False,
        'l2_smooth': 0.001,
        'bins': 100
    }
    hpRSF = {
        'num_trees': 80,
        'max_features': 0.1,
        'max_depth': 40,
        'min_node_size': 80,
        'sample_size_pct': 0.6,
        'importance_mode': 'permutation'
    }


    tdW = getDataset(dataS, includeArray3, testVar="Chemo", validation=False, RS=0, HPV=0, age=0)
    vdWB = getDataset(dataS, includeArray3DD, testVar="Chemo", validation=True, RS=0, HPV=0, age=0)
    vdWx = vdWB['x']
    vdWxPM = vdWB['xPM']
    vdW = {
    'x': vdWB['x'][includeArray3],
    't': vdWB['t'],
    'e': vdWB['e'],
    'xPM': vdWB['xPM'][includeArray3 + ['scores']]
    }

    vdWa1 = getDataset(dataS, includeArray3, testVar="Chemo", validation=True, RS=0, HPV=0, age=1)
    vdWa2 = getDataset(dataS, includeArray3, testVar="Chemo", validation=True, RS=0, HPV=0, age=2)
    vdWh1 = getDataset(dataS, includeArray3, testVar="Chemo", validation=True, RS=0, HPV=1, age=0)
    vdWh2 = getDataset(dataS, includeArray3, testVar="Chemo", validation=True, RS=0, HPV=2, age=0)

    plotSurvivalChemo(dataS[includeArray3], dataS['DX_LASTCONTACT_DEATH_MONTHS'], dataS['Dead'])
    checkStandard(vdWB, includeArray3DD, RS, 0, 0, "Chemo")

    modelT = "NNCPH"
    nncph = getModel(True, "Chemo", modelType=modelT, td=tdW, hp=hpNNCPH)
    treatmentDifferencesNN(nncph, dataS, columns=includeArray3, testVar="Chemo", ratio = 0)
    rec1, _, _, _ = checkModel(nncph, vdW, testVar="Chemo", modelType=modelT, IPTW=True, appendString="", ratio = 0)
    checkModel(nncph, vdWa1, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", Age > 70", ratio = 0)
    checkModel(nncph, vdWa2, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", Age <= 70", ratio = 0)
    checkModel(nncph, vdWh1, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", HPV+", ratio = 0)
    checkModel(nncph, vdWh2, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", HPV-", ratio = 0)

    modelT = "NNMTLR"
    nnmtm = getModel(True, "Chemo", modelType=modelT, td=tdW, hp=hpNNMTLR)
    treatmentDifferencesNN(nnmtm, dataS, columns=includeArray3, testVar="Chemo", ratio = 0)
    rec2, _, _, _ = checkModel(nnmtm, vdW, testVar="Chemo", modelType=modelT, IPTW=True, appendString="", ratio = 0)
    checkModel(nnmtm, vdWa1, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", Age > 70", ratio = 0)
    checkModel(nnmtm, vdWa2, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", Age <= 70", ratio = 0)
    checkModel(nnmtm, vdWh1, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", HPV+", ratio = 0)
    checkModel(nnmtm, vdWh2, testVar = "Chemo", modelType=modelT, IPTW=False, appendString = ", HPV-", ratio = 0)

    modelT = "RSF"
    rsf = getModel(True, "Chemo", modelType=modelT, td=tdW, hp=hpRSF)

    rec3, _, _, _ = checkModel(rsf, vdW, testVar="Chemo", modelType=modelT, IPTW=True, appendString="")
    checkModel(rsf, vdWa1, testVar="Chemo", modelType=modelT, IPTW=False, appendString=", Age > 70")
    checkModel(rsf, vdWa2, testVar="Chemo", modelType=modelT, IPTW=False, appendString=", Age <= 70")
    checkModel(rsf, vdWh1, testVar="Chemo", modelType=modelT, IPTW=False, appendString=", HPV+")
    checkModel(rsf, vdWh2, testVar="Chemo", modelType=modelT, IPTW=False, appendString=", HPV-")
    a = treatmentDifferencesSF(rsf, dataS, columns=includeArray3)

    plotSurvivalArray([rec1, rec2, rec3], vdW['x'], vdW['t'], vdW['e'], ["DeepSurv", "N-MTLR", "RSF"], 0)
    variableImportance(nncph, vdW['x'], vdW['t'], vdW['e'])
    variableImportance(nnmtm, vdW['x'], vdW['t'], vdW['e'])
    variableImportanceChunk(rsf, vdW['x'], vdW['t'], vdW['e'])


    from pysurvival.utils.metrics import bootstrap_concordance_index
    bootstrap_concordance_index(nncph, vdW['x'], vdW['t'], vdW['e'], n_size=4000)
    bootstrap_concordance_index(nnmtm, vdW['x'], vdW['t'], vdW['e'], n_size=4000)
    from pysurvival.utils.metrics import bootstrap_concordance_index_chunk
    bootstrap_concordance_index_chunk(rsf, vdW['x'], vdW['t'], vdW['e'], n_size=4000)


testModels()