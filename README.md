

# Case Study : Comparing Classifiers

[Link to notebook:]  AIML-Portfolio-Comparing-Classifiers/prompt_III.ipynb at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/prompt_III.ipynb) 



## Context

The goal of this project is to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines). The dataset used is related to the marketing of bank products over the telephone and comes from the [UCI Machine Learning repository Links to an external site.](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. 



# 1 Business Understanding



## 1.1 Background



The increasingly vast number of marketing campaigns over time has reduced its effect on the general public. Furthermore, economical pressures and competition has led marketing managers to invest on directed campaigns with a strict and rigorous selection of contacts. Such direct campaigns can be enhanced through the use of Business Intelligence (BI) and Data Mining (DM) techniques.

There are two main approaches for enterprises to promote products and/or services: through mass campaigns, targeting general indiscriminate public or directed marketing, targeting a specific set of contacts (Ling and Li 1998). Nowadays, in a global competitive world, positive responses to mass campaigns are typically very low, less than 1%, according to the same study. Alternatively, directed marketing focus on targets that assumable will be keener to that specific product/service, making this kind of campaigns more attractive due to its efficiency (Ou et al. 2003). Nevertheless, directed marketing has some drawbacks, for instance it may trigger a negative attitude towards banks due to the intrusion of privacy (Page and Luding 2003).

It should be stressed that due to internal competition and current financial crisis, there are huge pressures for European banks to increase a financial asset. To solve this issue, one adopted strategy is offer attractive long-term deposit applications with good interest rates, in particular by using directed marketing campaigns. Also, the same drivers are pressing for a reduction in costs and time. Thus, there is a need for an improvement in efficiency: lesser contacts should be done, but an approximately number of successes (clients subscribing the deposit) should be kept.

##### [Reference: **USING DATA MINING FOR BANK DIRECT MARKETING:** **AN APPLICATION OF THE CRISP-DM METHODOLOGY**; <u>provided by Universidade do Minho: RepositoriUM</u>]



## 1.2 Business Goals and KPI

The business objective is to find the best model that can explain success of a client subscribes to the deposit. This model will help in increasing the campaign efficiency by identifying the main characteristics that affect success, helping improve management of the available resources (e.g. human effort, phone calls, time) and selection of a high quality and affordable set of potential buying customers.



# 2 Data Understanding

This section provides information about the data, its description and is its exploration to make sure it fits the business goals.



## 2.1 **Gathering and Describing Data**

This data comes from [UCI Machine Learning repository Links to an external site.](https://archive.ics.uci.edu/ml/datasets/bank+marketing). Here is a sample of the data:



![original.png](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/original.png?raw=true)



Following features of the provided sample data above:

```
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)
```





The description of the data is as follows:

```
Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
   3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
   5 - default: has credit in default? (categorical: "no","yes","unknown")
   6 - housing: has housing loan? (categorical: "no","yes","unknown")
   7 - loan: has personal loan? (categorical: "no","yes","unknown")
   # related with the last contact of the current campaign:
   8 - contact: contact communication type (categorical: "cellular","telephone") 
   9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
  11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
   # other attributes:
  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  14 - previous: number of contacts performed before this campaign and for this client (numeric)
  15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
   # social and economic context attributes
  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
  20 - nr.employed: number of employees - quarterly indicator (numeric)

  Output variable (desired target):
  21 - y - has the client subscribed a term deposit? (binary: "yes","no")
```



Number of categories per 'object' feature is as follows:

```
Column Name                  # of Categories
______________              _________________
job																12
marital														4
education													8
default														3
housing														3
loan															3
contact														2
month															10
day_of_week												5
poutcome													3
y																	2
```



## 1.2 Early Data **Exploration** and Data Quality Check

Next step is to check the quality of the data. For example, since many of the column/variable is categorical, the summary of the data is checked for the types in each category. By doing this, the step needed for data cleaning or to be transformed is identified. For example, checking for missing/empty values.

Following is the summary statistics (mean, median, min, max, etc.) of the data:

```
               age      duration      campaign         pdays      previous  \
count  41176.00000  41176.000000  41176.000000  41176.000000  41176.000000   
mean      40.02380    258.315815      2.567879    962.464810      0.173013   
std       10.42068    259.305321      2.770318    186.937102      0.494964   
min       17.00000      0.000000      1.000000      0.000000      0.000000   
25%       32.00000    102.000000      1.000000    999.000000      0.000000   
50%       38.00000    180.000000      2.000000    999.000000      0.000000   
75%       47.00000    319.000000      3.000000    999.000000      0.000000   
max       98.00000   4918.000000     56.000000    999.000000      7.000000   

       emp.var.rate  cons.price.idx  cons.conf.idx     euribor3m   nr.employed  
count  41176.000000    41176.000000   41176.000000  41176.000000  41176.000000  
mean       0.081922       93.575720     -40.502863      3.621293   5167.034870  
std        1.570883        0.578839       4.627860      1.734437     72.251364  
min       -3.400000       92.201000     -50.800000      0.634000   4963.600000  
25%       -1.800000       93.075000     -42.700000      1.344000   5099.100000  
50%        1.100000       93.749000     -41.800000      4.857000   5191.000000  
75%        1.400000       93.994000     -36.400000      4.961000   5228.100000  
max        1.400000       94.767000     -26.900000      5.045000   5228.100000  
```



Following is the category of each column that is of type object:

```
job            [housemaid, services, admin., blue-collar, tec...
marital                     [married, single, divorced, unknown]
education      [basic.4y, high.school, basic.6y, basic.9y, pr...
default                                       [no, unknown, yes]
housing                                       [no, yes, unknown]
loan                                          [no, yes, unknown]
contact                                    [telephone, cellular]
month          [may, jun, jul, aug, oct, nov, dec, mar, apr, ...
day_of_week                            [mon, tue, wed, thu, fri]
poutcome                         [nonexistent, failure, success]
y                                                      [no, yes]
dtype: object
```



Missing data in percentage:

```
age               0.0
job               0.0
marital           0.0
education         0.0
default           0.0
housing           0.0
loan              0.0
contact           0.0
month             0.0
day_of_week       0.0
duration          0.0
campaign          0.0
pdays             0.0
previous          0.0
poutcome          0.0
emp.var.rate      0.0
cons.price.idx    0.0
cons.conf.idx     0.0
euribor3m         0.0
nr.employed       0.0
y                 0.0

```



Number of duplicate data:

```
Shape before treating duplicates: (41188, 21)
Duplicates : 12
```





# 2 Data Preparation

This section provides information on data preparation and cleaning, to allow for analysis as part of this case study and the future case studies covering predictions.  



## 2.1 Data Transformation

Following categorical features were transformed:

**a) 'y' -** 

'yes' was converted to a value of 1 and 'no' was converted to a value of 0.

**b) 'month' -** 

Months were converted to their corresponding integer values.

**b) Remaining -** 

For the rest of the catagorical data, Hot-Encoding method was used.



## 2.2 Data Cleansing

In this step, the data is handled based on the problem found during the data understanding phase. Based on the finding, the following steps are executed:

a) Duplicate data was found and removed.

b) There were no missing values. 

c) There were no NULL values found.



## 2.2 Final Data (post cleaning and transformation)



```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41176 entries, 0 to 41175
Data columns (total 55 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   age                            41176 non-null  int64  
 1   month                          41176 non-null  int64  
 2   duration                       41176 non-null  int64  
 3   campaign                       41176 non-null  int64  
 4   pdays                          41176 non-null  int64  
 5   previous                       41176 non-null  int64  
 6   emp.var.rate                   41176 non-null  float64
 7   cons.price.idx                 41176 non-null  float64
 8   cons.conf.idx                  41176 non-null  float64
 9   euribor3m                      41176 non-null  float64
 10  nr.employed                    41176 non-null  float64
 11  y                              41176 non-null  int64  
 12  job_admin.                     41176 non-null  float64
 13  job_blue-collar                41176 non-null  float64
 14  job_entrepreneur               41176 non-null  float64
 15  job_housemaid                  41176 non-null  float64
 16  job_management                 41176 non-null  float64
 17  job_retired                    41176 non-null  float64
 18  job_self-employed              41176 non-null  float64
 19  job_services                   41176 non-null  float64
 20  job_student                    41176 non-null  float64
 21  job_technician                 41176 non-null  float64
 22  job_unemployed                 41176 non-null  float64
 23  job_unknown                    41176 non-null  float64
 24  marital_divorced               41176 non-null  float64
 25  marital_married                41176 non-null  float64
 26  marital_single                 41176 non-null  float64
 27  marital_unknown                41176 non-null  float64
 28  education_basic.4y             41176 non-null  float64
 29  education_basic.6y             41176 non-null  float64
 30  education_basic.9y             41176 non-null  float64
 31  education_high.school          41176 non-null  float64
 32  education_illiterate           41176 non-null  float64
 33  education_professional.course  41176 non-null  float64
 34  education_university.degree    41176 non-null  float64
 35  education_unknown              41176 non-null  float64
 36  default_no                     41176 non-null  float64
 37  default_unknown                41176 non-null  float64
 38  default_yes                    41176 non-null  float64
 39  housing_no                     41176 non-null  float64
 40  housing_unknown                41176 non-null  float64
 41  housing_yes                    41176 non-null  float64
 42  loan_no                        41176 non-null  float64
 43  loan_unknown                   41176 non-null  float64
 44  loan_yes                       41176 non-null  float64
 45  contact_cellular               41176 non-null  float64
 46  contact_telephone              41176 non-null  float64
 47  day_of_week_fri                41176 non-null  float64
 48  day_of_week_mon                41176 non-null  float64
 49  day_of_week_thu                41176 non-null  float64
 50  day_of_week_tue                41176 non-null  float64
 51  day_of_week_wed                41176 non-null  float64
 52  poutcome_failure               41176 non-null  float64
 53  poutcome_nonexistent           41176 non-null  float64
 54  poutcome_success               41176 non-null  float64
dtypes: float64(48), int64(7)

```





# 3 Data Understanding - Deep Analysis

This section provides information about deeper exploration and analysis of the data conducted, in preparation for future machine learning model.



## 3.1 Exploratory Data Analysis (EDA)

In this section, the results of exploring and visualizing insight from the data is captured.



### 3.1.1 Categorical Data 

Figure 1 provide the categories and the frequency in percentage per feature. 
![AIML-Portfolio-Comparing-Classifiers/images/bar_categories.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/bar_categories.png) 
**Figure 1 - Categories and the frequency in percentage per feature**



In Figure 1, the disbribution of the  campaign population across various feature categories is presented, providing a hit as to which feature and associated categories may influence the 'deposit subscription' outcome as 'yes' or 'no'. Figure 2 provides the numerical feature distribution and a view of the outliers per feature. 



 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_age_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_age_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_campaign_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_campaign_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_cons.conf.idx_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_cons.conf.idx_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_cons.price.idx_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_cons.price.idx_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_duration_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_duration_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_emp.var.rate_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_emp.var.rate_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_euribor3m_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_euribor3m_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_nr.employed_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_nr.employed_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_pdays_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_pdays_dist.png) 

 ![AIML-Portfolio-Comparing-Classifiers/images/box_hist_previous_dist.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/box_hist_previous_dist.png) 

**Figure 2 - Numerical feature distribution and a view of the outliers per feature**



In Figure 2, the disbribution of the  campaign population across various features is presented.



Figure 3 provides the heatmap of the numerical features.



 ![AIML-Portfolio-Comparing-Classifiers/images/heatmap_numeric_features.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/heatmap_numeric_features.png) 

**Figure 3 - Heatmap of numeric features & 'y'**



From the heatmap, it is clear the following features influence the 'y' outcome:

a) 'duration' - last contact duration, in seconds. Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no').

b) amp.var.rate' - employment variation rate - quarterly indicator.

c) 'cons.price.idx' - consumer price index - monthly indicator.

d) 'nr.employed' - number of employees - quarterly indicator.

e) 'euribor3m' - euribor 3 month rate - daily indicator.

f) 'pdays' -  number of days that passed by after the client was last contacted from a previous campaign.





### 3.1.2 Addressing Business Questions

Following were some of the business questions answered based on the analysis of the data.

**1) What is the average acceptance rate across age?**

Figure 4 provides the average acceptance across age of the population. 



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_age.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_age.png) 

**Figure 4 - Avg acceptance across age**



The acceptance of age group above 60 is higher relative to the age group between 25 and 60.



**2) What is the average acceptance across campaign?**

Figure 5 provides the average acceptance across campaigns. 



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_campaign.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_campaign.png) 

**Figure 5 - Avg acceptance across campaign**



The likelihood of acceptance with fewer campains is much higher then compare to running it more than 5 time. 



**3) What is the average acceptance across jobs?**

Figure 6 provides the average acceptance across various jobs. 



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_job.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_job.png) 

**Figure 6 - Avg acceptance across job**



Population that are retired and are students have a higher acceptance rate.





### 4.1.3 Observation  

Initial analysis of the data showed that the chances of acceptance is high when it comes to the population with the following characteristics: 

a) age is over 60 and mostly retired

b) age below 24 and are students.

Figure 12 and 13 provide the maximum and minimum car prices based on individual categorical features. 



### 5.0   Modeling  

The goal of the modeling phase is to identify the best model to be used for identify the key features that drive the car prices. This phase will have four tasks:

**a) Select modeling technique**: Determine which algorithms to try (e.g. Linear Regression,
Ridge).

**b) Generate test design**: Pending the modeling approach, split the data into training, test, and validation sets.
**c) Build model**: Build and executing the selected models.
**d) Assess model**: Interpret the model results based on the pre-defined success criteria, and the test design.



### 5.1   Model Selection 

The modeling technique is as follows:

### 5.1.1   Pre-process 

The pre-process is defined to transform the categorical features to numerical values,  retaining the original information. As a result, "Ordinal Encoder" was executed against the 'month' feature since the categorical values had inherent order to it.

For the rest of the features, "OneHotEncoder" was used because the categorical values did not have inherent order or ranking.

### 5.1.2   Modeling Technique 

The models that will be evaluated are:

-  Logistic Regression
- KNN
- Decision Tree
- SVM

Following steps were taken to deterime the best model using default hyper parameters:

a) Baseline model accuracy and performance. This provided the baseline performance the selected model must meet.

b) Evaluate the above 4 identified classification models using default hyperparameters. 

The model that performs the best will be selected to identify the important features that contribute to driving the subscription rate higher. 



The next step was to determine the optimal hyperparameter, that has the best accuracy. The 'GridSearchCV' was used to identify the hyperparameter for each model. Finally, the models accuracy and performance were re-evaluated using the identified hyperparameter. And the confusion matrix was used extract insights.  Note: becasue the data was imbalanced, the F1-sore was used.

### 5.1.3   Testing Technique 

The pre-processed data will be split, of which 70% will be randomly allocated for taining and 30% will be allocated for testing.

### 5.1.4   Model assessment using default hyperparameter

The overall best performing model was Logistic Regression. Following is the comparison between the 4 models:



### 5.1.4.1   Summary of Assessment Results 

The dataset was split as follows:

```
X_train shape = (28823, 54)
 X_test shape = (12353, 54)
y_train shape = (28823,)
 y_test shape = (12353,)
```

Following is the comparison between the 4 models:

```
                 Model  Train Time (in msec)  Train Accuracy  Test Accuracy
0  Logistic Regression               82.8979        0.911911       0.907229
1                  KNN               12.9340        0.918607       0.890877
2        Decision Tree              134.1100        1.000000       0.877601
3                  SVM             5607.9931        0.924054       0.904962
```

From Table 1, Logistic Regression model has the best performance and therefore, it was selected to extract the important features that were selected.

 ![AIML-Portfolio-Comparing-Classifiers/images/lr_coeff.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/lr_coeff.png)  

**Figure 7 - Important features selected by Logestic Regression'**

Using the important features identified in Figure 7 and the correlation matrix, feaure engineering and exploration was conducted.



### 5.1.4.2   Feature engineering and exploration 

 The following features will be intestigated further: 

- non-categorical features : 'age', 'campaign', 'month', 'previous' 
- categorical features: 'job', 'education', 'marital', 'housing', 'loan', 'day_of_week', 'default'

Lets first examine the non-categorical features:

- a) 'age' - This feature will be dropped since (per correlation matrix) the correlation with 'y' is very low.
- b) 'campaign' - This feature will be dropped since (per correlation matrix) the correlation with 'y' is very low.
- c) 'month' - This feature will be dropped per the low ranking of the feature by the Logistic Regression model.
- d) 'previous' - This feature will be dropped since (per correlation matrix) the correlation with 'y' is very low (<0.3)

For the categorical features, decision of eliminating the features was based on the relationship to 'y' in the plots to follow.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_job.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_job.png)  

**Figure 8 - Avg acceptance across job'**

- Per the above plot 'job' feature was kept since 'job_retired' and 'job_admin.' is selected as top 1/3 tier important features by Logestic Regression model.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_education.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_education.png)  

**Figure 9 - Avg acceptance across education'**

- This feature was dropped because the 'unknown' was identified as important feature by the Logestic Regression model. This does not provide insight towards the business objective.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_marital.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_marital.png)  

**Figure 10 - Avg acceptance across marital'**

- This feature was dropped because the 'unknown' was identified as important feature by the Logestic Regression model. This does not provide insight towards the business objective.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_housing.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_housing.png)  

**Figure 11 - Avg acceptance across housing'**

- This feature will be dropped because the distribution across the 'housing' categories are almost the same. The total accepatance is ~30%.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_loan.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_loan.png)  

**Figure 12 - Avg acceptance across loan'**

- This feature will be dropped because the distribution across the 'loan' categories are almost the same. The total accepatance is ~30%.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_day_of_week.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_day_of_week.png)  

**Figure 13 - Avg acceptance across day_of_week'**

- This feature will be dropped because the distribution across the 'day_of_week' categories are almost the same. The total accepatance is ~55%.



 ![AIML-Portfolio-Comparing-Classifiers/images/accept_default.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/accept_default.png)  

**Figure 14 - Avg acceptance across default'**

- This feature will be dropped becasue the know status of default is little over ~12% then therefore not relevent to the business objective.



### 5.2.   Refining Data

Per the analysis steps in section 5.1.4.2, the identified features were dropped and the data was split into train/test - 70%/30%:

```
X_train shape = (28823, 24)
 X_test shape = (12353, 24)
y_train shape = (28823,)
 y_test shape = (12353,)
```



Dataframe features (info):

```
RangeIndex: 41176 entries, 0 to 41175
Data columns (total 25 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   duration              41176 non-null  int64  
 1   pdays                 41176 non-null  int64  
 2   emp.var.rate          41176 non-null  float64
 3   cons.price.idx        41176 non-null  float64
 4   cons.conf.idx         41176 non-null  float64
 5   euribor3m             41176 non-null  float64
 6   nr.employed           41176 non-null  float64
 7   y                     41176 non-null  int64  
 8   job_admin.            41176 non-null  float64
 9   job_blue-collar       41176 non-null  float64
 10  job_entrepreneur      41176 non-null  float64
 11  job_housemaid         41176 non-null  float64
 12  job_management        41176 non-null  float64
 13  job_retired           41176 non-null  float64
 14  job_self-employed     41176 non-null  float64
 15  job_services          41176 non-null  float64
 16  job_student           41176 non-null  float64
 17  job_technician        41176 non-null  float64
 18  job_unemployed        41176 non-null  float64
 19  job_unknown           41176 non-null  float64
 20  contact_cellular      41176 non-null  float64
 21  contact_telephone     41176 non-null  float64
 22  poutcome_failure      41176 non-null  float64
 23  poutcome_nonexistent  41176 non-null  float64
 24  poutcome_success      41176 non-null  float64
dtypes: float64(22), int64(3)
```





### 5.3.   Model Hyperparameter Tuning & Performance

### 5.3.1   Logistic Regression

Using the GridSearchCV, the following optimal hyperparameter was obtained:

```
{'C': 0.0001, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l1', 'random_state': None, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
```



Using the above hyper parameter, following performace results were produced:

```
The accuracy of the Logistic Regression model using best hyper parameters for training data is 0.89
The accuracy of the Logistic Regression model using best hyper parameters for test data is 0.88
Logistic Regression model training execution time: 21.405 msec
```



The following figure provides the important features identified by the Logestic Regression.

 ![AIML-Portfolio-Comparing-Classifiers/images/lr_best_hyper_imp_features.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/lr_best_hyper_imp_features.png)  

**Figure 15 - Important Features'** 



### 5.3.2   KNN

Using the GridSearchCV, the following optimal hyperparameter was obtained:

```
{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 15, 'p': 2, 'weights': 'uniform'}
```



Using the above hyper parameter, following performace results were produced:

```
The accuracy of the KNeighborsClassifier model using best hyper parameters for training data is  0.92
The accuracy of the KNeighborsClassifier model using best hyper parameters for test data is 0.90
KNeighborsClassifier model training execution time: 7.7181 msec
```



### 5.3.3   Decision Tree

Using the GridSearchCV, the following optimal hyperparameter was obtained:

```
{'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'random_state': 42, 'splitter': 'best'}
```



Using the above hyper parameter, following performace results were produced:

```
The accuracy of the DecisionTreeClassifier model using best hyper parameters for training data is  0.92
The accuracy of the DecisionTreeClassifier model using best hyper parameters for test data is 0.91
DecisionTreeClassifier model training execution time: 37.5171 msec
```



### 5.3.4   SVM

Using the GridSearchCV, the following optimal hyperparameter was obtained:

```
{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
{'C': 100, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.01, 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
```



Using the above hyper parameter, following performace results were produced:

```
The accuracy of the SVC model using best hyper parameters for training data is  0.92
The accuracy of the SVC model using best hyper parameters for test data is 0.90
SVC model training execution time: 5143.4753 msec
```





### 5.3.4   Model Comparison Using Optimal Hyperparameter

The table below provides a comparison between the models. 

```
                 Model  Train Time (in msec)  Train Accuracy  Test Accuracy
0  Logistic Regression               21.4050        0.888457       0.884724
1                  KNN                7.7181        0.916803       0.903424
2        Decision Tree               37.5171        0.917774       0.911277
3                  SVM             5143.4753        0.917323       0.903910
```

Decision Tree performance was the best.



### 5.3.4   Confusion Matrix, ROC, and AUC

The following figure provides the confusion matrix and ROC curve of Decision Tree model.

 ![AIML-Portfolio-Comparing-Classifiers/images/test_dt_confusion_matrix.png at main · bhaswarey/AIML-Portfolio-Comparing-Classifiers](https://github.com/bhaswarey/AIML-Portfolio-Comparing-Classifiers/blob/main/images/test_dt_confusion_matrix.png)  

**Figure 15 - Decision Tree - Confusion Matrix & ROC'** 



```
              precision    recall  f1-score   support

           0       0.93      0.97      0.95     10929
           1       0.66      0.48      0.56      1424

    accuracy                           0.91     12353
   macro avg       0.80      0.73      0.75     12353
weighted avg       0.90      0.91      0.91     12353
```





### 6.0   Deployment



### 6.0.1   Summary

Taking a directed campaign approach by using the identified best classification model to help narrow the target population to high quality and affordable set of potential buying customers, we must be careful of (risk):

- creating a negative impacted on the targeted population as a side effect of the targeted personilized campaign
- reducing the success of a client subscribes to the deposit actieved thus far.
- increase waistage of resources (e.g. human effort, phone calls, time)

Becasue we are dealing with imbalanced data we will look at F1-score which is 56% the model will classify subscriber as yes. We can also look at the AUC (which is 92%) to determine how well the model will perform separating between subscribe - yes or no.



### 6.0.2   Recommendation

Further analysis need to be execute to improve the f1-score to address the above identified risks.



### 7.0   Next Steps

The next step would be to cycle through the data preparation to relook at the categorical features to help improve the F1-score.
