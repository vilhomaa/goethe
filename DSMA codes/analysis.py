import pandas as pd
import statsmodels.api as sm
import miceforest as mf
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor, plot_tree,export_graphviz
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np
import graphviz
import seaborn as sn
import datetime
from tabulate import tabulate


# RUN THE CLEAN-UP SCRIPT BEFORE THIS SCRIPT



############################## PART 1 ANALYSIS ##############################
## Analysis, cross-sectional data


data_1 = pd.read_csv('data/data_part1.csv',index_col=False,na_values=['(NA)'])


# Creation of dummy columns for categorical variables

data_1 = pd.concat((data_1, pd.get_dummies(data_1["continent"])), axis=1)



data_1['HMI'] = data_1.HMI.replace({'x' : None}).astype(float)
data_1['RestaurantsPriceRange2'] = data_1.RestaurantsPriceRange2.replace({'None' : None}).astype(float)
data_1['breakfast'] = data_1.breakfast.astype(bool)
data_1['open_after_22'] = data_1.open_after_22.astype(bool)

# Taking the logarithms of checkin_idx_yearly_avg and photos due to outliers
data_1["log_checkin_idx_yearly_avg"] = np.log(data_1["checkin_idx_yearly_avg"])
data_1["log_photos"] = np.log(data_1["photos"])


data_1.replace([np.inf, -np.inf], np.nan, inplace=True)
# data_1.dropna(subset=["log_checkin_idx_yearly_avg", "log_photos"], how="all")


# Variables to exclude from original dataset
exclude_variables = [
    'business_id',
    'total_checkin_idx',
    'city',
    'continent',
    'state',
    'postal_code',
    'checkin_idx_yearly_avg',
    'photos',
    'latitude',
    'longitude'
]
# analysis_data will be used for regressions etc, data_1 will be used for visualizations
analysis_data= data_1.drop(columns = exclude_variables)



# Handling missings
# replace with mean? mice?
data_amp = mf.ampute_data(
    analysis_data,
    perc=0.25,
    random_state=2021,
)

kds = mf.KernelDataSet(data_amp, save_all_iterations=True, random_state=2021)

# Run the MICE algorithm for 3 iterations
kds.mice(3)

analysis_data = kds.complete_data()




############################## Descriptive analysis ##############################


# Graphs


# log photos x log checkin_idx
plt.figure(figsize=(9, 5))
plt.scatter(
    x=analysis_data["log_photos"],
    y=analysis_data["log_checkin_idx_yearly_avg"],
    s=0.5,
)
plt.ylabel("log_checkin_idx_yearly_avg", size=15)
plt.xlabel("log_photos", size=15)
plt.savefig("graphs/check_ins_x_photos.png")
plt.close()


# HMI x log_checkin_idx_yearly_avg 
analysis_data = analysis_data.sort_values('HMI')
plt.figure(figsize=(9, 5))
x_vals, y_vals = analysis_data["HMI"],analysis_data["log_checkin_idx_yearly_avg"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
p = np.polyfit(x_vals, y_vals, 3)
plt.plot(x_vals, np.polyval(p,x_vals), label="OLS best fitted line of 3rd order", color="orange")
plt.ylabel("log_checkin_idx_yearly_avg", size=15)
plt.xlabel("HMI", size=15)
plt.legend(loc="best")
plt.savefig("graphs/HMI_x_checking_idx_avg.png")
plt.close()



# population density x log_checkin_idx_yearly_avg 
analysis_data = analysis_data.sort_values('population_density')
plt.figure(figsize=(9, 5))
x_vals, y_vals = analysis_data["population_density"],analysis_data["log_checkin_idx_yearly_avg"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
p = np.polyfit(x_vals, y_vals, 3)
plt.plot(x_vals, np.polyval(p,x_vals), label="OLS best fitted line of 3rd order", color="orange")
plt.ylabel("log_checkin_idx_yearly_avg", size=15)
plt.xlabel("population_density", size=15)
plt.legend(loc="best")
plt.savefig("graphs/pop_dens_x_checking_idx_avg.png")
plt.close()


# HMI x log_checkin_idx_yearly_avg
analysis_data = analysis_data.sort_values('HMI')
plt.figure(figsize=(9, 5))
x_vals, y_vals = analysis_data["HMI"],analysis_data["log_checkin_idx_yearly_avg"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
p = np.polyfit(x_vals, y_vals, 3)
plt.plot(x_vals, np.polyval(p,x_vals), label="OLS best fitted line of 3rd order", color="orange")
plt.ylabel("log_checkin_idx_yearly_avg", size=15)
plt.xlabel("HMI", size=15)
plt.legend(loc="best")
plt.savefig("graphs/HMI_x_checking_idx_avg.png")
plt.close()


# HMI x population_density
analysis_data = analysis_data.sort_values('HMI')
plt.figure(figsize=(9, 5))
x_vals, y_vals = analysis_data["HMI"],analysis_data["population_density"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
p = np.polyfit(x_vals, y_vals, 1)
plt.plot(x_vals, np.polyval(p,x_vals), label="OLS best fitted line of 1st order", color="orange")
plt.ylabel("population_density", size=15)
plt.xlabel("HMI", size=15)
plt.legend(loc="best")
plt.savefig("graphs/HMI_x_population_density.png")
plt.close()

# Map visualization -> tableau



# Descriptive statistics


with open("graphs/descriptive_statistics_xsect.txt", "w") as fn:
    fn.write(
        analysis_data
        .describe( include='all')
        .round(decimals=2)
        .T.to_string()
    )



# regression


Y = analysis_data['log_checkin_idx_yearly_avg']
X = analysis_data.drop(columns = ['log_checkin_idx_yearly_avg','others'])# Makes the continent "others" as the baseline
X_2 = analysis_data.drop(columns = ['log_checkin_idx_yearly_avg','others','HMI','population_density'])# Makes the continent "others" as the baseline

result = sm.OLS(endog=Y, exog=X, missing="drop").fit()

xcols = list(X.columns)
with open("graphs/reg_xsect.txt", "w") as fh:
    fh.write(result.summary(xname=xcols, yname='log_checkin_idx_yearly_avg').as_text())

result = sm.OLS(endog=Y, exog=X_2, missing="drop").fit()

xcols = list(X_2.columns)
with open("graphs/reg_xsect_2.txt", "w") as fh:
    fh.write(result.summary(xname=xcols, yname='log_checkin_idx_yearly_avg').as_text())



# decision tree regressor -> trying to predict the score
# -> visualized as well

clf = DecisionTreeRegressor(
    max_depth = 3,
    random_state=2021
    )
model = clf.fit(X, Y)

fig = plt.figure(figsize=(22,15))
_ = plot_tree(clf, 
            max_depth = 3,
            feature_names=xcols,  
            class_names='log_checkin_idx_yearly_avg',
            filled=True)
fig.savefig("graphs/reg_tree_pt1.png")


# Categorical tree -> trying to predict the continent of the food

# reconstructing "continent" variable from dummies


analysis_data_for_class_tree = analysis_data.copy()
analysis_data_for_class_tree['continent'] = analysis_data_for_class_tree[['African','Asian', 'European', 'Latin', 'Middle Eastern', 'North American','others']].idxmax(axis=1)

analysis_data_for_class_tree.drop(columns = ['African','Asian', 'European', 'Latin', 'Middle Eastern', 'North American','others'],inplace=True)
xcols = analysis_data_for_class_tree.columns

Y = analysis_data_for_class_tree['continent']
X = analysis_data_for_class_tree.drop(columns = ['log_checkin_idx_yearly_avg','continent'])# Makes the continent "others" as the baseline
xcols = X.columns

clf = DecisionTreeClassifier(
    max_depth = 7,
    random_state=2021,
    min_samples_leaf=100
    )
model = clf.fit(X, Y)


dot_data = export_graphviz(clf, out_file=None, 
                                feature_names=xcols,  
                                class_names=sorted(analysis_data.continent.unique()),
                                filled=True)

graph = graphviz.Source(dot_data, format="png") 

graph.render("graphs/class_tree_1")
############################## Predictive analysis ##############################
# Constructing the attractiveness index
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score

pred_y_value = 'log_checkin_idx_yearly_avg'
pred_x_variables = list(analysis_data.columns)
pred_x_variables.remove('log_checkin_idx_yearly_avg')
pred_x_variables_wo_demographic_variables = [item for item in pred_x_variables if item != 'HMI' and item != 'population_density']


X_train, X_test, y_train, y_test = train_test_split(
    analysis_data[pred_x_variables], analysis_data[pred_y_value], test_size=0.25, random_state=2021)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    analysis_data[pred_x_variables_wo_demographic_variables], analysis_data[pred_y_value], test_size=0.25, random_state=2021)


gbm = GradientBoostingRegressor(
    loss="ls",
    criterion="friedman_mse",
    learning_rate=0.1,
    n_estimators=1000,
    n_iter_no_change = 10,
    min_samples_leaf=30,
    max_depth=3,
    random_state=2021,
    verbose=1,
)

# Train the gradient boosting model
gbm.fit(X_train, y_train)

# Compute the prediction over the training and testing sets
y_pred_test = gbm.predict(X_test)

# Compute the MSE for the training and testing sets
error_test_gbm = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(error_test_gbm)

analysis_data['attribute_idx_with_hmi_and_popdens'] = gbm.predict(analysis_data[pred_x_variables])

# generating attribute idx without hmi and population density
gbm.fit(X_train2, y_train2)
y_pred_test = gbm.predict(X_test2)
error_test_gbm2 = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(error_test_gbm2)

analysis_data['attribute_idx_wo_hmi_and_popdens'] = gbm.predict(analysis_data[pred_x_variables_wo_demographic_variables])



# Corr matx

ticklabels = ['checkin years','Is open','pricing','parking','reservations','Tableservice','GoodForGroups','WiFi','Alcohol','OutdoorSeating',
'Wheelchair','Weekdays open','Weekends open','Breakfast','Open after 22','RestaurantsInArea','Canada','Chain','HMI','PopDensity','African',
'Asian','European','Latin','MiddleEastern','NorthAmerican','Others','YVariable','Log photos','PredYwHMI','PredYwoHMI']

plt.figure(figsize=(9,7))
corrMatrix = analysis_data.corr()
sn.set(font_scale=0.8)
sn.heatmap(corrMatrix, annot=False,linewidths=.5,yticklabels=ticklabels,xticklabels=ticklabels)
plt.savefig('graphs/corr_heatmap.png')
plt.close()

# Adding business_id's back so they can be used for joining again
analysis_data['business_id'] = data_1['business_id'] # adding these so they can be joined





##################### Part 2 #####################

data_2 = pd.read_csv('data/data_part2.csv',index_col=False,na_values=['(NA)'])

business_info_to_part2_analysis = analysis_data[
    ['business_id',
    'attribute_idx_wo_hmi_and_popdens',
    'attribute_idx_with_hmi_and_popdens']
]

analysis_data2 = pd.merge(data_2,business_info_to_part2_analysis,on = 'business_id',how = 'inner')

analysis_data2['date'] = pd.to_datetime(analysis_data2['date'],format="%Y-%m-%d")
# We will only use dates from 2010 onwards
analysis_data2 = analysis_data2[analysis_data2['date']>datetime.datetime(2010,1,1)]

# Creating daily, monthly and yearly dummies
# day name
day_names = pd.get_dummies(analysis_data2['date'].dt.day_name())
day_names = day_names.drop(columns=['Monday']) # setting monday as the baseline

# months
months = pd.get_dummies(analysis_data2['date'].dt.month_name())
months = months.drop(columns=['January']) # setting January as the baseline

# year
years = pd.get_dummies(analysis_data2.date.dt.year, prefix='year')
years = years.drop(columns=['year_2010']) # setting January as the baseline


timedummy_df = pd.concat((years, months, day_names), axis=1)
analysis_data2 = pd.concat((analysis_data2, timedummy_df), axis=1)
analysis_data2 = analysis_data2.drop(columns = ['Unnamed: 0'])


# reconstructing check-in true/false values from the cumulative sum
# Would be better to have this as a part of the cleaning script -> didn't fix it due to time limitations
analysis_data2['check_in'] = analysis_data2.groupby('business_id')['cumcount_checkins'].diff(1)
analysis_data2['check_in'] = analysis_data2['check_in'].fillna(0)
analysis_data2['check_in'] = np.where(analysis_data2['check_in']<0,0,analysis_data2['check_in'])
analysis_data2['check_in'] = np.where(analysis_data2['check_in']>1,1,analysis_data2['check_in'])


############ Descriptive Statistic of pt2 data ############


with open("graphs/descriptive_statistics_ts.txt", "w") as fn:
    fn.write(
        analysis_data2.drop(columns = ['date','business_id'])
        .describe( include='all')
        .round(decimals=2)
        .T.to_string()
    )



Y1 = analysis_data2['check_in']
X1 = analysis_data2.drop(columns = ['attribute_idx_with_hmi_and_popdens','business_id','date','check_in'])
X2 = analysis_data2.drop(columns = ['attribute_idx_wo_hmi_and_popdens','business_id','date','check_in'])


result1 = sm.Logit(endog=Y1, exog=X1, missing="drop").fit()
result2 = sm.Logit(endog=Y1, exog=X2, missing="drop").fit()


xcols = list(X1.columns)
with open("graphs/reg_ts1.txt", "w") as fh:
    fh.write(result1.summary(xname=xcols, yname='check_in').as_text())

xcols = list(X2.columns)
with open("graphs/reg_ts2.txt", "w") as fh:
    fh.write(result2.summary(xname=xcols, yname='check_in').as_text())


# Calculate AUC's
auc_wo = roc_auc_score(Y1, result1.predict(X1))
auc_with = roc_auc_score(Y1, result2.predict(X2))



# Table for evaluation metrics

table = [['Analysis,metric','Without HMI & pop.dens', 'With HMI & pop.dens'],
    ['1, RSME', error_test_gbm2,error_test_gbm],
    ['2, AUC',auc_wo,auc_with]
]
table = tabulate(table, headers='firstrow', tablefmt='fancy_grid')
with open("graphs/perf_table.txt", "w") as fh:
    fh.write(table)



# Other graphs


analysis_data = analysis_data.sort_values('HMI')
plt.figure(figsize=(9, 5))
x_vals, y_vals = np.log(analysis_data["HMI"]*analysis_data["population_density"]*analysis_data["restaurants_in_same_zip"]),analysis_data["log_checkin_idx_yearly_avg"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
plt.ylabel("Log of Check-ins compared to median", size=15)
plt.xlabel("log of HMI * Pop. Density * RestaurantsInZip", size=15)
plt.legend(loc="best")
plt.savefig("graphs/HMI_et_population_density_x_checkins.png")
plt.close()


analysis_data = analysis_data.sort_values('HMI')
plt.figure(figsize=(9, 5))
x_vals, y_vals = analysis_data["restaurants_in_same_zip"],analysis_data["log_checkin_idx_yearly_avg"]
plt.scatter(
    x=x_vals,
    y=y_vals,
    s=0.5,
)
p = np.polyfit(x_vals, y_vals, 1)
plt.plot(x_vals, np.polyval(p,x_vals), label="OLS best fitted line of 1st order", color="orange")
plt.ylabel("Log of Check-ins compared to median", size=15)
plt.xlabel("Restaurants in same zip", size=15)
plt.legend(loc="best")
plt.savefig("graphs/restaurantsInZip_x_checkins.png")
plt.close()



## Disclaimer, this code was not reviewed, and has a lot of possibilities for improvement
# eg. all the reoccuring code could have been replaced with functions
