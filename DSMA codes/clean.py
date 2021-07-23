import pandas as pd
import json
import re
import numpy as np
from collections import Counter
import os
from datetime import datetime
import itertools



## Cleaning and data preparation script
# Can take up to 4 hours to complete

# Prerequisities:
# Following full unedited csv tables  from Yelp with values in JSON format in a column in stored in to folder "data"
# - Business
# - Checkin
# - Review
# THE TOTAL DATA OF THE REVIEW TABLE IS NEAR 5 GB ->
#   Cannot be downloaded from the SQL server as a single csv
#   -> Solution: download the data in chunks of 500k rows, include them all
#               in a folder "reviews" within the data folder. Then by looping read all of them
# - Photos
# - Tips 
# External Data sources:

# Canada population data
# Source: https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/hlt-fst/pd-pl/comprehensive.cfm
# - stored as "T301EN.csv"

# Joining canada's income data
# Source: https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/dt-td/Rp-eng.cfm?TABID=4&LANG=E&A=R&APATH=3&DETAIL=0&DIM=0&FL=A&FREE=0&GC=01&GL=-1&GID=1159582&GK=1&GRP=1&O=D&PID=110192&PRID=10&PTYPE=109445&S=0&SHOWALL=0&SUB=999&Temporal=2016&THEME=119&VID=0&VNAMEE=&VNAMEF=&D1=0&D2=0&D3=0&D4=0&D5=0&D6=0 
# - stored as "98-400-X2016099_English_CSV_data.csv"

# US data 
# source:  https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2018-zip-code-data-soi
# - stored as "18zpallagi.csv"



# Part 1.: preparing data for analyses 1
# - Sources:
#   - All cross-sectional data
#   - Population density data
#   - Household median income data

# Part 2.: preparing data for analysis 2
# - Sources
#   - All data with dates



##  Part 1:
#  Business table


# Loading data
print('Loading business table')
json_df = pd.read_csv('data/businesstable.csv') #public
business_df = json_df.join(json_df['j'].apply(json.loads).apply(pd.Series))
business_df = business_df.drop(columns = ['j'])


# Defining food into categories according to continent
print('Cleaning business table')
nationalities_and_continents = pd.read_csv('data/nationalities.csv')
nationalities = [i for i in nationalities_and_continents['Nationality']]
nationalities_continents_tuple_list = list(zip(nationalities_and_continents['Nationality'], nationalities_and_continents['Continent']))


# filtering the data to only include restaurants
categories = [category.strip() for category in business_df.categories.unique() if  type(category) == str and len(category) > 1]
categories = [string for string in categories if 'Restaurants' in string]

# taking all rows that refer to restaurants
business_df = business_df[business_df['categories'].notna()]
business_df_restaurants = business_df[business_df['categories'].str.contains("Restaurants")]




# Functions for extracting ethnicity of a restaurant and interpreting the continent of said restaurant
def return_first_ethnicity(string):
    for word in string.strip().split(","):
        for nationality in nationalities:
            if nationality in word:
                # Some entries are "American (New)" -> strip away the parenthesis
                return word.split('(')[0].strip()
    return 'others'

def return_continent_of_ethnicity(ethnicity):
    for nationality,continent in nationalities_continents_tuple_list:
        if nationality in ethnicity:
            return continent
    return 'others'

business_df_restaurants['ethnicity'] = [return_first_ethnicity(category) for category in business_df_restaurants['categories']]
business_df_restaurants['continent'] = [return_continent_of_ethnicity(ethnicity) for ethnicity in business_df_restaurants['ethnicity']]


# Generating columns from the 'attributes' column

cols_from_attributes = [
    'RestaurantsPriceRange2', 
    'BusinessParking',
    'RestaurantsReservations',
    'RestaurantsTableService',
    'RestaurantsGoodForGroups',
    'WiFi',
    'Alcohol',
    'OutdoorSeating',
    'WheelchairAccessible'
]
for col in cols_from_attributes:
    business_df_restaurants[col] = [row[col] if type(row) is dict and col in row.keys() else None for row in business_df_restaurants['attributes']]


# Parking column is yet another dict -> lets fix 
# here BusinessParking col values are as strings instead of dicts -> lets change them to dict
business_df_restaurants['BusinessParking'] = [eval(row) if type(row) is str else None for row in business_df_restaurants['BusinessParking']]

# This code returns true if any of the keys valet, street or lot are true
business_df_restaurants['parking'] = [True if (type(row) is dict and ('street' in row.keys() and 'lot' in row.keys())) and (row['street'] == True or (row['lot'] == True or row['valet'] == True)) else False for row in business_df_restaurants['BusinessParking']]


business_df_restaurants['WiFi'] = [False if 'no' in str(row) or row is None else True for row in business_df_restaurants['WiFi']]
business_df_restaurants['Alcohol'] = [False if 'no' in str(row) or row is None else True for row in business_df_restaurants['Alcohol']]


# Lastly, for these columns, we force them to be False if the value is not true. i.e. none's are put to be false
for col in cols_from_attributes[2:]:
    business_df_restaurants[col] = business_df_restaurants[col].astype('bool')
    business_df_restaurants[col] = [True if row is True else False for row in business_df_restaurants[col]]


# Generating columns around the opening hours


def opening_day_interpreter(hour_dict,days):
    try:
        i = 0
        if days == 'weekend':
            for day in hour_dict.keys():
                if 'Sunday' in day or 'Saturday' in day:
                    i+=1
            return i
        else:
            for day in hour_dict.keys():
                if 'Sunday' not in day and 'Saturday' not in day:
                    i+=1
            return i           
    except:
        return None

def opening_hour_interpreter(hours_dict,time):
    try:
        if time == 'late':
            for key in hours_dict:
                closing_hour = int(hours_dict[key].split('-')[1].split(':')[0])
                if closing_hour > 22 or closing_hour < 5:
                    return True
            return False
        else:
            for key in hours_dict:
                opening_hour = int(hours_dict[key].split('-')[0].split(':')[0])
                if opening_hour < 10:
                    return True
            return False
    except:
        return None          

business_df_restaurants['days_open_on_weekends'] = [opening_day_interpreter(row,'weekend') for row in business_df_restaurants['hours']]
business_df_restaurants['days_open_on_weekdays'] = [opening_day_interpreter(row,'weekday') for row in business_df_restaurants['hours']]

business_df_restaurants['breakfast'] = [opening_hour_interpreter(row,'early') for row in business_df_restaurants['hours']]
business_df_restaurants['open_after_22'] = [opening_hour_interpreter(row,'late') for row in business_df_restaurants['hours']]


# Generating a column that counts all the other restaurants in the same zip

business_df_restaurants['restaurants_in_same_zip'] = business_df_restaurants.groupby('postal_code')['name'].transform(len)

# Dummy for indicating a chain restaurant
chain_restaurants = [restaurant_name for restaurant_name in business_df_restaurants['name']]
# We identify chain restaurants to be restaurants if there is 5 or more restaurants with the same name
restaurants_counted = Counter(chain_restaurants)
business_df_restaurants['chain'] = business_df_restaurants['name'].apply(lambda x: True if restaurants_counted[x] > 4 else False)



# Generating a dummycolumn to indicate that the restaurant is located in canada
def postal_code_canadian(code):
    try:
        int(code)
        return False
    except:
        return True


business_df_restaurants['canada'] = business_df_restaurants['postal_code'].apply(lambda x : postal_code_canadian(x))

# Choosing what columns to keep

business_columns_to_keep = [
    'business_id',
    'city',
    'state',
    'postal_code',
    'is_open',
    'RestaurantsPriceRange2', 
    'parking',
    'RestaurantsReservations',
    'RestaurantsTableService',
    'RestaurantsGoodForGroups',
    'WiFi',
    'Alcohol',
    'OutdoorSeating',
    'WheelchairAccessible',
    'continent',
    'days_open_on_weekends',
    'days_open_on_weekdays',
    'breakfast',
    'open_after_22',
    'restaurants_in_same_zip',
    'canada',
    'chain',
    'latitude',
    'longitude'
]

# Take all columns above
business_df_final = business_df_restaurants[business_columns_to_keep]




print('importing photo data')
json_df = pd.read_csv('data/photo.csv') #public
photo_df = json_df.join(json_df['j'].apply(json.loads).apply(pd.Series))


photo_df_grouped = photo_df.groupby('business_id').agg({'label' : 'count'}).rename(columns = {'label':'photos'})


final_df = pd.merge(business_df_final,photo_df_grouped,how = 'left',on = 'business_id')
final_df['photos'] = final_df['photos'].fillna(0)


## Join check-in data

print('importing checkin data')
json_df = pd.read_csv('data/checkin.csv') #public
checkin_df = json_df.join(json_df['j'].apply(json.loads).apply(pd.Series))
checkin_df = checkin_df.drop(columns = ['j'])

final_df_copy = pd.merge(final_df,checkin_df,how = 'left',on='business_id')

years = [f'checkins_{year}' for year in range(2010,2019)]

for year in years:
    final_df_copy[year] = final_df_copy['date'].str.count(year.split('_')[1])

final_df_copy_copy = final_df_copy[years].div(final_df_copy[years].sum(axis=0), axis=1)
final_df_copy = pd.concat([final_df_copy['business_id'], final_df_copy_copy], axis=1)

# Normalize amount of checkins to measure the percentage of all checkins made that year
# -> index counts the checkins/year -> index tells the restaurants yearly average checkins compared to the average 
final_df_copy['total_checkin_idx'] = final_df_copy.loc[:, years].sum(axis=1)
final_df_copy['years_of_checkins'] = final_df_copy[years].replace({'0':np.nan, 0:np.nan}).count(axis=1)
final_df_copy['checkin_idx_yearly_avg'] = final_df_copy['total_checkin_idx']  / final_df_copy['years_of_checkins']
final_df_copy['checkin_idx_yearly_avg'] = final_df_copy["checkin_idx_yearly_avg"]/final_df_copy["checkin_idx_yearly_avg"].median()
final_df_copy = final_df_copy[['business_id','total_checkin_idx','years_of_checkins','checkin_idx_yearly_avg']]

final_df = pd.merge(final_df_copy,final_df,on = 'business_id')


#### POPULATION & INCOME DATA JOINING ####

# Canada population data
# Source: https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/hlt-fst/pd-pl/comprehensive.cfm

print('Adding canada population data')
canada_pop = pd.read_csv('data/T301EN.CSV',encoding = 'ISO 8859-16')
# interpreting the state based on the province code
canada_province_codes = pd.read_csv('data/canada_codes.csv',index_col=False)
canada_pop = pd.merge(canada_pop,canada_province_codes[['Province','province_code']],how='left',left_on='Province / territory, english',right_on = 'Province')
canada_pop = canada_pop[canada_pop['Population density per square kilometre, 2016'].notna()]


canada_pop = canada_pop.rename(columns={
    'Geographic name, english': 'city',
    'province_code': 'state',
    'Population, 2016': 'can_population',
    'Population density per square kilometre, 2016' : 'can_population_density'
})

# Some duplicates exist, -> too lazy to figure out why. I will keep the one with a higher population
# (most of the duplicate values were such that the other one had values of 0)
canada_pop = canada_pop.sort_values(['city','can_population'],ascending = False)
canada_pop = canada_pop.drop_duplicates(['city','state']) 

canada_pop = canada_pop[[
    'city',
    'state',
    'can_population_density'
    ]] 






assert len(final_df) == len(pd.merge(final_df,canada_pop,how='left',on=['city','state'])), "DUPLICATES GENERATED IN JOINING CANADA'S POPULATION DATA"
final_df = pd.merge(final_df,canada_pop,how='left',on=['city','state'])

# sanity check with deleted code indicated that:
# 165 out of 23489 location-geo_code combinations in final df not found in canada_pop
# --> not too many missing  values



# Joining canada's income data
# Source: https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/dt-td/Rp-eng.cfm?TABID=4&LANG=E&A=R&APATH=3&DETAIL=0&DIM=0&FL=A&FREE=0&GC=01&GL=-1&GID=1159582&GK=1&GRP=1&O=D&PID=110192&PRID=10&PTYPE=109445&S=0&SHOWALL=0&SUB=999&Temporal=2016&THEME=119&VID=0&VNAMEE=&VNAMEF=&D1=0&D2=0&D3=0&D4=0&D5=0&D6=0 

print('Adding canada income data')
canada_income = pd.read_csv('data/98-400-X2016099_English_CSV_data.csv')
# Only total income statistics
canada_income = canada_income[canada_income['DIM: Household type including census family structure (11)'] == 'Total - Household type including census family structure']
# Only areas, not the stats for the whole of canada
canada_income = canada_income[canada_income['GEO_LEVEL'] == 3 ]

# interpreting the province code from geo_code
canada_income['GEO_CODE (POR)'] = canada_income['GEO_CODE (POR)'].astype(str).str[:2].astype(int)
canada_income = pd.merge(canada_income,canada_province_codes[['geo_code','province_code']],how = 'left', left_on = 'GEO_CODE (POR)', right_on = 'geo_code')

# the geo names are in format xyxyxy , X -> we wanna delete the letter after the comma
canada_income['GEO_NAME'] = canada_income['GEO_NAME'].str.split(',').str[0]

canada_income = canada_income.rename(columns={
    'Dim: Household income statistics (3): Member ID: [2]: Median total income of households ($)':'can_HMI',
    'GEO_NAME' : 'city',
    'province_code': 'state'
})

# Some duplicates city & state combinations in data -> from the data we cannot surely know what are in the yelp-data
# the area corresponds to because of the differences in canada's statistical and postal areas. 
#  - Solution: We choose the are with more households.
canada_income.sort_values(['city','Dim: Household income statistics (3): Member ID: [1]: Total - Household income statistics (Note: 3)'],ascending = False).to_csv('data/test.csv')
canada_income = canada_income.drop_duplicates(['city','state'])


canada_income = canada_income[
    [
        'city',
        'state',
        'can_HMI',
    ]
]



# -> selvitä mitkä duplikaatit ottaa mukaan ja mistä päästä eroon....
# CSD_TYPE_NAME = NO ?? poista??mikä toi csd ees on?

assert len(final_df) == len(pd.merge(final_df,canada_income,how='left',on=['city','state'])), "DUPLICATES GENERATED IN JOINING CANADA'S INCOME DATA"
final_df = pd.merge(final_df,canada_income,how='left',on=['city','state'])
# sanity check with deleted code indicated that:
# 289 out of 23489 location-geo_code combinations in final df not found in canada_income
# -->> suitable for merging, not too much data is missing


# US POPULATION DATA -> JOINING POPULATION DENSITY
# source: https://simplemaps.com/data/us-zips 
print('Adding US population data')

us_pop = pd.read_csv(
    'data/uszips.csv',
    usecols = ['zip','density']
)
us_pop['zip'] = us_pop['zip'].astype(str)
us_pop.columns = ['postal_code','us_population_density']

assert len(us_pop) == len(us_pop.postal_code.unique()), 'DUPLICATE ROWS IN US POPULATION DATA'
final_df = pd.merge(final_df,us_pop,how='left',on = 'postal_code')

# US data -> count median income from classifications in the irs data -> count cumsum of people
# and then calculate what the median income would be.
# source:  https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2018-zip-code-data-soi

print('Adding US income data')

us_income = pd.read_csv(
    'data/18zpallagi.csv',
    usecols = ['zipcode', 'agi_stub','N1'])


# Calculating a cumulative sum for the amount of households in-group
# generating columns for easier calculations -> will be deleted later
us_income['N1_cumsum'] = us_income.groupby('zipcode')['N1'].transform(pd.Series.cumsum)
us_income['N1_cumsum_L'] = us_income.groupby('zipcode')['N1_cumsum'].shift(1).fillna(0)
us_income['median_hh'] = us_income.groupby('zipcode')['N1'].transform(pd.Series.sum)/2

# Calculates the percentage of the income groups income where the median household would be located
# e.g. Median hh income is 2000. Income group 1 ends at the 1800th household, and income grp
# ends at the 2500th household. Then the median HH is at 200/700 % on the interval between grp 1 and 2
us_income['median_hh_income_grp_ratio'] = [1 + (median_hh-N1_cumsum_L)/(N1_cumsum-N1_cumsum_L) if median_hh < N1_cumsum and median_hh >= N1_cumsum_L else 0 for median_hh,N1_cumsum,N1_cumsum_L in zip(us_income['median_hh'],us_income['N1_cumsum'],us_income['N1_cumsum_L'])]

# 1st number: The lowest amount of earnings for income group x
# 2nd number: The size of the interval between income group x and x+1
income_levels_and_intervals = { 
    1:[0,25000],
    2:[25000,25000],
    3:[50000,25000],
    4:[75000,25000],
    5:[100000,100000],
    6:[200000,200000] # The number for the size of interval is a guess, no number was provided in the data's documentation
}

# Calculates the MHI with the help of the dict above 
us_income['median_hh_income'] = [income_levels_and_intervals[agi_stub][0] + (median_hh_income_grp_ratio - 1)*income_levels_and_intervals[agi_stub][1] if median_hh_income_grp_ratio != 0 else 0 for median_hh_income_grp_ratio,agi_stub in zip(us_income['median_hh_income_grp_ratio'],us_income['agi_stub'])]


us_income_to_join = us_income[['zipcode' , 'median_hh_income']]
us_income_to_join = us_income_to_join[us_income_to_join['median_hh_income'] != 0] 


assert len(us_income.zipcode.unique()) == len(us_income_to_join.zipcode.unique()), "Rows have disappeared - some HMI's were not calculated"
us_income_to_join.columns = ['postal_code', 'usa_HMI']
us_income_to_join['postal_code'] = us_income_to_join['postal_code'].astype(str)

final_df = pd.merge(final_df, us_income_to_join, how='left' , on = 'postal_code')


# Merging all population and HMI columns 
final_df['HMI'] = final_df['usa_HMI'].combine_first(final_df['can_HMI'])
final_df['population_density']  = final_df['can_population_density'].combine_first(final_df['us_population_density']) 

final_df = final_df.drop(columns = ['usa_HMI', 'can_HMI','can_population_density','us_population_density'])

final_df.to_csv('data/data_part1.csv',index=False)


print('Data for part 1 generated.')

# final_df = pd.read_csv('data/data_part1.csv')
business_ids = final_df['business_id']




############################## DATA PART 2 ##############################

def load_and_parse_csv_file_with_json(path):
    json_df = pd.read_csv(path) 
    parsed_df = json_df.join(json_df['j'].apply(json.loads).apply(pd.Series))
    return parsed_df.drop(columns = ['j'])

print('Starting to generate data for part 2 (might take 2 hours)')

############ REVIEW TABLE ############
# Cumulative star average ratings
# Cumulative counts of reviews 

print('Importing Review Table')

final_df_pt2 = pd.DataFrame()
master_review_df = pd.DataFrame()

# THE TOTAL DATA OF THE REVIEW TABLE IS NEAR 5 GB ->
#   Cannot be downloaded from the SQL server as a single csv
#   -> Solution: download the data in chunks of 500k rows, include them all
#               in a folder "reviews" within the data folder. Then by looping read all of them
for file in os.listdir('data/reviews'):
    print(f'getting file {file}')
    df = load_and_parse_csv_file_with_json(f'data/reviews/{file}')
    df = pd.merge(df,business_ids,on='business_id',how='inner')
    master_review_df = master_review_df.append(df)
    

master_review_df['date'] = master_review_df['date'].str.split(' ').str[0]
master_review_df['date'] = pd.to_datetime(master_review_df['date'],format="%Y-%m-%d")

master_review_df = master_review_df.drop_duplicates(subset=['review_id'])

master_review_df = master_review_df.sort_values(['business_id','date'])
master_review_df['tempcol'] = 1
master_review_df["cumcount_reviews"] = master_review_df.groupby("business_id").tempcol.transform(
    lambda x: x.cumsum().shift()
)
master_review_df['cumcount_reviews'] = master_review_df['cumcount_reviews'].fillna(0)

master_review_df['cum_stars_avg'] = master_review_df.groupby("business_id").stars.transform(
    lambda x: x.cumsum().shift()
)
master_review_df['cum_stars_avg'] = master_review_df['cum_stars_avg'] / master_review_df['cumcount_reviews'] 
master_review_df['cum_stars_avg'] = master_review_df['cum_stars_avg'].fillna(0)

# Selecting only relevant columns
master_review_df = master_review_df[['business_id','date','cum_stars_avg','cumcount_reviews']]



############ TIPS TABLE ############

print('Importing Tips Table')

tips_df = load_and_parse_csv_file_with_json('data/tip.csv')
tips_df = pd.merge(tips_df,business_ids,on = 'business_id',how = 'inner') 

tips_df['date'] = tips_df['date'].str.split(' ').str[0]
tips_df['date'] = pd.to_datetime(tips_df['date'],format="%Y-%m-%d")

tips_df['tempcol'] = 1
tips_df = tips_df.sort_values(['business_id','date'])
tips_df["cumcount_tips"] = tips_df.groupby("business_id").tempcol.transform(
    lambda x: x.cumsum().shift()
)
tips_df['cumcount_tips'] = tips_df['cumcount_tips'].fillna(0)


tips_df = tips_df[[
    'date',
    'business_id',
    'cumcount_tips'
]]


final_df_pt2 = pd.merge(master_review_df,tips_df,how = 'outer',on = ['date','business_id'])

final_df_pt2 = final_df_pt2.sort_values(['business_id','date'])

# outer join creates nulls -> fillin them with the latest value of every column
final_df_pt2 = final_df_pt2.fillna(method='ffill')
final_df_pt2.cumcount_tips = final_df_pt2.cumcount_tips.fillna(0) # ffill only fills forward, first observations missing



############ CHECK-IN TABLE ############
# -> need to convert the data to panel data

print('Generating Review data')

# This function stores all individual dates and business_id's to lists that can be then transformed to a df
def extract_date_business_id_pairs(date_list,business_id):
    return_dates = []
    return_business_ids = []
    for date in date_list.split(','):
        return_dates.append(date)
        return_business_ids.append(business_id)
    return return_dates,return_business_ids


storage_dict = {'date':[],'business_id':[]}

# Filtering unnecessary business_ids
checkin_df = pd.merge(checkin_df,business_ids,on = 'business_id',how = 'inner') 

for date_list,business_id in zip(checkin_df.date,checkin_df.business_id):
    date_buffer,bid_buffer = extract_date_business_id_pairs(date_list,business_id)
    storage_dict['date'].append(date_buffer)
    storage_dict['business_id'].append(bid_buffer)


# flattening the nested lists
storage_dict['date'] = list(itertools.chain.from_iterable(storage_dict['date']))
storage_dict['business_id'] = list(itertools.chain.from_iterable(storage_dict['business_id']))

checkin_panel_df = pd.DataFrame(storage_dict)

# Cumsums of check-ins

checkin_panel_df['date'] = checkin_panel_df['date'].str.strip().str.split(' ').str[0]
checkin_panel_df['date'] = pd.to_datetime(checkin_panel_df['date'],format="%Y-%m-%d")

checkin_panel_df['tempcol'] = 1

checkin_panel_df["cumcount_checkins"] = checkin_panel_df.groupby("business_id").tempcol.transform(
    lambda x: x.cumsum().shift()
)
checkin_panel_df['cumcount_checkins'] = checkin_panel_df['cumcount_checkins'].fillna(0)

checkin_panel_df = checkin_panel_df.groupby(['business_id','date']).agg({'cumcount_checkins': max}).reset_index(level = 'business_id')


final_df_pt2 = pd.merge(final_df_pt2,checkin_panel_df,how = 'outer',on = ['date','business_id'])
final_df_pt2 = final_df_pt2.sort_values(['business_id','date'])
final_df_pt2 = final_df_pt2.fillna(method='ffill')
final_df_pt2.cumcount_checkins = final_df_pt2.cumcount_checkins.fillna(0) # ffill only fills forward, first observations missing


print('saving ' + str(len(final_df_pt2)) + ' rows')
final_df_pt2.to_csv('data/data_part2.csv')
print('Part 2 Complete')



