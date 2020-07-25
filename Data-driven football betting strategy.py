#!/usr/bin/env python
# coding: utf-8

# <center><h1>Data-driven football betting strategy</h1></center>
# <hr>
# ### Author(s)
#
# - Dyian Lim (@Dyian Lim)
#
# - Li Jing Cheng (@Jing Cheng)

# # Executive Summary
#
# Football betting is one of the most popular gambling activity today and betting sites often offer a mulititude of bet types to punters. Bookmakers in turn generate revenue by factoring in an overround to the odds that are offered to punters. These odds are then updated live until the start of the match based on the betting amounts placed by punters in a manner that reduces the total exposure of bookmakers to one side of a bet. Considering this, coupled with the tendency for most gamblers to rely on intuition, we will attempt to create a system that places value bets when they are available.

# # Research Topic & Hypothesis
#
#
# The first topic of interest is to consider the relationship between the different match statistics and the number of goals scored or conceded. Some examples of the statistics we will attempt to collect include ball possession, pass accuracy (%), corners, fouls, shot accuracy(%), offsides, tackles won, headers won, etc. We will evaluate this relationship with the use of historical data available (e.g. matches played in English Premier League from the 2010/11 to 2017/18 season).
#
# In the above mentioned step, we utlizied data-cleaning techniques, visualizations, modelling and web scrapping for data collection. After identifying the most important predictors, we fitted these variables into a linear regression model that returned the expected goals scored or conceded by the home and away team.
#
# In the next step, we tested the robustness of the model with the match results of the 2018/19 season (recently ended) by utilizing the match odds for matches in 2018/19 and the match statistics that will be available to us on a rolling basis. We converted the match odds into implied odds for certain events happening (e.g. 2/1 offered on home win will mean a 33% chance of home win in the long run). After which, we used a Poisson distribution to translate the expected goals scored (from our earlier model) into percentages of different events occuring. If the percentages of our model exceeds the implied odds by a certain margin, we will place a bet and record the results of the bet.
#
# The final step involves evaluation the results of all our bets and whether we can generate a statistically significant profit.
#
# **Literature / Articles referred to:**
# - Computing predicted goals ([Link 1](https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/MD62MLXUMKMXZ6A8)) retrieved on 13 Jul 2019
# - Statistical modelling of football results ([Link 2](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/)) retrieved on 13 Jul 2019
#
#
# **Datasets used:**
# - Source 1 ([Link 3](https://www.premierleague.com/)) retrieved on 13 Jul 2019 (Web-scrapped)
# - Source 2 ([Link 4](https://www.football-data.co.uk/data.php)) retrieved on 13 Jul 2019 (CSV format)

# # Methodology

# In[1]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from selenium import webdriver
import re
import time
import csv
from sklearn import linear_model
from scipy.stats import poisson

# ## _Step 1: Crawling data from the web_

# In[ ]:

'''
my_url = 'https://www.premierleague.com/clubs?se=79'
uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()

# In[ ]:


clubs_soup = soup(page_html, 'html.parser')
clubs_link_div = clubs_soup.find('div', {'class': 'indexSection'})
clubs_link_containers = clubs_link_div.findAll('li')

clubs_link_list = []
i = 0
for link in clubs_link_containers:
    i += 1
    if i < 16: continue
    club_link = link.a['href']
    club_link = 'https://www.premierleague.com' + club_link
    club_link = re.sub('overview', 'stats', club_link)
    club_link = club_link + '?se='
    clubs_link_list.append(club_link)

# club_names = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton and Hove Albion', 'Burnley', 'Chelsea',
#               'Crystal Palace', 'Everton',
#               'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United', 'Norwich City',
#               'Sheffield United',
#               'Southampton', 'Tottenham Hotspur', 'Watford', 'West Ham United', 'Wolverhampton Wonderers']
club_names = ['Southampton', 'Tottenham Hotspur', 'Watford', 'West Ham United', 'Wolverhampton Wonderers']
clubs_link_dict = {}
i = 0

for name in club_names:
    clubs_link_dict[name] = clubs_link_list[i]
    i += 1
clubs_link_dict

# In[ ]:


years_dict = {'1992/93': '1', '1993/94': '2', '1994/95': '3', '1995/96': '4', '1996/97': '5',
              '1997/98': '6', '1998/99': '7', '1999/00': '8', '2000/01': '9', '2001/02': '10',
              '2002/03': '11', '2003/04': '12', '2004/05': '13', '2005/06': '14', '2006/07': '15',
              '2007/08': '16', '2008/09': '17', '2009/10': '18', '2010/11': '19', '2011/12': '20',
              '2012/13': '21', '2013/14': '22', '2014/15': '27', '2015/16': '42', '2016/17': '54',
              '2017/18': '79', '2018/19': '210'}

clubs_years_links_dict = {}

for club, link in clubs_link_dict.items():
    for year, id_ in years_dict.items():
        club_year_link = link + id_
        if club not in clubs_years_links_dict:
            clubs_years_links_dict[club] = {}
        clubs_years_links_dict[club][year] = club_year_link

clubs_years_links_dict

# In[ ]:


headers = ['Year',
           'Goals',
           'Goals per match',
           'Shots',
           'Shots on target',
           'Shooting accuracy %',
           'Penalties scored',
           'Big chances created',
           'Hit woodwork',
           'Passes',
           'Passes per match',
           'Pass accuracy %',
           'Crosses',
           'Cross accuracy %',
           'Clean sheets',
           'Goals conceded',
           'Goals conceded per match',
           'Saves',
           'Tackles',
           'Tackle success %',
           'Blocked shots',
           'Interceptions',
           'Clearances',
           'Headed Clearance',
           'Aerial Battles/Duels Won',
           'Errors leading to goal',
           'Own goals',
           'Yellow cards',
           'Red cards',
           'Fouls',
           'Offsides']

# In[ ]:


# clubs_stats = {}

for club in clubs_years_links_dict:
    clubs_stats = {}
    for year, link in clubs_years_links_dict[club].items():
        my_url = link
        driver = webdriver.Chrome()
        driver.get(my_url)

        time.sleep(10)

        clubs_soup = soup(driver.page_source, 'html.parser')
        #         year = links_and_years_dict_new[link]

        club_stat_containers = clubs_soup.find_all('div', {'class': 'normalStat'})

        for container in club_stat_containers:
            stat = container.text.strip()
            stat = re.sub('\s+', ' ', stat)
            stat = re.split('\s', stat)

            stat_title = ''

            for word in stat[:-1]:
                stat_title = stat_title + ' ' + word
            stat_title = stat_title.strip()
            stat_num = stat[-1]

            if year not in clubs_stats:
                clubs_stats[year] = {}
            clubs_stats[year][stat_title] = stat_num

        df = pd.DataFrame.from_dict(clubs_stats)
        df = df.transpose().reset_index()
        df.rename(columns={'index': 'Year'}, inplace=True)
        df = df[headers]
        # df['club'] = str(club)
        csv_name = club + '_stats_all_years.csv'
        df.to_csv(csv_name)

# ## _Step 2: Reading data_

# In[2]:

'''
path = r'./Data/EPL/EPL_Hist_Data'
all_files = glob.glob(path + "/*.csv")

data = []

# read csv files containing scrapped data
for file in all_files:
    df = pd.read_csv(file)
    data.append(df)

# combine csv files for different clubs into one data frame
EPL_df = pd.concat(data, axis=0, ignore_index=True)

print(EPL_df.shape, '\n')
EPL_df.head()

# ## _Step 3: Cleaning data_

# In[3]:


# extract years for later filtering
EPL_df['startYear'] = EPL_df['Year'].str.slice(start=2, stop=4).astype(int)
EPL_df.drop(EPL_df.columns[0], axis=1, inplace=True)

# Filter out seasons before 2008/09 and our test year 2018/19
Cond1 = ((EPL_df['startYear'] >= 8) & (EPL_df['startYear'] <= 18))

# Exclude zero rows, i.e. when clubs were not in EPL for certain seasons
Cond2 = (EPL_df['Goals'] > 0)

EPL_cleandf = EPL_df.loc[Cond1 & Cond2]

pd.options.display.max_columns = None
EPL_cleandf.head()

# In[4]:


pd.options.mode.chained_assignment = None

# Filtering out one specific problematic data point we found
# EPL_cleandf.loc[EPL_cleandf['Shots on target'] > 600]
# EPL_cleandf.drop([128], inplace=True)

# In[5]:


print(EPL_cleandf.dtypes)

# In[6]:


pd.options.mode.chained_assignment = None

# Converting columns with object types into numerical values
for col in EPL_cleandf.columns:
    if (EPL_cleandf.loc[:, col].dtype == object) & (col != 'Year') & (col != 'Club'):
        EPL_cleandf[col] = EPL_cleandf[col].astype(str)
        EPL_cleandf[col] = EPL_cleandf[col].apply(lambda x: re.sub("[,.;%']", "", x))
        EPL_cleandf[col] = EPL_cleandf[col].astype(float)

# ## _Step 4: Analyzing data_

# In[7]:


# Potential predictor variables for expected goals scored
offense_df = EPL_cleandf[['Goals', 'Shots', 'Shots on target', 'Shooting accuracy', 'Hit woodwork',
                          'Passes', 'Pass accuracy', 'Crosses', 'Cross accuracy', 'Offsides']]

# In[8]:


# pair plot for potential predictor variables for expected goals scored
# sns.set(font_scale=1.5)
# pplot = sns.pairplot(offense_df, dropna=True, kind='reg')
# pplot.fig.suptitle("Pair plot of potential predictor variables for goals scored", y=1.01)
#
# plt.show()

# In[9]:


offense_df.corr()

# ### _Observations_
#
# - As expected, the number of goals scored share a strong (positive) linear relationship with the number of shots, shots on target and shooting accuracy.
#
# - The number of times the woodwork is hit also shows a fairly strong relationship with goals scored and it may be useful in adjusting for 'unlucky' events in matches.
#
# - Another meaningful explanatory variable when it comes to explaining goals scored are the number of passes made and the passing accuracy.
#
# - Crosses do not appear to correlate with the number of goals scored. Surprisingly, the linear regression line suggests a negative linear relationship between the crossing accuracy and the number of goals scored.
#
# - Offside statistics do not suggest any meaningful relationships with the other variables.

# In[10]:


# Potential predictor variables for expected goals conceded
defense_df = EPL_cleandf[
    ['Goals conceded', 'Clean sheets', 'Saves', 'Blocked shots', 'Interceptions', 'Clearances', 'Headed Clearance',
     'Aerial Battles*Duels Won']]

# remove data points with 0 as entry, due to lack of data collection in earlier seasons for certain stats
defense_df = defense_df[(defense_df[
                             ['Goals conceded', 'Clean sheets', 'Saves', 'Blocked shots', 'Interceptions', 'Clearances',
                              'Headed Clearance',
                              'Aerial Battles*Duels Won']] != 0).all(axis=1)]

# remove data points that have single digit saves, indicating likely data entry error
defense_df = defense_df[defense_df['Saves'] > 38]

# In[11]:


# pair plot for potential predictor variables for expected goals conceded
# sns.set(font_scale=1.5)
# pplot = sns.pairplot(defense_df, dropna=True, kind='reg', height=3)
# pplot.fig.suptitle("Pair plot of potential predictor variables for goals conceded", y=1.01)
#
# plt.show()

# In[12]:


defense_df.corr()

# ### _Observations_
#
# - Clearly, we would expect that the greater the number of clean sheets kept, the lower the number of goals conceded and we can see this clearly from the scatter plot.
#
# - Intuitively, we would expect that the greater the number of saves made in a match would lead to lesser goals conceded due to an oustanding performance from the keeper. However, the scatter plot suggests a strong positive relationship between the number of saves and number of goals conceded. This makes sense if we consider that teams that are weaker defensively gives away more shot opportunities, leading to more goals conceded as well as saves made by their keeper.
#
# - On the other hand, blocked shots show a strong negative correlation with goals conceded. This is likely due to a stronger and more responsive defense, which in turn puts less pressure on their keeper.
#
# - Interceptions seem to be less useful in predicting goals conceded, and might be better used in assessing midfield performance.
#
# - The same can probably be said about clearances, headed clearances and aerial battles won.

# ## _Step 5: Building the regression model_

# In[13]:


# Build linear regression model for expected goals scored based on statistics per match (38 games played per season)
x_EGS = EPL_cleandf[['Shots on target']] / 38
y_EGS = EPL_cleandf[['Goals per match']]
# 'Hit woodwork', 'Passes'
goals_scored_lr = linear_model.LinearRegression()
goals_scored_lr.fit(x_EGS, y_EGS)

# Note: We are using only 1 predictor variable here for the purpose of this project.
# The later testing of our model is on a match to match basis, where we draw data from a seperate source, which do not have
# the same columns as the earlier data frame except for certain key column.


# In[14]:


print(goals_scored_lr.coef_)
print(goals_scored_lr.intercept_)

# In[15]:


# Build linear regression model for expected goals conceded based on statistics per match (38 games played per season)
x_EGC = EPL_cleandf[['Clean sheets']] / 38
y_EGC = EPL_cleandf[['Goals conceded per match']]
# 'Saves', 'Blocked shots'
goals_conceded_lr = linear_model.LinearRegression()
goals_conceded_lr.fit(x_EGC, y_EGC)

# For the same reason as anove, we are only using 1 predictor variable here.


# In[16]:


print(goals_conceded_lr.coef_)
print(goals_conceded_lr.intercept_)

# ## _Step 6: Cleaning another data source for testing_

# In[17]:


# Reading in data from downloaded csv, where each row represents a match in the 18/19 season
test_df = pd.read_csv('./Data/EPL/EPL_test/season-1819.csv')
# pd.read_csv("https://www.football-data.co.uk/mmz4281/1819/E0.csv")

test_df.head()

# In[18]:


# Retrieving the relevant columns
test_df = test_df[
    ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HST', 'AST', 'BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5',
     'BbAv<2.5']]
test_df.head()

# In[19]:


# Creating clean sheet columns for modelling purposes later on
test_df['Home_clean_sheet'] = np.where(test_df['FTAG'] == 0, 1, 0)
test_df['Away_clean_sheet'] = np.where(test_df['FTHG'] == 0, 1, 0)
test_df.head()

# ## _Step 7: Reshaping data_

# ### Comment: With our earlier model coefficients, we are going to use the averages of earlier matches played during the season as our input for the model. Therefore, our following steps will only start betting after the fifth match of each team when a more stable average can be achieved.

# In[20]:


dictOfTeamDF = {}
for team in test_df['HomeTeam'].unique():
    dictOfTeamDF[team] = test_df[(test_df['HomeTeam'] == team) |
                                 (test_df['AwayTeam'] == team)].reset_index().drop(columns='index')


# In[21]:


# This function essentially adds the predicted goals scored and conceded for each team on a rolling average basis.

def add_predictions(teamName, teamDF):
    # Shots on target
    accu_SoT = []
    # Clean sheets
    accu_CS = []
    match_records = []

    for index, row in teamDF.iterrows():
        match_record = []
        if row['HomeTeam'] == teamName:
            accu_SoT.append(row['HST'])
            accu_CS.append(row['Home_clean_sheet'])
        else:
            accu_SoT.append(row['AST'])
            accu_CS.append(row['Away_clean_sheet'])

        if index >= 4:
            # Shift 1 so that the SoT for the current match played is not taken into account when predicting
            # goals scored, so there is no cheating.
            predicted_GS = (np.array(pd.DataFrame(accu_SoT).shift(1).dropna()).mean() * goals_scored_lr.coef_ +
                            goals_scored_lr.intercept_)[0][0]
            predicted_GC = (np.array(pd.DataFrame(accu_CS).shift(1).dropna()).mean() * goals_conceded_lr.coef_ +
                            goals_conceded_lr.intercept_)[0][0]
            match_record.append(row['Date'])
            match_record.append(row['HomeTeam'])
            match_record.append(row['AwayTeam'])
            match_record.append(predicted_GS)
            match_record.append(predicted_GC)
            match_records.append(match_record)

    name = teamName + '_'

    df = pd.DataFrame(match_records, columns=['Date', 'HomeTeam', 'AwayTeam',
                                              name + 'predicted_GS', name + 'predicted_GC'])
    return df


# In[22]:


listOfResultsdf = []

for teamName, teamDF in dictOfTeamDF.items():
    listOfResultsdf.append(add_predictions(teamName, teamDF))

# In[23]:


combined_df = test_df
for result in listOfResultsdf:
    combined_df = pd.merge(combined_df, result, how='left',
                           left_on=['Date', 'HomeTeam', 'AwayTeam'], right_on=['Date', 'HomeTeam', 'AwayTeam'])
combined_df.tail()

# In[24]:


# Creating a new data frame with a cleaner interface as compared to the earlier combined_df

predict_df = []
for index, row in combined_df.iterrows():
    row_details = []
    row_details.append(row['Date'])
    row_details.append(row['HomeTeam'])
    row_details.append(row['AwayTeam'])
    row_details.append(row['FTHG'])
    row_details.append(row['FTAG'])
    row_details.append(row['BbAvH'])
    row_details.append(row['BbAvD'])
    row_details.append(row['BbAvA'])
    row_details.append(row['BbAv>2.5'])
    row_details.append(row['BbAv<2.5'])
    hgs = row['HomeTeam'] + '_predicted_GS'
    hgc = row['HomeTeam'] + '_predicted_GC'
    ags = row['AwayTeam'] + '_predicted_GS'
    agc = row['AwayTeam'] + '_predicted_GC'

    # In our simulation, the expected goals scored by the home team is a simple average of the results of the goals scored
    # and goals conceded model
    predict_hgs = (row[hgs] + row[agc]) / 2
    predict_ags = (row[hgc] + row[ags]) / 2
    row_details.append(predict_hgs)
    row_details.append(predict_ags)
    predict_df.append(row_details)

predict_df = pd.DataFrame(predict_df, columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'BbAvH', 'BbAvD',
                                               'BbAvA', 'BbAv>2.5', 'BbAv<2.5', 'predict_hgs', 'predict_ags'])

predict_df.tail()

# In[25]:


# remove the first few rows which have no predictions as data were not available yet
predict_df = predict_df[~(predict_df['predict_hgs'].isna()) & ~(predict_df['predict_ags'].isna())]
predict_df.head()

# ## _Step 8: Applying Poisson distribution_

# In[26]:


# With the expected goals scored and using the Poisson distribution, we can compute probabilities of the different
# scorelines occuring. e.g. a 0-0 scoreline is simply the product of the PMF for the home team and away team scoring 0 goals,
# assuming independence between these 2 PMFs.

predict_df['0-0'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(0, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(0, x))

predict_df['1-1'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(1, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(1, x))

predict_df['1-0'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(1, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(0, x))

predict_df['2-0'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(2, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(0, x))

predict_df['0-1'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(0, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(1, x))

predict_df['0-2'] = predict_df['predict_hgs'].apply(lambda x: poisson.pmf(0, x)) * predict_df['predict_ags'].apply(
    lambda x: poisson.pmf(2, x))

# For the purpose of this project, we will evaluate the profitability of betting in the over/under 2.5 types.

# The probability of 2 goals and under is the sum of the 6 scoreline probabilities.
predict_df['under2Goals'] = predict_df['0-0'] + predict_df['1-1'] + predict_df['1-0'] + predict_df['2-0'] + predict_df[
    '0-1'] + predict_df['0-2']

# The remaining probability is for 3 goals and above.
predict_df['over2Goals'] = 1 - predict_df['under2Goals']

predict_df.head()

# In[27]:


# Creating a column for total goals scored
predict_df['TGS'] = predict_df['FTHG'] + predict_df['FTAG']

# The implied probability is simply the reciprocal of the odds in decimal form (which is as given)
predict_df['implied_over2.5'] = 1 / predict_df['BbAv>2.5']
predict_df['implied_under2.5'] = 1 / predict_df['BbAv<2.5']
predict_df.head()

# In[28]:


# The margin reflects how many % over the implied % before we are willing to bet.
# e.g. if the implied probability of over 2.5 goals is 55% and our model reflects a 58% probability, we will not bet.
# Otherwhise, if our model reflects a 61% probability, we will bet.

margin = 0.05

BetOver = np.where(predict_df['over2Goals'] - margin > predict_df['implied_over2.5'], 1, 0)
BetUnder = np.where(predict_df['under2Goals'] - margin > predict_df['implied_under2.5'], -1, 0)

# 1 represents an over 2.5 bet, while -1 represents an under 2.5 bet. 0 indicates no bet.
# It is safe to do this as it is an either-or bet,meaning there will be no scenario where our model indicates a bet for
# both under and over 2.5 in the same match.

predict_df['betPlaced'] = BetOver + BetUnder
predict_df.head()


# ## _Step 9: Evaluating profit & loss_

# In[29]:


# This function calculates the winnings or losses for each match, assuming a $1 bet on each case.
def compute_PL(df):
    if (df['betPlaced'] == 1):
        if (df['TGS'] > 2):
            # since odds are given in decimal form, the -1 reflects the return of original bet
            return df['BbAv>2.5'] - 1
        else:
            return -1

    elif (df['betPlaced'] == -1):
        if (df['TGS'] <= 2):
            return df['BbAv<2.5'] - 1
        else:
            return -1

    else:
        return 0


# In[30]:


predict_df['PL'] = predict_df.apply(compute_PL, axis=1)

# In[31]:


print('Ending wealth (assuming $1 bet each time):', round(predict_df['PL'].sum(), 2))
print('Maximum drawdown:', round(min(predict_df['PL'].cumsum()), 2))

predict_df['PL'].cumsum().plot()
plt.title('Cumulative P/L profile')
plt.xlabel('Matches played')
plt.ylabel('dollar winnings')
plt.show()

# # Insights and Evaluation
# - From our earlier observations, the predictor variables for the goals are probably well-known by now.
# - The cumulatve P/L seems to suggest that predictions improve later on in the season, when teams' performance evaluation is more accurate with the availability of more data.
# - There are several limitations with the model, including but not limited to the following:
#     - Failure to consider any home advantage and factoring it into the goals scored.
#     - Assumption of the same Poisson distribution throughout a single match.
#     - Multicollinearity is quite a big problem due to the high correlation between many of the known predictor variables.
#
# We will illustrate the first 2 limitations below.

# In[32]:


path = r'./Data/EPL/original'
all_files = glob.glob(path + "/*.csv")

data = []

for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    data.append(df)

EPL_df = pd.concat(data, axis=0, ignore_index=True, sort=False)

EPL_newdf = EPL_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS',
                    'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5',
                    'BbAv<2.5']]

FTHG = EPL_newdf['FTHG'].value_counts().rename_axis('GS').reset_index(name='HGcounts').sort_values(['GS'])
FTAG = EPL_newdf['FTAG'].value_counts().rename_axis('GS').reset_index(name='AGcounts').sort_values(['GS'])

addRow = {'GS': 8, 'AGcounts': 0}
addRow2 = {'GS': 9, 'AGcounts': 0}
FTAG = FTAG.append(addRow, ignore_index=True)
FTAG = FTAG.append(addRow2, ignore_index=True)

minGoals = 0.0
maxGoals = max(max(EPL_newdf['FTHG']), max(EPL_newdf['FTAG']))

# create plot
fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(int(maxGoals + 1))
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index - 0.2, FTHG['HGcounts'], bar_width,
                 alpha=opacity,
                 color='b',
                 label=('Full-time home goals'))

rects2 = plt.bar(index + 0.2, FTAG['AGcounts'], bar_width,
                 alpha=opacity,
                 color='g',
                 label=('Full-time away goals'))

plt.xlabel('Number of goals scored')
plt.ylabel('Frequency')
plt.title('Frequency of goals scored by home vs away teams')
plt.xticks(index)
plt.legend()

plt.tight_layout()
plt.show()

# print statistics here
print('Mean of home goals:', round(EPL_newdf['FTHG'].mean(), 2))
print('Mean of away goals:', round(EPL_newdf['FTAG'].mean(), 2))

print(
    'Clearly, there is some form of home advantage and teams playing at home can expect to score more goals on average.')

# In[33]:


fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1 = EPL_newdf["FTR"].value_counts().plot.pie(autopct="%1.0f%%",
                                               colors=sns.color_palette("rainbow", 3),
                                               wedgeprops={"linewidth": 2, "edgecolor": "white"},
                                               labels=['Home win', 'Away win', 'Draw'], ax=ax1)

ax1.set_title("Proportion of results at full-time")
ax1.set_ylabel(None)
ax1.set_xlabel(None)

ax2 = EPL_newdf["HTR"].value_counts().plot.pie(autopct="%1.0f%%",
                                               colors=sns.color_palette("rainbow", 3),
                                               wedgeprops={"linewidth": 2, "edgecolor": "white"},
                                               labels=['Home win', 'Away win', 'Draw'], ax=ax2)

ax2.set_title("Proportion of results at half-time")
ax2.set_ylabel(None)
ax2.set_xlabel(None)

plt.show()
print('Another illustration of home team advantage')

# In[34]:


EPL_newdf['FTGD'] = EPL_newdf['FTHG'] - EPL_newdf['FTAG']
EPL_newdf['HTGD'] = EPL_newdf['HTHG'] - EPL_newdf['HTAG']

EPL_newdf['secHalfHG'] = EPL_newdf['FTHG'] - EPL_newdf['HTHG']
EPL_newdf['secHalfAG'] = EPL_newdf['FTAG'] - EPL_newdf['HTAG']

print('Goals scored in the first-half')
print('------------------------------')
print('home team:', EPL_newdf['HTHG'].sum())
print('away team:', EPL_newdf['HTAG'].sum(), '\n')

print('Goals scored in the second-half')
print('------------------------------')
print('home team:', EPL_newdf['secHalfHG'].sum())
print('away team:', EPL_newdf['secHalfAG'].sum())

print('\n', 'More goals are scored in the second-half compared to the first-half')

# # Conclusion
#
# - While our model does not appear to generate any significant profits, it is clear that a data-driven approach has the potential to help bettors. Furthermore, the model that we used for this project is fairly trivial utilizing only one predictor.
# - The availability and quality of data needed is also fairly important in a project like this. There are a fair number of commercial data providers which could potentially enhance the results of a football prediction model with better quality data.
# - The odds offered by Singapore Pools are a travesty.