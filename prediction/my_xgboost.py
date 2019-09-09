#!/usr/bin/env python
# coding: utf-8

# # Predicting the Winning Football Team
# 
# Can we design a predictive model capable of accurately predicting if the home team will win a football match? 
# 
# ![alt text](https://6544-presscdn-0-22-pagely.netdna-ssl.com/wp-content/uploads/2017/04/English-Premier-League.jpg "Logo Title Text 1")
# 
# ## Steps
# 
# 1. We will clean our dataset
# 2. Split it into training and testing data (12 features & 1 target (winning team (Home/Away/Draw))
# 3. Train 3 different classifiers on the data 
#   -Logistic Regression
#   -Support Vector Machine 
#   -XGBoost
# 4. Use the best Classifer to predict who will win given an away team and a home team
# 
# ## History
# 
# Sports betting is a 500 billion dollar market (Sydney Herald)
# 
# ![alt text](https://static1.squarespace.com/static/506a95bbc4aa0491a951c141/t/51a55d97e4b00f4428967e64/1369791896526/sports-620x349.jpg "Logo Title Text 1")
# 
# Kaggle hosts a yearly competiton called March Madness 
# 
# https://www.kaggle.com/c/march-machine-learning-mania-2017/kernels
# 
# Several Papers on this 
# 
# https://arxiv.org/pdf/1511.05837.pdf
# 
# "It is possible to predict the winner of English county twenty twenty cricket games in almost two thirds of instances."
# 
# https://arxiv.org/pdf/1411.1243.pdf
# 
# "Something that becomes clear from the results is that Twitter contains enough information to be useful for
# predicting outcomes in the Premier League"
# 
# https://qz.com/233830/world-cup-germany-argentina-predictions-microsoft/
# 
# For the 2014 World Cup, Bing correctly predicted the outcomes for all of the 15 games in the knockout round.
# 
# So the right questions to ask are
# 
# -What model should we use?
# -What are the features (the aspects of a game) that matter the most to predicting a team win? Does being the home team give a team the advantage? 
# 
# ## Dataset
# 
# - Football is played by 250 million players in over 200 countries (most popular sport globally)
# - The English Premier League is the most popular domestic team in the world
# - Retrived dataset from http://football-data.co.uk/data.php
# 
# ![alt text](http://i.imgur.com/YRIctyo.png "Logo Title Text 1")
# 
# - Football is a team sport, a cheering crowd helps morale
# - Familarity with pitch and weather conditions helps
# - No need to travel (less fatigue)
# 
# Acrononyms- https://rstudio-pubs-static.s3.amazonaws.com/179121_70eb412bbe6c4a55837f2439e5ae6d4e.html
# 
# ## Other repositories
# 
# - https://github.com/rsibi/epl-prediction-2017 (EPL prediction)
# - https://github.com/adeshpande3/March-Madness-2017 (NCAA prediction)

# ## Import Dependencies

# In[ ]:


#data preprocessing
import pandas as pd
import numpy as np
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
# the outcome (dependent variable) has only a limited number of possible values.
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
#displayd data
from IPython.display import display
# save model
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")


leagueId = {'英超': '36', '西甲': '31', '意甲': '34', '德甲': '8', '法甲': '11',
            '英冠': '37', '苏超': '29', '葡超': '23', '挪超': '22', '瑞典超': '26',
            '中超': '60', '日职联': '25', '日职乙': '284', '韩K联': '15',
            '美职业': '21', '巴西甲': '4'}


league = '日职乙'
# In[2]:
modelname = r'./model/xgboost_joblib({}).dat'.format(league)
# Read data and drop redundant column.
data = pd.read_csv(r'./datasets/final_dataset/final_dataset({}).csv'.format(leagueId[league]), encoding = "gbk")

data.dropna(inplace=True)

data.drop(['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'VTWinStreak3', 'VTWinStreak5', 'VTLossStreak3', 'VTLossStreak5',
            'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean',
            'HM1','HM2','HM3','AM1','AM2','AM3','DiffLP','diff_win_rate',
           'hh_nb_games','hh_nb_wins','hh_nb_draws','aa_nb_games','aa_nb_wins','aa_nb_draws',
           'az_value9','oz_odds_value9',

           # 'oz_home9_std','oz_draw9_std','oz_away9_std','az_value0', 'Diff_AZ_Value','oz_odds_value0','Diff_OZ_Value'

           ], 1, inplace=True)

# data.drop(['oz_home9_std', 'oz_draw9_std', 'oz_away9_std', 'oz_odds_value0', 'az_value0',
#            'Diff_OZ_Value','Diff_AZ_Value'],  1, inplace=True)

# data.drop(['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts',
#            'HTP', 'ATP', 'HHTGD', 'AATGD','HHTP', 'AATP',
#            'VTFormPts', 'Diff_HA_Pts', 'a_win_rate','h_win_rate'], 1, inplace=True)

# Preview data.
display(data.head())


#Full Time Result (H=Home Win, D=Draw, A=Away Win)
#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

#Input - 12 other features (fouls, shots, goals, misses,corners, red card, yellow cards)
#Output - Full Time Result (H=Home Win, D=Draw, A=Away Win) 


# ## Data Exploration

# In[3]:


#what is the win rate for the home team?

# Total number of matches.
n_matches = data.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = data.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(data[data.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("Total number of matches: {}".format(n_matches))
print("Number of features: {}".format(n_features))
print("Number of matches won by home team: {}".format(n_homewins))
print("Win rate of home team: {:.2f}%".format(win_rate))


# In[4]:


# Visualising distribution of data
from pandas.plotting import scatter_matrix

#the scatter matrix is plotting each of the columns specified against each other column.
#You would have observed that the diagonal graph is defined as a histogram, which means that in the 
#section of the plot matrix where the variable is against itself, a histogram is plotted.

#Scatter plots show how much one variable is affected by another. 
#The relationship between two variables is called their correlation
#negative vs positive correlation

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

# scatter_matrix(data[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']], figsize=(10,10))


# ## Preparing the Data

# In[5]:


# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(['FTR','FTRR'],1)
y_all = data['FTR']

# Standardising the data.
from sklearn.preprocessing import scale

# Center to the mean and component wise scale to unit variance.
# cols = [['HTGD','ATGD','HTP','ATP','oz_home9_std','oz_draw9_std','oz_away9_std','az_value0','az_value9','DiffPts','DiffFormPts','DiffValue']]
# for col in cols:
#     X_all[col] = scale(X_all[col])
    


# In[6]:


#last 3 wins for both sides
# X_all.HM1 = X_all.HM1.astype('str')
# X_all.HM2 = X_all.HM2.astype('str')
# X_all.HM3 = X_all.HM3.astype('str')
# X_all.AM1 = X_all.AM1.astype('str')
# X_all.AM2 = X_all.AM2.astype('str')
# X_all.AM3 = X_all.AM3.astype('str')

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output


# X_part = X_all[['VM1','VM2','VM3','VM4','VM5']]      # ,'VM1','VM2','VM3','VM4','VM5'
# X_part = preprocess_features(X_part)

X_all = X_all.drop(['VM1','VM2','VM3','VM4','VM5'],1)
# X_all = pd.concat([X_all, X_part], axis=1)

print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[7]:


# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())



# from sklearn.preprocessing import StandardScaler
# standardizer = StandardScaler()
# X_all = standardizer.fit_transform(X_all)




# Shuffle and split the dataset into training and testing set.
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
#                                                     test_size = 0.3,
#                                                     random_state = 200,
#                                                     stratify = y_all)

X_train = X_all[:3000]
X_test = X_all[3000:]
y_train = y_all[:3000]
y_test = y_all[3000:]


# ## Training and Evaluating Models

# In[9]:


#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    # return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))
    return recall_score(target, y_pred, average='macro'), precision_score(target, y_pred, average='macro')

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))




# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print('')


# **Clearly XGBoost seems like the best model as it has the highest F1 score and accuracy score on the test set.**

# # Tuning the parameters of XGBoost.
# 
# ![alt text](https://i.stack.imgur.com/9GgQK.jpg "Logo Title Text 1")

# In[39]:


# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot

# TODO: Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.01],
               'n_estimators' : [80],
               'max_depth': [1],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }

# TODO: Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score,pos_label='H')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=7)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)


# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))


pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()
# plot feature importance
plot_importance(clf)
pyplot.show()





# Fit model using each importance as a threshold
thresholds = np.sort(clf.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(clf, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = xgb.XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [value for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))






# y_prob = clf.predict_proba(X_test)[:,1]
# y_pred1 = np.where(y_prob > 0.5, 'NH', 'H')



# save model to file
joblib.dump(clf, modelname)

xgboost_model = joblib.load(modelname)
y_pred = xgboost_model.predict(X_test)
acc = sum(y_test == y_pred) / float(len(y_pred))
print('acc = {}'.format(acc))


#prediction


# Possible Improvements?
# 
# -Adding Sentiment from Twitter, News Articles
# -More features from other data sources (how much did others bet, player specific health stats)
# 

