from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

#displayd data
from IPython.display import display

BATCH_SIZE = 256



if __name__=='__main__':
    leagueId = {'英超': '36', '西甲': '31', '意甲': '34', '德甲': '8', '法甲': '11',
                '英冠': '37', '苏超': '29', '葡超': '23', '挪超': '22', '瑞典超': '26', '俄超': '10',
                '中超': '60', '日职联': '25', '日职乙': '284', '韩K联': '15',
                '巴西甲': '4',
                '混合': 'max', }

    league = '混合'
    # In[2]:
    modelname = r'./model/xgboost_joblib({}).dat'.format(league)
    # Read data and drop redundant column.
    data = pd.read_csv(r'./datasets/final_dataset/final_dataset({}).csv'.format(leagueId[league]), encoding="gbk")

    data.dropna(inplace=True)
    print(list(data.columns))

    # select_features = ['FTR', 'FTRR','az_value9', 'Diff_AZ_Value', 'oz_odds_value9', 'Diff_OZ_Value', 'coff_home', 'coff_away']
    select_features = ['FTR', 'FTRR',
                       'HTGD', 'ATGD', 'HTP', 'ATP', 'HHTGD', 'HHTP', 'AATGD', 'AATP',
                       'oz_home9_std', 'oz_draw9_std', 'oz_away9_std',
                       'az_value0', 'Diff_AZ_Value', 'oz_odds_value0', 'Diff_OZ_Value',
                       'h_win_rate', 'a_win_rate', 'VTFormPts', 'DiffPts', 'DiffFormPts', 'Diff_HA_Pts',
                       ]
    data = data[select_features]

    # Preview data.
    display(data.head())

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

    # Separate into feature set and target variable
    #FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
    X_all = data.drop(['FTR','FTRR'],1)
    y_all = data[['FTR']]
    y_all[y_all['FTR'] == 'H'] = 1
    y_all[y_all['FTR'] == 'NH'] = 0

    X_all = X_all.values
    y_all = y_all.values


    # print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
    #
    # # Show the feature information by printing the first five rows
    # print("\nFeature values:")
    # display(X_all.head())

    X_train = X_all[:4000]
    X_val = X_all[4000:4900]
    X_test = X_all[4900:]

    y_train = y_all[:4000]
    y_val = y_all[4000:4900]
    y_test = y_all[4900:]

    # Normalise based on mean and variance of variables in training data
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_val = standardizer.transform(X_val)
    X_test = standardizer.transform(X_test)

    model = Sequential()

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,
                                   restore_best_weights=True)
    callbacks = [early_stopping]

    model.fit(X_train, y_train, epochs=300, batch_size=BATCH_SIZE, callbacks=callbacks, validation_data=(X_val, y_val))

    probas = model.predict(X_test)

    print('What is the Area under the ROC curve?')
    print('AUROC: {}'.format(roc_auc_score(y_test, probas)))
    print('')

    print('What are the metrics for a threshold of 0.5')
    predictions = [1 if prob[0]>=0.5 else 0 for prob in probas]
    print('Accuracy @ thresh 0.5: {}'.format(accuracy_score(y_test, predictions)))
    print('Precision @ thresh 0.5: {}'.format(precision_score(y_test, predictions)))
    print('Recall @ thresh 0.5: {}'.format(recall_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))
    print('')

    print('What is the threshold @ maximum precision?')
    precisions, recalls, thresh_prec_rec = precision_recall_curve(y_test, probas)
    idx_max_precision = np.argmax(precisions)
    thresh_max_precision = thresh_prec_rec[idx_max_precision]
    print('Thresh @ Highest Precision: {}'.format(thresh_max_precision))
    print('')

    print('What are the metrics for the threshold @ maximum precision?')
    predictions_max_prec = [1 if prob[0] >= thresh_max_precision else 0 for prob in probas]
    print('Accuracy @ Highest Precision: {}'.format(accuracy_score(y_test, predictions_max_prec)))
    print('Highest Precision: {}'.format(precision_score(y_test, predictions_max_prec)))
    print('Recall @ Highest Precision: {}'.format(recall_score(y_test, predictions_max_prec)))
    print(confusion_matrix(y_test, predictions_max_prec))
    print('')

    print(len(y_test))
    print([i for i in range(len(predictions_max_prec)) if predictions_max_prec[i]==1])

    # fpr, tpr, thresholds = roc_curve(y_test, probas)
    # plt.plot(fpr, tpr)
    # plt.show()




