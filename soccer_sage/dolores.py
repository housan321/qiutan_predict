#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:53:19 2019
@author: Juan Beleño
"""

from soccer_sage.config import SoccerSageConfig
from soccer_sage.dataset import (save_dataset,bayesian_network_prediction2)
from soccer_sage.preprocess import get_rank
from soccer_sage.dataset import bayesian_network_prediction
from soccer_sage.train import train_model, train_regression_model
from soccer_sage.predict import predict_results, predict_regressions
from predict import prediction
import numpy as np
import pandas as pd
import time

def dolores():
    save_dataset()
# train_model()
# train_regression_model()
# predict_results()
# predict_regressions()



def predict_match(qiutan):
    loc = r"D:\qiutan_predict\prediction\\"
    loadfile = loc + r"datasets\taday_matchs.csv"
    pred_res = pd.DataFrame()
    taday_matchs = pd.read_csv(loadfile, encoding="gbk")

    match_info = taday_matchs[['league', 'hometeam', 'awayteam', 'bs_time']]
    oz_odds = taday_matchs[['oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean']]

    ad_cpt, gh_cpt, ga_cpt, prediction_cpt, pi_ratings = save_dataset()

    for n in range(len(match_info)):
        match = match_info.ix[n]
        league = match.league
        home_team_id = match.hometeam
        away_team_id = match.awayteam
        if home_team_id in pi_ratings:
            if league in pi_ratings[home_team_id]:
                pi_ratings_home = pi_ratings[home_team_id][league]['home']
            else:continue
        else:continue
        if away_team_id in pi_ratings:
            if league in pi_ratings[away_team_id]:
                pi_ratings_away = pi_ratings[away_team_id][league]['away']
            else:continue
        else:continue
        dis_ratings = pi_ratings_home - pi_ratings_away
        rank = get_rank(dis_ratings)
        y_pred, pred_gh, pred_ga = bayesian_network_prediction2(rank, ad_cpt, gh_cpt, ga_cpt, prediction_cpt)
        match = match.as_matrix()
        res = np.concatenate((match, y_pred), axis=0)
        res = pd.DataFrame([res], columns=['联赛', '主队', '客队', '比赛时间', 'y_pred_away', 'y_pred_draw', 'y_pred_home'])
        # print(league, home_team_id, away_team_id, pred)

        QI = 1/oz_odds.loc[n]['oz_home9_mean'] + 1/oz_odds.loc[n]['oz_draw9_mean'] + 1/oz_odds.loc[n]['oz_away9_mean']
        return_percentage = 1/QI
        res['prob_home'] = return_percentage / oz_odds.loc[n]['oz_home9_mean']  # kelly_home
        res['prob_draw'] = return_percentage / oz_odds.loc[n]['oz_draw9_mean']  # kelly_draw
        res['prob_away'] = return_percentage / oz_odds.loc[n]['oz_away9_mean']  # kelly_away
        res = round(res, 2)
        pred_res = pred_res.append(res, ignore_index=True)

    # ## 计算哪些比赛具有投注价值
    value_bet = []
    pred = pred_res[['y_pred_home', 'y_pred_draw','y_pred_away']]
    prob = pred_res[['prob_home', 'prob_draw', 'prob_away']]
    pred.columns = ['home', 'draw', 'away']
    prob.columns = ['home', 'draw', 'away']
    margin = pred - prob
    max_value = pred.max(axis = 1)
    max_index = np.argmax(pred.values, axis = 1)
    max_margin = margin.max(axis = 1)
    for i in range(len(margin)):
        index = max_index[i]
        value = max_value[i]
        difference = max_margin[i]
        if difference >= 0.15: #预测值与庄家概率的差值
            value_bet.append('反路')
            continue
        if value >= 0.48 and abs(margin.iloc[i, index]) < 0.07: #只选择概率大于48的比赛
            value_bet.append('正路')
            continue
        value_bet.append(' ') # 排除的场次
    pred_res['value_bet'] = value_bet

    # pred_result = pd.concat([match_info, pred_res], axis=1)
    nowtime = time.strftime('%Y%m%d_%H_%M', time.localtime())
    pred_res.to_csv("./prediction/datasets/predResult{}.csv".format(nowtime), encoding = "gbk", index=None)



    # predict = prediction()
    # X_all = predict.extract_feature(qiutan, taday_matchs)
    #
    #
    # for n in range(len(match_info)):
    #     data = X_all[n:n+1].values
    #     league = match_info.league[n]
    #     model_file = r'./prediction/model/final_model/xgboost_joblib({}final).dat'.format(league)
    #     if (os.path.exists(model_file)):
    #         xgboost_model = joblib.load(model_file)
    #     else: xgboost_model = joblib.load(r'./prediction/model/final_model/xgboost_joblib(混合RR).dat') ##如果没有该轮赛的预测器，则默认用西甲轮赛预测器
    #     y_pred = xgboost_model.predict_proba(data)
    #     res = pd.DataFrame(y_pred, columns=['y_pred_away', 'y_pred_draw','y_pred_home'])
    #     res = res[['y_pred_home', 'y_pred_draw','y_pred_away']] #调整列的顺序
    #     # res = pd.DataFrame(y_pred, columns=['y_pred_away', 'y_pred_home'])
    #     # res = res[['y_pred_home', 'y_pred_away']]
    #
    #     QI = 1/oz_odds.loc[n]['oz_home9_mean'] + 1/oz_odds.loc[n]['oz_draw9_mean'] + 1/oz_odds.loc[n]['oz_away9_mean']
    #     return_percentage = 1/QI
    #     res['probability_home'] = return_percentage / oz_odds.loc[n]['oz_home9_mean']  # kelly_home
    #     res['probability_draw'] = return_percentage / oz_odds.loc[n]['oz_draw9_mean']  # kelly_draw
    #     res['probability_away'] = return_percentage / oz_odds.loc[n]['oz_away9_mean']  # kelly_away
    #
    #     # kelly_home = (res['y_pred_home'] * oz_odds.loc[n]['oz_home9_mean'] - 1) / (oz_odds.loc[n]['oz_home9_mean'] - 1)
    #     # kelly_draw = (res['y_pred_draw'] * oz_odds.loc[n]['oz_draw9_mean'] - 1) / (oz_odds.loc[n]['oz_draw9_mean'] - 1)
    #     # kelly_away = (res['y_pred_away'] * oz_odds.loc[n]['oz_away9_mean'] - 1) / (oz_odds.loc[n]['oz_away9_mean'] - 1)
    #     # res['kelly_home'] = kelly_home
    #     # res['kelly_draw'] = kelly_draw
    #     # res['kelly_away'] = kelly_away
    #     # res['QI'] = QI
    #     # res['return_percentage'] = return_percentage
    #
    #     res = round(res, 2)
    #     pred_res = pred_res.append(res, ignore_index=True)
    #
    # ## 计算哪些比赛具有投注价值
    # value_bet = []
    # pred = pred_res[['y_pred_home', 'y_pred_draw','y_pred_away']]
    # prob = pred_res[['probability_home', 'probability_draw', 'probability_away']]
    # pred.columns = ['home', 'draw', 'away']
    # prob.columns = ['home', 'draw', 'away']
    # margin = pred - prob
    # max_value = pred.max(axis = 1)
    # max_index = np.argmax(pred.values, axis = 1)
    # max_margin = margin.max(axis = 1)
    # for i in range(len(margin)):
    #     index = max_index[i]
    #     value = max_value[i]
    #     difference = max_margin[i]
    #     if difference >= 0.15: #预测值与庄家概率的差值
    #         value_bet.append('反路')
    #         continue
    #     if value >= 0.48 and abs(margin.iloc[i, index]) < 0.07: #只选择概率大于48的比赛
    #         value_bet.append('正路')
    #         continue
    #     value_bet.append(' ') # 排除的场次
    # pred_res['value_bet'] = value_bet
    #
    # pred_result = pd.concat([match_info, pred_res], axis=1)
    # nowtime = time.strftime('%Y%m%d_%H_%M', time.localtime())
    # pred_result.to_csv("./prediction/datasets/predResult{}.csv".format(nowtime), encoding = "gbk", index=None)






