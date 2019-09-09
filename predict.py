#data preprocessing
import time
import pandas as pd
from sklearn.externals import joblib
from prediction.dataCleaning import dataClean





class prediction(object):
    row_w = {'VM1': 'W', 'VM2': 'W','VM3': 'W', 'VM4': 'W', 'VM5': 'W'}
    row_d = {'VM1': 'D', 'VM2': 'D', 'VM3': 'D','VM4': 'D', 'VM5': 'D'}
    row_l = {'VM1': 'L', 'VM2': 'L', 'VM3': 'L','VM4': 'L', 'VM5': 'L'}


    def extract_feature(self, playing_stat):
        clean = dataClean()
        playing_stat = clean.select_data(playing_stat)
        playing_stat = clean.add_form(playing_stat, 5)
        playing_stat = clean.get_3form_points(playing_stat)
        playing_stat = clean.get_win_loss_streak(playing_stat)
        playing_stat = clean.get_diff(playing_stat)
        playing_stat = clean.scale_by_week(playing_stat)
        playing_stat = clean.get_rates(playing_stat)
        playing_stat = clean.get_oz_odds_value(playing_stat)
        playing_stat = playing_stat.drop(
            ['season', 'lunci', 'hometeam','awayteam', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HLP', 'ALP', 'HTFormPts', 'ATFormPts', 'VTFormPtsStr',
             'HTFormPtsStr', 'ATFormPtsStr', 'HM4', 'HM5', 'AM4', 'AM5',
             'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std',
             'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean',
             'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std'], axis=1)

        playing_stat = playing_stat.drop(['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'VTWinStreak3', 'VTWinStreak5', 'VTLossStreak3', 'VTLossStreak5',
                                         'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean',
                                         'HM1','HM2','HM3','AM1','AM2','AM3', 'VM1', 'VM2', 'VM3', 'VM4', 'VM5', 'DiffLP','diff_win_rate',
                                         'hh_nb_games','hh_nb_wins','hh_nb_draws','aa_nb_games','aa_nb_wins','aa_nb_draws',
                                         'az_value9','oz_odds_value9',
                                         'FTR','FTRR'], 1)

        #############################################################################################################
        # X_part = playing_stat[['VM1', 'VM2', 'VM3', 'VM4', 'VM5']]  # ,'VM1','VM2','VM3','VM4','VM5'
        # temp_row = [self.row_w, self.row_d, self.row_l]  # 中间数据，只要用于one-hot
        # temp_df = pd.DataFrame(temp_row)
        # X_part = pd.concat([X_part, temp_df], axis=0, ignore_index=True)
        # X_part = clean.preprocess_features(X_part)
        # X_part = X_part.drop(X_part.tail(3).index)

        # playing_stat = playing_stat.drop(['VM1', 'VM2', 'VM3', 'VM4', 'VM5'], 1)
        # playing_stat = pd.concat([playing_stat, X_part], axis=1)
        ###############################################################################################################
        return playing_stat








if __name__ == '__main__':  # 在win系统下必须要满足这个if条件
    loc = r"D:\qiutan_predict\prediction\\"
    filename = loc + r"datasets\league_match_data(31).csv"
    taday_matchs = pd.read_csv(filename, encoding="gbk")
    match_info = taday_matchs[['league', 'hometeam', 'awayteam', 'bs_time']]

    predict = prediction()
    X_all = predict.extract_feature(taday_matchs)

    # load xgboost model
    xgboost_model = joblib.load('./prediction/model/xgboost_joblib(all).dat')
    # make predictions for test data
    y_pred = xgboost_model.predict(X_all)
    res = pd.DataFrame(y_pred, columns=['y_pred'])
    pred_res =pd.concat([match_info, res], axis=1)
    nowtime = time.strftime('%Y%m%d%H%M', time.localtime())
    pred_res.to_csv("./prediction/datasets/prediction{}.csv".format(nowtime), encoding = "gbk", index=None)




