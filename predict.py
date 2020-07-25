#data preprocessing
import time
import pandas as pd
from sklearn.externals import joblib
from prediction.dataCleaning import dataClean
from qiutan.league import leagueId




class prediction(object):

    # 从已结束的比赛计算赛果参数,qiutan: 数据库，factor:实力更新因子
    def get_general_coefficient(self, qiutan, playing_stat, factor):
        con = qiutan.db
        # spider = EcSpider()
        # leagueId = spider.leagueId
        all_bs_data = pd.read_sql("select * from all_bs_data", con)
        all_coeff = dict()
        col_h = '{}general_coeff_h'.format(factor)
        col_a = '{}general_coeff_a'.format(factor)
        for league in leagueId:
            data = all_bs_data[all_bs_data.league == league]
            data = data.sort_values(by=['season', 'lunci'], ascending=True)
            playteam = pd.concat([data['hometeam'], data['awayteam']])
            playteam = playteam.drop_duplicates(keep='first').reset_index(drop=True)
            playteam = playteam.to_dict()
            coeff = dict([val, 1.0] for key, val in playteam.items())  # 每队原始实力参数赋于1
            data = data.reset_index(drop=True)

            for n in range(len(data)):
                match = data.iloc[n]
                if match['FTR'] == 'H':
                    coeff_home = coeff[match['hometeam']] + 1/factor * coeff[match['awayteam']]
                    coeff_away = coeff[match['awayteam']] - 1/factor * coeff[match['awayteam']]
                elif match['FTR'] == 'D':
                    diff = coeff[match['hometeam']] - coeff[match['awayteam']]
                    coeff_home = coeff[match['hometeam']] - 1/factor * diff
                    coeff_away = coeff[match['awayteam']] + 1/factor * diff
                elif match['FTR'] == 'A':
                    coeff_home = coeff[match['hometeam']] - 1/factor * coeff[match['hometeam']]
                    coeff_away = coeff[match['awayteam']] + 1/factor * coeff[match['hometeam']]

                coeff[match['hometeam']] = coeff_home
                coeff[match['awayteam']] = coeff_away
            all_coeff = dict(all_coeff, **coeff)

        # playing_stat['coeff_home'] = 0
        # playing_stat['coeff_away'] = 0
        for n in range(len(playing_stat)):
            playing_stat.loc[n, col_h] = all_coeff[playing_stat.loc[n, 'hometeam']]
            playing_stat.loc[n, col_a] = all_coeff[playing_stat.loc[n, 'awayteam']]

        return playing_stat


    def extract_feature(self, qiutan, playing_stat):
        clean = dataClean()

        # playing_stat = clean.get_offensive_defensive_coefficient(playing_stat)
        playing_stat = self.get_general_coefficient(qiutan, playing_stat, 3)
        playing_stat = self.get_general_coefficient(qiutan, playing_stat, 7)


        playing_stat = clean.select_data(playing_stat)
        playing_stat = clean.add_form(playing_stat, 5)
        playing_stat = clean.get_3form_points(playing_stat)
        playing_stat = clean.get_win_loss_streak(playing_stat)
        playing_stat = clean.get_diff(playing_stat)
        playing_stat = clean.scale_by_week(playing_stat)
        playing_stat = clean.get_rates(playing_stat)
        playing_stat = clean.get_oz_odds_value(playing_stat)
        # playing_stat = playing_stat.drop(
        #     ['season', 'lunci', 'hometeam','awayteam', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HLP', 'ALP', 'HTFormPts', 'ATFormPts', 'VTFormPtsStr',
        #      'HTFormPtsStr', 'ATFormPtsStr', 'HM4', 'HM5', 'AM4', 'AM5',
        #      'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std',
        #      'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean',
        #      'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std'], axis=1)
        #
        # playing_stat = playing_stat.drop(['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'VTWinStreak3', 'VTWinStreak5', 'VTLossStreak3', 'VTLossStreak5',
        #                                  'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean',
        #                                  'HM1','HM2','HM3','AM1','AM2','AM3', 'VM1', 'VM2', 'VM3', 'VM4', 'VM5', 'DiffLP','diff_win_rate',
        #                                  'hh_nb_games','hh_nb_wins','hh_nb_draws','aa_nb_games','aa_nb_wins','aa_nb_draws',
        #                                  'az_value9','oz_odds_value9',
        #                                  'FTR','FTRR'], 1)

        final_features = [
                    'HTGD', 'ATGD', 'HTP', 'ATP', 'HHTGD', 'HHTP', 'AATGD', 'AATP',
                    # 'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean',
                    'oz_odds_value0', 'oz_odds_value9', 'Diff_OZ_Value',
                    'oz_home9_std', 'oz_draw9_std', 'oz_away9_std',
                    # 'az_value0', 'az_value9','Diff_AZ_Value',
                    'h_win_rate', 'a_win_rate', 'VTFormPts', 'DiffPts', 'DiffFormPts', 'Diff_HA_Pts',
                    '3general_coeff_h', '3general_coeff_a', '7general_coeff_h', '7general_coeff_a',
                        ]

        # final_features = [
        #             'HTGD', 'ATGD', 'HTP', 'ATP', 'HHTGD', 'HHTP', 'AATGD', 'AATP',
        #             'oz_home9_std', 'oz_draw9_std', 'oz_away9_std',
        #             'az_value0', 'VTFormPts', 'DiffPts', 'DiffFormPts','Diff_AZ_Value', 'Diff_HA_Pts',
        #             'h_win_rate', 'a_win_rate',  'oz_odds_value0',  'Diff_OZ_Value'
        #                 ]


        playing_stat = playing_stat[final_features]

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
    filename = loc + r"datasets\taday_matchs.csv"
    taday_matchs = pd.read_csv(filename, encoding="gbk")
    match_info = taday_matchs[['league', 'hometeam', 'awayteam', 'bs_time']]

    predict = prediction()
    X_all = predict.extract_feature(taday_matchs)

    # load xgboost model
    xgboost_model = joblib.load(r'./prediction/model/final_model/xgboost_joblib(混合).dat')
    # make predictions for test data
    y_pred = xgboost_model.predict(X_all)
    res = pd.DataFrame(y_pred, columns=['y_pred'])
    pred_res =pd.concat([match_info, res], axis=1)
    nowtime = time.strftime('%Y%m%d_%H_%M', time.localtime())
    pred_res.to_csv("./prediction/datasets/predResult{}.csv".format(nowtime), encoding = "gbk", index=None)




