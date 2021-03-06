#!/usr/bin/env python
# coding: utf-8

# In[309]:

import pandas as pd
import re

class dataClean(object):
    def only_hw(self, string):
        if string == 'H':
            return 'H'
        else:
            return 'NH'

    # Identify Win/Loss Streaks if any.
    def get_3game_ws(self, string):
        if string[-3:] == 'WWW':
            return 1
        else:
            return 0

    def get_5game_ws(self, string):
        if string == 'WWWWW':
            return 1
        else:
            return 0

    def get_3game_ls(self, string):
        if string[-3:] == 'LLL':
            return 1
        else:
            return 0

    def get_5game_ls(self, string):
        if string == 'LLLLL':
            return 1
        else:
            return 0

    def get_points(self, result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0

    # Gets the form points.
    def get_form_points(self, string):
        sum = 0
        for letter in string:
            sum += self.get_points(letter)
        return sum

    # 主、客对赛的积分差
    def get_vs_points(self, result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 0
        else:
            return -3

    # Gets the form points.
    def get_vs_form_points(self, string):
        sum = 0
        for letter in string:
            sum += self.get_vs_points(letter)
        return sum


    #读入数据
    def select_data(self, playing_stat):
        columns_req = [ 'season','hometeam','awayteam', 'lunci', 'FTR', 'FTRR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTGD', 'ATGD', 'HTP', 'ATP', 'HLP','ALP', 'VTFormPtsStr', 'HTFormPtsStr', 'ATFormPtsStr',
                        'hh_nb_games', 'hh_nb_wins', 'hh_nb_draws','HHTGD', 'HHTP', 'aa_nb_games', 'aa_nb_wins', 'aa_nb_draws', 'AATGD', 'AATP',
                        'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std', 'oz_home9_std', 'oz_draw9_std', 'oz_away9_std',
                        'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean', 'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std', 'az_value0', 'az_value9',
                        '3general_coeff_h', '3general_coeff_a','7general_coeff_h', '7general_coeff_a',
                        # 'offensive_coeff_h','defensive_coeff_h','offensive_coeff_a','defensive_coeff_a'
                        ]
        playing_stat = playing_stat[columns_req]
        return playing_stat

    #整理近5场赛果
    def add_form(self, playing_stat, num):
        VTFormPtsStr = playing_stat['VTFormPtsStr']
        HTFormPtsStr = playing_stat['HTFormPtsStr']
        ATFormPtsStr = playing_stat['ATFormPtsStr']
        for n in range(1, num+1):
            playing_stat['VM' + str(n)] = VTFormPtsStr.str.slice(n-1, n)
            playing_stat['HM' + str(n)] = HTFormPtsStr.str.slice(n-1, n)
            playing_stat['AM' + str(n)] = ATFormPtsStr.str.slice(n-1, n)

        return playing_stat

    #计算主、客队近5场的积分，及近5场对赛的积分
    def get_3form_points(self, playing_stat):
        playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(self.get_form_points)
        playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(self.get_form_points)
        playing_stat['VTFormPts'] = playing_stat['VTFormPtsStr'].apply(self.get_vs_form_points)

        return playing_stat

    #计算主、客队是否连胜3场或5场
    def get_win_loss_streak(self, playing_stat):
        playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(self.get_3game_ws)
        playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(self.get_5game_ws)
        playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(self.get_3game_ls)
        playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(self.get_5game_ls)

        playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(self.get_3game_ws)
        playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(self.get_5game_ws)
        playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(self.get_3game_ls)
        playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(self.get_5game_ls)

        playing_stat['VTWinStreak3'] = playing_stat['VTFormPtsStr'].apply(self.get_3game_ws)
        playing_stat['VTWinStreak5'] = playing_stat['VTFormPtsStr'].apply(self.get_5game_ws)
        playing_stat['VTLossStreak3'] = playing_stat['VTFormPtsStr'].apply(self.get_3game_ls)
        playing_stat['VTLossStreak5'] = playing_stat['VTFormPtsStr'].apply(self.get_5game_ls)
        return playing_stat

    # 计算主、客队积分差、近5场积分差、排名差
    def get_diff(self, playing_stat):
        playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
        playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']
        playing_stat['DiffLP'] = playing_stat['HLP'] - playing_stat['ALP']
        playing_stat['Diff_AZ_Value'] = playing_stat['az_value9'] - playing_stat['az_value0']
        # playing_stat['Diff_offensive_h'] = playing_stat['offensive_coeff_h'] - playing_stat['defensive_coeff_a']
        # playing_stat['Diff_offensive_a'] = playing_stat['offensive_coeff_a'] - playing_stat['defensive_coeff_h']

        return playing_stat


    # Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
    def scale_by_week(self, playing_stat):
        cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP',  'HTGS', 'ATGS', 'HTGC', 'ATGC']
        playing_stat.lunci = playing_stat.lunci.astype(float)
        for col in cols:
            playing_stat[col] = playing_stat[col] / playing_stat.lunci

        playing_stat['HHTGD'] = playing_stat['HHTGD'] / playing_stat['hh_nb_games']
        playing_stat['AATGD'] = playing_stat['AATGD'] / playing_stat['aa_nb_games']
        playing_stat['HHTP'] = playing_stat['HHTP'] / playing_stat['hh_nb_games']
        playing_stat['AATP'] = playing_stat['AATP'] / playing_stat['aa_nb_games']
        playing_stat['Diff_HA_Pts'] = playing_stat['HHTP'] - playing_stat['AATP']

        return playing_stat

    # 计算欧赔平均值所指示值  OHMR:oz_home_mean rise 欧赔主胜赔率的增长率，ODMR:欧赔平手赔率的增长率
    def oz_odds_mean_index(self,  playing_stat):
        playing_stat['OHMR'] = (playing_stat['oz_home9_mean'] - playing_stat['oz_home0_mean']) / playing_stat['oz_home0_mean']
        playing_stat['ODMR'] = (playing_stat['oz_draw9_mean'] - playing_stat['oz_draw0_mean']) / playing_stat['oz_draw0_mean']
        playing_stat['OAMR'] = (playing_stat['oz_away9_mean'] - playing_stat['oz_away0_mean']) / playing_stat['oz_away0_mean']
        playing_stat['mean_idx'] = playing_stat['oz_away9_mean']
        rise_df = playing_stat[['OHMR','ODMR','OAMR']]
        for n in range(len(rise_df)):
            rise = rise_df.iloc[n]
            if abs(rise['OHMR'])<=0.05 and abs(rise['ODMR'])<=0.05 and abs(rise['OAMR'])<=0.05:
                index = 'D'
            else:
                loc = rise.nsmallest(1).index.values
                if loc == 'OHMR': index = 'H'
                elif loc == 'OAMR': index = 'A'
                else: index = 'D'
            playing_stat.loc[n, 'mean_idx'] = index
        return playing_stat

    # 计算欧赔标准差所指示值
    def oz_odds_std_index(self, playing_stat):
        playing_stat['std_idx'] = playing_stat['oz_away9_mean']
        oz_meanstd = playing_stat[['oz_home9_mean', 'oz_draw9_mean','oz_away9_mean','oz_home9_std', 'oz_draw9_std','oz_away9_std']]
        for n in range(len(oz_meanstd)):
            meanstd = oz_meanstd.iloc[n]
            if meanstd['oz_home9_mean']<=3.5 and meanstd['oz_draw9_mean']<=3.5 and meanstd['oz_away9_mean']<=3.5:  ##赔率小于3.5，求标准差最大的那个
                loc = meanstd[['oz_home9_std', 'oz_draw9_std','oz_away9_std']].nlargest(1).index.values
                if loc == 'oz_home9_std': index = 'H'
                elif loc == 'oz_draw9_std': index = 'D'
                else: index = 'A'
            else:
                loc = meanstd[['oz_home9_std', 'oz_draw9_std', 'oz_away9_std']].nsmallest(1).index.values   ## 只要有一个赔率大于3.5，求标准差最小的那个
                if loc == 'oz_home9_std': index = 'H'
                elif loc == 'oz_draw9_std': index = 'D'
                else: index = 'A'
            playing_stat.loc[n, 'std_idx'] = index
        return playing_stat

    # one-hot
    # we want continous vars that are integers for our input data, so lets remove any categorical vars
    def preprocess_features(self, data):
        ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
        # Initialize new output DataFrame
        output = pd.DataFrame(index=data.index)

        # Investigate each feature column for the data
        for col, col_data in data.iteritems():

            # If data type is categorical, convert to dummy variables
            if col_data.dtype == object:
                col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
            output = output.join(col_data)

        return output

    #获取主、客队的胜率，客队不败当作胜利
    def get_rates(self, playing_stat):
        playing_stat['h_win_rate'] = playing_stat['hh_nb_wins'] / playing_stat['hh_nb_games']
        playing_stat['a_win_rate'] = (playing_stat['aa_nb_wins'] + playing_stat['aa_nb_draws']) / playing_stat['aa_nb_games']
        playing_stat['diff_win_rate'] = playing_stat['h_win_rate'] - playing_stat['a_win_rate']

        return playing_stat

    #由欧赔中的初赔和终赔计算出一个特征值
    def get_oz_odds_value(self, playing_stat):
        temp_sum = 1/playing_stat['oz_home0_mean'] + 1/playing_stat['oz_draw0_mean'] + 1/playing_stat['oz_away0_mean']
        oz_home0_prob = (1/playing_stat['oz_home0_mean']) / temp_sum
        oz_draw0_prob = (1/playing_stat['oz_draw0_mean']) / temp_sum
        oz_away0_prob = (1/playing_stat['oz_away0_mean']) / temp_sum
        playing_stat['oz_odds_value0'] = (oz_home0_prob-oz_away0_prob) / oz_draw0_prob

        temp_sum = 1/playing_stat['oz_home9_mean'] + 1/playing_stat['oz_draw9_mean'] + 1/playing_stat['oz_away9_mean']
        oz_home9_prob = (1/playing_stat['oz_home9_mean']) / temp_sum
        oz_draw9_prob = (1/playing_stat['oz_draw9_mean']) / temp_sum
        oz_away9_prob = (1/playing_stat['oz_away9_mean']) / temp_sum
        playing_stat['oz_odds_value9'] = (oz_home9_prob-oz_away9_prob) / oz_draw9_prob
        playing_stat['Diff_OZ_Value'] = playing_stat['oz_odds_value9'] - playing_stat['oz_odds_value0']

        return playing_stat

    #获取球队实力参数，参照论文：“Beating the Bookies: Predicting the Outcome of Soccer Games”
    def get_general_coefficient(self, playing_stat, fraction):
        factor = 1/fraction  #赛果因子
        col_h = '{}general_coeff_h'.format(fraction)
        col_a = '{}general_coeff_a'.format(fraction)
        new_playing_stat = pd.DataFrame()
        playing_stat = playing_stat.sort_values(by=['season','lunci'], ascending=True)
        playing_stat.season = playing_stat.season.astype('str')
        playing_stat[col_h] = 0
        playing_stat[col_a] = 0

        season_lis = ['{}-{}'.format(i, i + 1) for i in range(2011, 2019)]  #赛季格式 2018-2019
        # season_lis = ['{}'.format(i) for i in range(2011, 2019)]  #赛季格式 2018
        for season in season_lis:
            data =  playing_stat[playing_stat.season == season]
            hometeam = data['hometeam']
            playteam = hometeam.drop_duplicates(keep='first')
            playteam = playteam.to_dict()
            coeff = dict([val, 1] for key, val in playteam.items())  #每队原始参数赋于1
            data = data.reset_index(drop=True)
            for n in range(len(data)):
                data.loc[n, col_h] = coeff[data.loc[n, 'hometeam']]
                data.loc[n, col_a] = coeff[data.loc[n, 'awayteam']]
                match = data.iloc[n]
                if match['FTR'] == 'H':
                    general_coeff_h = coeff[match['hometeam']] + factor * coeff[match['awayteam']]
                    general_coeff_a = coeff[match['awayteam']] - factor * coeff[match['awayteam']]
                elif match['FTR'] == 'D':
                    diff = coeff[match['hometeam']] - coeff[match['awayteam']]
                    general_coeff_h = coeff[match['hometeam']] - factor * diff
                    general_coeff_a = coeff[match['awayteam']] + factor * diff
                elif  match['FTR'] == 'A':
                    general_coeff_h = coeff[match['hometeam']] - factor * coeff[match['hometeam']]
                    general_coeff_a = coeff[match['awayteam']] + factor * coeff[match['hometeam']]

                coeff[match['hometeam']] = general_coeff_h
                coeff[match['awayteam']] = general_coeff_a

            new_playing_stat = new_playing_stat.append(data, ignore_index=True)

        return new_playing_stat

    def get_offensive_defensive_coefficient(self, playing_stat):
        factor_h = 1 / 8  # 攻击/防守因子
        factor_a = 1 / 5  # 攻击/防守因子
        new_playing_stat = pd.DataFrame()
        playing_stat = playing_stat.sort_values(by=['season','lunci'], ascending=True)
        playing_stat.season = playing_stat.season.astype('str')
        playing_stat['offensive_coeff_h'] = 0
        playing_stat['offensive_coeff_a'] = 0
        playing_stat['defensive_coeff_h'] = 0
        playing_stat['defensive_coeff_a'] = 0

        season_lis = ['{}-{}'.format(i, i + 1) for i in range(2011, 2019)]  #赛季格式 2018-2019
        # season_lis = ['{}'.format(i) for i in range(2011, 2019)]  #赛季格式 2018
        for season in season_lis:
            data =  playing_stat[playing_stat.season == season]
            hometeam = data['hometeam']
            playteam = hometeam.drop_duplicates(keep='first')
            playteam = playteam.to_dict()
            offensive_coeff = dict([val, 1] for key, val in playteam.items())  #每队原始参数赋于1
            defensive_coeff = dict([val, 1] for key, val in playteam.items())  # 每队原始参数赋于1

            data = data.reset_index(drop=True)
            for n in range(len(data)):
                data.loc[n, 'offensive_coeff_h'] = offensive_coeff[data.loc[n, 'hometeam']]
                data.loc[n, 'defensive_coeff_h'] = defensive_coeff[data.loc[n, 'hometeam']]
                data.loc[n, 'offensive_coeff_a'] = offensive_coeff[data.loc[n, 'awayteam']]
                data.loc[n, 'defensive_coeff_a'] = defensive_coeff[data.loc[n, 'awayteam']]
                match = data.iloc[n]
                res_score = match['res_score']
                res = re.split("--", res_score)
                home_goals = int(res[0])
                away_goals = int(res[1])
                avg_home_goals = data.loc[n, 'HTGS'] / data.loc[n, 'lunci']
                avg_away_goals = data.loc[n, 'ATGS'] / data.loc[n, 'lunci']

                if home_goals > 0:
                    offensive_coeff_h = offensive_coeff[match['hometeam']] + home_goals * factor_h * defensive_coeff[match['awayteam']]
                    defensive_coeff_a = defensive_coeff[match['awayteam']] - home_goals * factor_h * defensive_coeff[match['awayteam']]
                else:
                    offensive_coeff_h = offensive_coeff[match['hometeam']] - avg_home_goals * factor_a * offensive_coeff[match['hometeam']]
                    defensive_coeff_a = defensive_coeff[match['awayteam']] + avg_home_goals * factor_a * offensive_coeff[match['hometeam']]

                if away_goals > 0:
                    offensive_coeff_a = offensive_coeff[match['awayteam']] + away_goals * factor_a * defensive_coeff[match['hometeam']]
                    defensive_coeff_h = defensive_coeff[match['hometeam']] - away_goals * factor_a * defensive_coeff[match['hometeam']]
                else:
                    offensive_coeff_a = offensive_coeff[match['awayteam']] - avg_away_goals * factor_h  * offensive_coeff[match['awayteam']]
                    defensive_coeff_h = defensive_coeff[match['hometeam']] + avg_away_goals * factor_h  * offensive_coeff[match['awayteam']]

                offensive_coeff[match['hometeam']] = offensive_coeff_h
                offensive_coeff[match['awayteam']] = offensive_coeff_a
                defensive_coeff[match['hometeam']] = defensive_coeff_h
                defensive_coeff[match['awayteam']] = defensive_coeff_a

            new_playing_stat = new_playing_stat.append(data, ignore_index=True)

        return new_playing_stat





if __name__ == '__main__':  # 在win系统下必须要满足这个if条件
    league = 26
    loadfile = r"datasets/league/league_match_data({}).csv".format(league)
    savefile = r'datasets/final_dataset/final_dataset({}).csv'.format(league)
    playing_stat = pd.read_csv(loadfile, encoding="gbk")

    clean = dataClean()

    # playing_stat = clean.get_offensive_defensive_coefficient(playing_stat)
    playing_stat = clean.get_general_coefficient(playing_stat, 3)
    playing_stat = clean.get_general_coefficient(playing_stat, 7)

    playing_stat = clean.select_data(playing_stat)



    # 移除前三周比赛并移除多余特征
    playing_stat = playing_stat[playing_stat.lunci > 3]

    playing_stat = clean.add_form(playing_stat, 5)
    playing_stat = clean.get_3form_points(playing_stat)
    playing_stat = clean.get_win_loss_streak(playing_stat)
    playing_stat = clean.get_diff(playing_stat)
    playing_stat = clean.scale_by_week(playing_stat)
    # playing_stat = clean.oz_odds_mean_index(playing_stat)
    # playing_stat = clean.oz_odds_std_index(playing_stat)
    playing_stat = clean.get_rates(playing_stat)
    playing_stat = clean.get_oz_odds_value(playing_stat)




    # playing_stat['FTR'] = playing_stat.FTR.apply(clean.only_hw)
    # playing_stat['FTRR'] = playing_stat.FTRR.apply(clean.only_hw)

    # playing_stat = playing_stat.drop(['season', 'lunci', 'hometeam','awayteam', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HLP', 'ALP',
    #                                  'HTFormPts', 'ATFormPts', 'VTFormPtsStr', 'HTFormPtsStr', 'ATFormPtsStr', 'HM4','HM5', 'AM4', 'AM5',
    #                                   'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std',
    #                                   'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean',
    #                                   'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std'], axis=1)

    # Testing set
    # playing_stat_test = playing_stat[1500:]
    # playing_stat = playing_stat[:]
    playing_stat.to_csv(savefile, encoding = "gbk", index=None)
    # playing_stat_test.to_csv(loc + "test.csv", index=None)




