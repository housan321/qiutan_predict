# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:43:56 2019

@author: ASUS
"""

import os,sys,time
import pandas as pd
from multiprocessing import Process
from scrapy import cmdline
from qiutan.db_sql import MySql
from predict import prediction
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

def spider_get_data():
    print('爬虫进程开始运行...')
    cmdline.execute('scrapy crawl predict'.split())


def delete_database(db):
    #1 先清除旧数据
    del_sql1 = 'truncate table all_match_score;'
    del_sql2 = 'truncate table all_match_oz_odds;'
    del_sql3 = 'truncate table all_match_az_odds;'
    del_sql4 = 'truncate table taday_matchs;'
    db.del_item(del_sql1)
    db.del_item(del_sql2)
    db.del_item(del_sql3)
    db.del_item(del_sql4)

#3 合并数据，保存到文件
def union_save_database(db):
    print('合并数据库...')
    # union_sql = 'insert into taday_matchs (league,season,bs_num_id,lunci,hometeam, awayteam,bs_time,FTR,FTRR,h_win,h_draw,h_lost,a_win,a_draw,a_lost,HTGS,ATGS,HTGC,ATGC,HTGD,ATGD,HTP,ATP,HomeLP,AwayLP,VTFormPtsStr,HTFormPtsStr,ATFormPtsStr,' \
    #             ' oz_home0_mean, oz_draw0_mean, oz_away0_mean, oz_home9_mean, oz_draw9_mean, oz_away9_mean, oz_home0_std, oz_draw0_std, oz_away0_std, oz_home9_std, oz_draw9_std, oz_away9_std, ' \
    #             'az_home0_mean, az_size0_mean, az_away0_mean, az_home9_mean, az_size9_mean, az_away9_mean, az_home0_std, az_size0_std, az_away0_std, az_home9_std, az_size9_std, az_away9_std, az_value0, az_value9)' \
    #             ' SELECT score.league, score.season, score.bs_num_id, score.lunci, score.hometeam, score.awayteam, score.bs_time, score.FTR, score.FTRR, score.h_win, score.h_draw, score.h_lost, score.a_win, score.a_draw, score.a_lost, ' \
    #             'score.HTGS, score.ATGS, score.HTGC, score.ATGC, score.HTGD, score.ATGD, score.HTP, score.ATP, score.HomeLP, score.AwayLP, score.VTFormPtsStr, score.HTFormPtsStr, score.ATFormPtsStr, ' \
    #             'oz.oz_home0_mean, oz.oz_draw0_mean, oz.oz_away0_mean, oz.oz_home9_mean, oz.oz_draw9_mean, oz.oz_away9_mean, oz.oz_home0_std, oz.oz_draw0_std, oz.oz_away0_std, oz.oz_home9_std, oz.oz_draw9_std, oz.oz_away9_std,' \
    #             ' az.az_home0_mean, az.az_size0_mean, az.az_away0_mean, az.az_home9_mean, az.az_size9_mean, az.az_away9_mean, az.az_home0_std, az.az_size0_std, az.az_away0_std, az.az_home9_std, az.az_size9_std, az.az_away9_std, az.az_value0, az.az_value9 ' \
    #             'FROM all_match_score score JOIN  all_match_oz_odds oz  ON score.bs_num_id=oz.bs_num_id JOIN all_match_az_odds az ON score.bs_num_id=az.bs_num_id order by bs_time;'

    union_sql = 'insert into taday_matchs (league,season,bs_num_id,lunci,hometeam, awayteam,bs_time,FTR,FTRR,' \
                'h_nb_wins,h_nb_draws,h_nb_losts,HTGS,HTGC,HTGD,HTP,HLP,hh_nb_games,hh_nb_wins,hh_nb_draws,hh_nb_losts,HHTGS,HHTGC,HHTGD,HHTP,HHLP,' \
                'a_nb_wins,a_nb_draws,a_nb_losts,ATGS,ATGC,ATGD,ATP,ALP,aa_nb_games,aa_nb_wins,aa_nb_draws,aa_nb_losts,AATGS,AATGC,AATGD,AATP,AALP,VTFormPtsStr,HTFormPtsStr,ATFormPtsStr,' \
                'oz_home0_mean, oz_draw0_mean, oz_away0_mean, oz_home9_mean, oz_draw9_mean, oz_away9_mean, oz_home0_std, oz_draw0_std, oz_away0_std, oz_home9_std, oz_draw9_std, oz_away9_std, ' \
                'az_home0_mean, az_size0_mean, az_away0_mean, az_home9_mean, az_size9_mean, az_away9_mean, az_home0_std, az_size0_std, az_away0_std, az_home9_std, az_size9_std, az_away9_std, az_value0, az_value9)' \
                ' SELECT sc.league, sc.season, sc.bs_num_id, sc.lunci, sc.hometeam, sc.awayteam, sc.bs_time, sc.FTR, sc.FTRR, ' \
                'sc.h_nb_wins, sc.h_nb_draws, sc.h_nb_losts, sc.HTGS, sc.HTGC, sc.HTGD, sc.HTP, sc.HLP, sc.hh_nb_games, sc.hh_nb_wins, sc.hh_nb_draws, sc.hh_nb_losts, sc.HHTGS, sc.HHTGC, sc.HHTGD, sc.HHTP, sc.HHLP,' \
                'sc.a_nb_wins, sc.a_nb_draws, sc.a_nb_losts, sc.ATGS, sc.ATGC, sc.ATGD, sc.ATP, sc.ALP, sc.aa_nb_games, sc.aa_nb_wins, sc.aa_nb_draws, sc.aa_nb_losts, sc.AATGS, sc.AATGC, sc.AATGD, sc.AATP, sc.AALP, sc.VTFormPtsStr, sc.HTFormPtsStr, sc.ATFormPtsStr,' \
                'oz.oz_home0_mean, oz.oz_draw0_mean, oz.oz_away0_mean, oz.oz_home9_mean, oz.oz_draw9_mean, oz.oz_away9_mean, oz.oz_home0_std, oz.oz_draw0_std, oz.oz_away0_std, oz.oz_home9_std, oz.oz_draw9_std, oz.oz_away9_std, ' \
                'az.az_home0_mean, az.az_size0_mean, az.az_away0_mean, az.az_home9_mean, az.az_size9_mean, az.az_away9_mean, az.az_home0_std, az.az_size0_std, az.az_away0_std, az.az_home9_std, az.az_size9_std, az.az_away9_std, az.az_value0, az.az_value9 ' \
                'FROM all_match_score sc JOIN all_match_oz_odds oz ON sc.bs_num_id = oz.bs_num_id JOIN all_match_az_odds az ON sc.bs_num_id = az.bs_num_id  order by bs_time;'
    db.union_item(union_sql)

    if (os.path.exists('D:/qiutan_predict/prediction/datasets/taday_matchs.csv')):
        os.remove('D:/qiutan_predict/prediction/datasets/taday_matchs.csv')
    print('导出数据库...')
    # save_sql = "select * into outfile \'D:/qiutan_predict/prediction/datasets/taday_matchs.csv\'character set gbk fields terminated by ',' lines terminated by '\n' " \
    #            "from(select 'league', 'season', 'bs_num_id', 'lunci', 'hometeam', 'awayteam', 'bs_time', 'FTR','FTRR', 'h_win', 'h_draw', 'h_lost', 'a_win', 'a_draw', 'a_lost'," \
    #            " 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTGD', 'ATGD', 'HTP', 'ATP', 'HomeLP', 'AwayLP', 'VTFormPtsStr', 'HTFormPtsStr', 'ATFormPtsStr', " \
    #            " 'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std', 'oz_home9_std', 'oz_draw9_std', 'oz_away9_std'," \
    #            " 'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean', 'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std', 'az_value0', 'az_value9'" \
    #            " UNION SELECT league, season, bs_num_id, lunci, hometeam, awayteam, bs_time, FTR, FTRR, h_win, h_draw, h_lost, a_win, a_draw, a_lost, HTGS, ATGS, HTGC, ATGC, HTGD, ATGD, HTP, ATP, HomeLP, AwayLP, VTFormPtsStr, HTFormPtsStr, ATFormPtsStr," \
    #            " oz_home0_mean, oz_draw0_mean, oz_away0_mean, oz_home9_mean, oz_draw9_mean, oz_away9_mean, oz_home0_std, oz_draw0_std, oz_away0_std, oz_home9_std, oz_draw9_std, oz_away9_std," \
    #            " az_home0_mean, az_size0_mean, az_away0_mean, az_home9_mean, az_size9_mean, az_away9_mean, az_home0_std, az_size0_std, az_away0_std, az_home9_std, az_size9_std, az_away9_std, az_value0, az_value9 from taday_matchs) a ;"

    save_sql = "select * into outfile \'D:/qiutan_predict/prediction/datasets/taday_matchs.csv\'character set gbk fields terminated by ',' lines terminated by '\n' " \
               "from(select 'league','season','bs_num_id','lunci','hometeam','awayteam','bs_time','FTR','FTRR'," \
               " 'h_nb_wins','h_nb_draws','h_nb_losts','HTGS','HTGC','HTGD','HTP','HLP','hh_nb_games','hh_nb_wins','hh_nb_draws','hh_nb_losts','HHTGS','HHTGC','HHTGD','HHTP','HHLP'," \
               " 'a_nb_wins','a_nb_draws','a_nb_losts','ATGS','ATGC','ATGD','ATP','ALP','aa_nb_games','aa_nb_wins','aa_nb_draws','aa_nb_losts','AATGS','AATGC','AATGD','AATP','AALP','VTFormPtsStr','HTFormPtsStr','ATFormPtsStr'," \
               " 'oz_home0_mean', 'oz_draw0_mean', 'oz_away0_mean', 'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean', 'oz_home0_std', 'oz_draw0_std', 'oz_away0_std', 'oz_home9_std', 'oz_draw9_std', 'oz_away9_std'," \
               " 'az_home0_mean', 'az_size0_mean', 'az_away0_mean', 'az_home9_mean', 'az_size9_mean', 'az_away9_mean', 'az_home0_std', 'az_size0_std', 'az_away0_std', 'az_home9_std', 'az_size9_std', 'az_away9_std', 'az_value0', 'az_value9' " \
               "UNION SELECT league,season,bs_num_id,lunci,hometeam,awayteam,bs_time,FTR,FTRR," \
               "h_nb_wins,h_nb_draws,h_nb_losts,HTGS,HTGC,HTGD,HTP,HLP,hh_nb_games,hh_nb_wins,hh_nb_draws,hh_nb_losts,HHTGS,HHTGC,HHTGD,HHTP,HHLP," \
               "a_nb_wins,a_nb_draws,a_nb_losts,ATGS,ATGC,ATGD,ATP,ALP,aa_nb_games,aa_nb_wins,aa_nb_draws,aa_nb_losts,AATGS,AATGC,AATGD,AATP,AALP,VTFormPtsStr,HTFormPtsStr,ATFormPtsStr," \
               "oz_home0_mean, oz_draw0_mean, oz_away0_mean, oz_home9_mean, oz_draw9_mean, oz_away9_mean, oz_home0_std, oz_draw0_std, oz_away0_std, oz_home9_std, oz_draw9_std, oz_away9_std, " \
               "az_home0_mean, az_size0_mean, az_away0_mean, az_home9_mean, az_size9_mean, az_away9_mean, az_home0_std, az_size0_std, az_away0_std, az_home9_std, az_size9_std, az_away9_std, az_value0, az_value9 from taday_matchs) a ;"
    db.save_to_csv(save_sql)


def predict_match():
    loc = r"D:\qiutan_predict\prediction\\"
    loadfile = loc + r"datasets\taday_matchs.csv"
    pred_res = pd.DataFrame()
    taday_matchs = pd.read_csv(loadfile, encoding="gbk")
    match_info = taday_matchs[['league', 'hometeam', 'awayteam', 'bs_time']]

    predict = prediction()
    X_all = predict.extract_feature(loadfile)

    for n in range(len(match_info)):
        data = X_all[n:n+1]
        league = match_info.league[n]
        model_file = r'./prediction/model/final_model/xgboost_joblib({}final).dat'.format(league)
        if (os.path.exists(model_file)):
            xgboost_model = joblib.load(model_file)
        else: xgboost_model = joblib.load(r'./prediction/model/final_model/xgboost_joblib(西甲final).dat') ##如果没有该轮赛的预测器，则默认用西甲轮赛预测器
        y_pred = xgboost_model.predict(data)
        res = pd.DataFrame(y_pred, columns=['y_pred'])
        # pred_res = pd.concat([info, res], axis=1)
        pred_res = pred_res.append(res, ignore_index=True)
    pred_result = pd.concat([match_info, pred_res], axis=1)
    nowtime = time.strftime('%Y%m%d_%H_%M', time.localtime())
    pred_result.to_csv("./prediction/datasets/predResult{}.csv".format(nowtime), encoding = "gbk", index=None)




if __name__=='__main__':#在win系统下必须要满足这个if条件
    db = MySql('localhost', 'root', '123456', 'qiutan', 3306)

    # download = True
    download = False

    if download:
        #1 先清除旧数据
        delete_database(db)

        #2 开启爬虫进程
        print('%',os.getpid())#主进程号
        spi = Process(target=spider_get_data)#创建子进程对象，并传参args=(X,),注意只传一个参数的时候必须在参数后加逗号
        spi.start()
        spi.join()
        print('爬虫进程结束...')
        print('%', os.getppid())#主进程的父进程

        #3 合并数据，保存到文件
        union_save_database(db)

    #4 预测
    predict_match()
    print('完成预测')




