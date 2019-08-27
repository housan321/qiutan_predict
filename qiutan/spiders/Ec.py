# -*- coding: utf-8 -*-
import scrapy
import re, time
from datetime import timedelta, datetime
import pandas as pd
from qiutan.items import SaichengItem
from qiutan.items import Team_DataItem
from qiutan.items import Member_Data_New_Item
from qiutan.items import Member_Data_Old_Item

from qiutan.items import Match_Score_New_Item
from qiutan.items import Match_OZ_Odds_New_Item
from qiutan.items import Match_AZ_Odds_New_Item


class EcSpider(scrapy.Spider):
    name = 'predict'
    allowed_domains = ['zq.win007.com', 'bf.win007.com']

    leagueId = {'英超': '36', '西甲': '31', '意甲': '34', '德甲': '8', '法甲': '11',
                 '苏超': '29', '葡超': '23', '挪超': '22', '瑞典超': '26',  '荷甲': '16','俄超':'10',
                '英冠': '37', '德乙':'9',
                '中超': '60', '日职联': '25', '日职乙': '284', '韩K联': '15',
                '美职业': '21', '巴西甲': '4'}
    subleagueId = {'英超': '', '西甲': '', '意甲': '', '德甲': '', '法甲': '',
                   '苏超': '', '葡超': '_1123', '挪超': '', '瑞典超': '_431', '荷甲': '_98','俄超':'_591',
                   '英冠': '_87','德乙':'_132',
                   '中超': '', '日职联': '_943', '日职乙': '_808', '韩K联': '_313',
                   '美职业': '_165', '巴西甲': ''}

    # 将不同年份url交给Scheduler
    def start_requests(self):
        hour = time.strftime('%H', time.localtime())
        if hour <= '09':
            yesterday = datetime.today() + timedelta(-1)
            re = yesterday.strftime('%Y%m%d')
        else:
            re = time.strftime('%Y%m%d', time.localtime())  # 2019042509

        base_url = 'http://www.win0168.com/football/Next_{}.htm'
        req_base = scrapy.Request(base_url.format(re), callback=self.my_parse)
        yield req_base

    def team_data_id(self, response):
        # 获取每个队伍的id和队名
        pat = re.compile("\[(\d+),'(.*?)'")
        ballteam = pat.findall(response.text)[1:]
        lis_all_team = []
        for item in ballteam:
            lis_all_team.append(item[0])
            lis_all_team.append(item[-1])
        return lis_all_team

    # 表2 全部轮次的数据表
    def my_parse(self, response):
        # 获取赛季名称
        season = '2019-2020'

        ball_lunci_team = response.xpath("//table[@id='table_live']/tr")
        num = 0
        # 根据38轮遍历每一小轮
        for n in range(1, len(ball_lunci_team)):
            # 每小页数据
            item = SaichengItem()
            # # 每轮次的10条数据
            eve_turn = ball_lunci_team[n]
            eve_turn_team = eve_turn.xpath("./td")

            league = eve_turn_team[0].xpath("./font/text()").extract()[0]
            bs_time = eve_turn_team[1].xpath("./text()").extract()[0]
            hometeam = eve_turn_team[3].xpath("./text()").extract()[0]
            awayteam = eve_turn_team[5].xpath("./text()").extract()[0]
            id_str = eve_turn_team[7].xpath("./a/@onclick").extract()[0]
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            bs_num_id = re.findall(pattern, id_str)[0]

            # 只预测常见的轮赛
            if league not in self.leagueId.keys(): continue

            #1 拼接每个比赛详细分析 url http://zq.win007.com/analysis/851859cn.htm
            url = 'http://zq.win007.com/analysis/{}cn.htm'.format(bs_num_id)
            # url = 'http://zq.win007.com/analysis/404801cn.htm'
            req = scrapy.Request(url, callback=self.bs_score, errback=self.bs_resquest_err)
            req.meta['err_id'] = '001'
            req.meta['league'] = league
            req.meta['season'] = season
            req.meta['bs_num_id'] = bs_num_id
            req.meta['FTR'] = 'UNKNOWN'
            req.meta['FTRR'] = 'UNKNOWN'
            req.meta['hometeam'] = hometeam
            req.meta['awayteam'] = awayteam
            req.meta['bs_time'] = bs_time
            yield req

            #2 拼接每个比赛欧盘赔率 url http://1x2d.win007.com/1130517.js
            # # 2013-08-17 ,2014-5-12 老版页面  判断年份 保存版本
            # if item['bs_time'] < '2014-05-12 0:00':
            url = 'http://1x2d.win007.com/{}.js'.format(bs_num_id)
            req = scrapy.Request(url, dont_filter=True, callback=self.bs_odds_oz, errback=self.bs_resquest_err)
            req.meta['err_id'] = '002'
            req.meta['league'] = league
            req.meta['season'] = season
            req.meta['bs_num_id'] = bs_num_id
            req.meta['hometeam'] = hometeam
            req.meta['awayteam'] = awayteam
            yield req

            #3 拼接每个比赛亚盘赔率 url http://vip.win007.com/AsianOdds_n.aspx?id=987100
            # # 2013-08-17 ,2014-5-12 老版页面  判断年份 保存版本
            # if item['bs_time'] < '2014-05-12 0:00':
            url = 'http://vip.win007.com/AsianOdds_n.aspx?id={}'.format(bs_num_id)
            # url = 'http://vip.win007.com/AsianOdds_n.aspx?id=851594'
            req = scrapy.Request(url, dont_filter=True, callback=self.bs_odds_az, errback=self.bs_resquest_err)
            req.meta['err_id'] = '003'
            req.meta['league'] = league
            req.meta['season'] = season
            req.meta['bs_num_id'] = bs_num_id
            req.meta['hometeam'] = hometeam
            req.meta['awayteam'] = awayteam
            yield req


        # team_url = 'http://zq.win007.com/jsData/teamInfo/teamDetail/tdl{}.js?version={}'
        # # 根据 偶数索引 取 球队id
        # for i in range(len(lis_all_team)):
        #     if i % 2 == 0:
        #         url = team_url.format(lis_all_team[i], response.meta['re'])
        #         req = scrapy.Request(url, callback=self.team_data)
        #         # 加上防盗链获取接口
        #         req.meta['Referer'] = 'http://zq.win007.com/cn/team/Summary/{}.html'.format(lis_all_team[i])
        #         yield req


    # 请求失败的处理
    def bs_resquest_err(self, response):
        bs_num_id = response.meta['bs_num_id']
        err_id = response.meta['err_id']
        if err_id == '001':
            print('match' + bs_num_id + ' resquest score page failure！' )
        elif err_id == '002':
            print('match' + bs_num_id + ' resquest oz odds page failure！')
        elif err_id == '003':
            print('match' + bs_num_id + ' resquest az odds page failure！')

    # 主、客队进失球，积分、排名、近5场赛果等: 新版
    def bs_score(self, response):
        league = response.meta['league']
        season = response.meta['season']
        bs_num_id = response.meta['bs_num_id']
        hometeam = response.meta['hometeam']
        awayteam = response.meta['awayteam']
        FTR = response.meta['FTR']
        FTRR = response.meta['FTRR']
        bs_time = response.meta['bs_time']

        print(season, bs_num_id, response.status)

        # 实例化Item
        item = Match_Score_New_Item()
        if season > '2013-2014':
        # if season >= '2014':
            table_num = 1
        else: table_num = 0

        tables = response.xpath("//*[text()='联赛积分排名']/../../../..//table")
        if not tables:
            tables = response.xpath("//*[text()='联赛积分排名']/../../../../..//table")
        home_table = tables[table_num].xpath('./tr/td/text()').extract()
        away_table = tables[table_num+1].xpath('./tr/td/text()').extract()
        if not home_table: #如果没有数据，是因为table的位置不对应
            home_table = tables[table_num+1].xpath('./tr/td/text()').extract()
            away_table = tables[table_num+2].xpath('./tr/td/text()').extract()
        # elif season < '2012-2013':
        # # elif season < '2012':
        #     tables = response.xpath("//*[text()='联赛积分排名']/../../../../..//table")
        #     home_table = tables[1].xpath('./tr/td/text()').extract()
        #     away_table = tables[2].xpath('./tr/td/text()').extract()

        # tables = response.xpath("//*[text()='联赛积分排名']/../../../../..//table")
        # home_table = tables[0].xpath('./tr/td/text()').extract()
        # away_table = tables[1].xpath('./tr/td/text()').extract()

        VTFormPtsStr = self.get_VS_result(response, 'v_data.*?\[(\[.*?\])\];')
        HTFormPtsStr = self.get_VS_result(response, 'h_data.*?\[(\[.*?\])\];')
        ATFormPtsStr = self.get_VS_result(response, 'a_data.*?\[(\[.*?\])\];')

        item['league'] = league
        item['season'] = season
        item['bs_num_id'] = bs_num_id
        item['lunci'] = int(home_table[12])
        item['hometeam'] = hometeam
        item['awayteam'] = awayteam
        item['bs_time'] = bs_time
        item['FTR'] = FTR
        item['FTRR'] = FTRR

        item['h_nb_wins'] = home_table[13]
        item['h_nb_draws'] = home_table[14]
        item['h_nb_losts'] = home_table[15]
        item['HTGS'] = home_table[16]
        item['HTGC'] = home_table[17]
        item['HTGD'] = home_table[18]
        item['HTP'] = home_table[19]
        item['HLP'] = home_table[20]
        item['hh_nb_games'] = home_table[22]
        item['hh_nb_wins'] = home_table[23]
        item['hh_nb_draws'] = home_table[24]
        item['hh_nb_losts'] = home_table[25]
        item['HHTGS'] = home_table[26]
        item['HHTGC'] = home_table[27]
        item['HHTGD'] = home_table[28]
        item['HHTP'] = home_table[29]
        item['HHLP'] = home_table[30]

        item['a_nb_wins'] = away_table[13]
        item['a_nb_draws'] = away_table[14]
        item['a_nb_losts'] = away_table[15]
        item['ATGS'] = away_table[16]
        item['ATGC'] = away_table[17]
        item['ATGD'] = away_table[18]
        item['ATP'] = away_table[19]
        item['ALP'] = away_table[20]
        item['aa_nb_games'] = away_table[32]
        item['aa_nb_wins'] = away_table[33]
        item['aa_nb_draws'] = away_table[34]
        item['aa_nb_losts'] = away_table[35]
        item['AATGS'] = away_table[36]
        item['AATGC'] = away_table[37]
        item['AATGD'] = away_table[38]
        item['AATP'] = away_table[39]
        item['AALP'] = away_table[40]

        item['VTFormPtsStr'] = VTFormPtsStr
        item['HTFormPtsStr'] = HTFormPtsStr
        item['ATFormPtsStr'] = ATFormPtsStr
        yield item


    # 本场比赛赔率数据: 新版
    def bs_odds_oz(self, response):
        company_id = ['281', '115', '82', '173', '81', '90', '71', '104',
                      '16', '18', '976', '255', '545', '80', '474', '499']
        oz_odds = pd.DataFrame(columns=['home0', 'draw0', 'away0', 'home9', 'draw9', 'away9'])
        odds = pd.Series(index=['home0', 'draw0', 'away0', 'home9', 'draw9', 'away9'])
        list_id = 0

        league = response.meta['league']
        season = response.meta['season']
        bs_num_id = response.meta['bs_num_id']

        # 实例化Item
        item = Match_OZ_Odds_New_Item()
        result = re.findall(r'var game=Array(.*?);', response.text)
        if len(result) < 1:
            result = re.findall(r'game=Array(.*?);', response.text)
        result = re.findall(r"\(\"(.*)\"\)", result[0])
        result = re.split("\",\"", result[0])

        item['league'] = league
        item['season'] = season
        item['bs_num_id'] = bs_num_id

        for each in result:
            res = re.split("\|", each)
            if res[0] in company_id:
                odds[0] = res[3]
                odds[1] = res[4]
                odds[2] = res[5]
                if res[10]: # 即时盘有数据
                    odds[3] = res[10]
                    odds[4] = res[11]
                    odds[5] = res[12]
                else:
                    odds[3] = res[3]
                    odds[4] = res[4]
                    odds[5] = res[5]
                oz_odds = oz_odds.append(odds.T, ignore_index=True)

        odds_mean = oz_odds.mean()
        odds_std = oz_odds.std()

        item['oz_home0_mean'] = float(odds_mean[0])
        item['oz_draw0_mean'] = float(odds_mean[1])
        item['oz_away0_mean'] = float(odds_mean[2])
        item['oz_home9_mean'] = float(odds_mean[3])
        item['oz_draw9_mean'] = float(odds_mean[4])
        item['oz_away9_mean'] = float(odds_mean[5])
        item['oz_home0_std'] = float(odds_std[0])
        item['oz_draw0_std'] = float(odds_std[1])
        item['oz_away0_std'] = float(odds_std[2])
        item['oz_home9_std'] = float(odds_std[3])
        item['oz_draw9_std'] = float(odds_std[4])
        item['oz_away9_std'] = float(odds_std[5])

        yield item


    # 本场比赛赔率数据: 新版
    def bs_odds_az(self, response):
        league = response.meta['league']
        season = response.meta['season']
        bs_num_id = response.meta['bs_num_id']

        # 实例化Item
        item = Match_AZ_Odds_New_Item()

        az_odds = pd.DataFrame(columns=['az_home0', 'az_size0', 'az_away0', 'az_home9', 'az_size9', 'az_away9'])
        odds = pd.Series(index=['az_home0', 'az_size0', 'az_away0', 'az_home9', 'az_size9', 'az_away9'])
        az_value = pd.Series(index=['value0', 'value9'])

        item['league'] = league
        item['season'] = season
        item['bs_num_id'] = bs_num_id

        td_list = response.xpath("//table[@id=\"odds\"]/tr/td")
        flag = False
        for index, td in enumerate(td_list):
            # cid = td.xpath("./span[@class='jia']")
            cid = td.xpath("./span[@companyid]")
            if cid:
                have_value = td_list[index+1].xpath("./text()")
                if not have_value : continue  ### 没有赔率数据就继续找下家
                az_home0 = td_list[index+1].xpath("./text()").extract()[0]
                az_size0 = td_list[index+2].xpath("./@goals").extract()[0]
                az_away0 = td_list[index+3].xpath("./text()").extract()[0]
                az_home9 = td_list[index+7].xpath("./text()").extract()[0]
                az_size9 = td_list[index+8].xpath("./@goals").extract()[0]
                az_away9 = td_list[index+9].xpath("./text()").extract()[0]
                odds[0] = float(az_home0)
                odds[1] = float(az_size0)
                odds[2] = float(az_away0)
                odds[3] = float(az_home9)
                odds[4] = float(az_size9)
                odds[5] = float(az_away9)
                az_odds = az_odds.append(odds.T, ignore_index=True)
                if not flag:
                    az_value[0], az_value[1] = self.convert_az_odds(odds) #只取一家赔率
                    flag = True

        odds_mean = az_odds.mean()
        odds_std = az_odds.std()

        item['az_home0_mean'] = float(odds_mean[0])
        item['az_size0_mean'] = float(odds_mean[1])
        item['az_away0_mean'] = float(odds_mean[2])
        item['az_home9_mean'] = float(odds_mean[3])
        item['az_size9_mean'] = float(odds_mean[4])
        item['az_away9_mean'] = float(odds_mean[5])
        item['az_home0_std'] = float(odds_std[0])
        item['az_size0_std'] = float(odds_std[1])
        item['az_away0_std'] = float(odds_std[2])
        item['az_home9_std'] = float(odds_std[3])
        item['az_size9_std'] = float(odds_std[4])
        item['az_away9_std'] = float(odds_std[5])
        item['az_value0'] = float(az_value[0])
        item['az_value9'] = float(az_value[1])

        yield item

    # 把亚盘转换成一个数值，代表强弱的表现
    def  convert_az_odds(self, odds):
        if odds[1] == 0:
            odds[1] = 0.1
        if odds[1] >= 0:
            value0 = odds[1] * (odds[2] / odds[0])
        else:
            value0 = odds[1] * (odds[0] / odds[2])
        if odds[4] == 0:
            odds[4] = 0.1
        if odds[4] >= 0:
            value9 = odds[4] * (odds[5] / odds[3])
        else:
            value9 = odds[4] * (odds[3] / odds[5])

        return value0, value9

    # 主队、客队近5场赛果，主、客队近5场对赛赛果
    def get_VS_result(self, response, pattern):
        TFormPtsStr = ''
        data = re.findall(pattern, response.text)
        if len(data) == 0:
            TFormPtsStr = 'DDDDD'
            return TFormPtsStr   #没有比赛
        data_list = re.findall(r'\[.*?\]', data[0])
        match_num = min(5, len(data_list)) #只取5场比赛
        for n in range(match_num):
            lis = data_list[n].strip('[|]').replace('\'', '').split(',')
            if len(lis) == 0:  # 如果没有找到数据，则按平局处理
                TFormPtsStr = 'DDDDD'
                return TFormPtsStr
            if lis[12] == '-1':
                TFormPtsStr = TFormPtsStr + 'L'
            elif lis[12] == '0':
                TFormPtsStr = TFormPtsStr + 'D'
            else:
                TFormPtsStr = TFormPtsStr + 'W'

        if len(TFormPtsStr) >= 5:
            TFormPtsStr = TFormPtsStr[:5]
        else:
            for i in range(5 - len(TFormPtsStr)): TFormPtsStr = TFormPtsStr + 'D'  # 如果对赛不够5场,以平局补够5场

        return TFormPtsStr














    # 每场比赛队员数据: 新版
    def bs_data_new(self, response):
        # 实例化Item
        item = Member_Data_New_Item()
        # 分别 取上下两个队伍的信息
        member_lis_tr_s = response.xpath('//div[@id="content"]/div[3]/table//tr[position()>2]')
        member_lis_tr_x = response.xpath('//div[@id="content"]/div[4]/table//tr[position()>2]')
        for member_lis in member_lis_tr_s:
            item['bs_num_id'] = response.meta['bs_num_id']
            item['team_id'] = response.meta['l_team_id']
            item['member_id'] = member_lis.xpath('./td[1]/text()').extract_first()
            item['member_name'] = member_lis.xpath('./td[2]/a//text()').extract_first().strip()
            item['position'] = member_lis.xpath('./td[3]/text()').extract_first().strip()
            item['shoot_d'] = member_lis.xpath('./td[4]/text()').extract_first()
            item['shoot_z'] = member_lis.xpath('./td[5]/text()').extract_first()
            item['key_ball'] = member_lis.xpath('./td[6]/text()').extract_first()
            item['guoren'] = member_lis.xpath('./td[7]/text()').extract_first()
            item['chuanq_count'] = member_lis.xpath('./td[8]/text()').extract_first()
            item['chuanq_succ'] = member_lis.xpath('./td[9]/text()').extract_first()
            item['passing'] = member_lis.xpath('./td[10]/text()').extract_first()
            item['hengchuanc'] = member_lis.xpath('./td[11]/text()').extract_first()
            item['success_zd'] = member_lis.xpath('./td[17]/text()').extract_first()
            item['body_jc'] = member_lis.xpath('./td[18]/text()').extract_first()
            item['score'] = member_lis.xpath('./td[30]/text()').extract_first()
            item['key_event'] = member_lis.xpath('./td[31]/a/img/@title').extract_first()
            yield item

        for member_lis in member_lis_tr_x:
            item['bs_num_id'] = response.meta['bs_num_id']
            item['team_id'] = response.meta['r_team_id']
            item['member_id'] = member_lis.xpath('./td[1]/text()').extract_first()
            item['member_name'] = member_lis.xpath('./td[2]/a/text()').extract_first().strip()
            item['position'] = member_lis.xpath('./td[3]/text()').extract_first().strip()
            item['shoot_d'] = member_lis.xpath('./td[4]/text()').extract_first()
            item['shoot_z'] = member_lis.xpath('./td[5]/text()').extract_first()
            item['key_ball'] = member_lis.xpath('./td[6]/text()').extract_first()
            item['guoren'] = member_lis.xpath('./td[7]/text()').extract_first()
            item['chuanq_count'] = member_lis.xpath('./td[8]/text()').extract_first()
            item['chuanq_succ'] = member_lis.xpath('./td[9]/text()').extract_first()
            item['passing'] = member_lis.xpath('./td[10]/text()').extract_first()
            item['hengchuanc'] = member_lis.xpath('./td[11]/text()').extract_first()
            item['success_zd'] = member_lis.xpath('./td[17]/text()').extract_first()
            item['body_jc'] = member_lis.xpath('./td[18]/text()').extract_first()
            item['score'] = member_lis.xpath('./td[30]/text()').extract_first()
            item['key_event'] = member_lis.xpath('./td[31]/a/img/@title').extract_first()

            yield item

    def bs_data_old(self, response):
        # 获取13年左边的阵容数据和后备数据,返回列表[含字符串,]
        member_lis_l1 = response.xpath("/html/body/table[1]/tr[1]/td[1]/table/tr[3]/td/a//text()").extract()
        member_lis_l2 = response.xpath("/html/body/table[1]/tr[1]/td[1]/table/tr[5]/td/a/text()").extract()
        # 获取13年右边的阵容数据和后备数据
        member_lis_r1 = response.xpath("/html/body/table[1]/tr[1]/td[3]/table/tr[3]/td/a/text()").extract()
        member_lis_r2 = response.xpath("/html/body/table[1]/tr[1]/td[3]/table/tr[5]/td/a/text()").extract()
        item = Member_Data_Old_Item()

        # 将阵容和后备列表合并
        member_lis_l = member_lis_l1 + member_lis_l2
        member_lis_r = member_lis_r1 + member_lis_r2
        # 遍历每个元组(球员号,球员名字)
        for member in member_lis_l:
            res = member.strip()
            member_list = re.findall('(\d+)\s?(.*)', res)[0]  # ('22', '雅斯科莱宁') ('11', '麦加')
            item['bs_num_id'] = response.meta['bs_num_id']
            item['team_id'] = response.meta['l_team_id']
            item['member_id'] = member_list[0]
            item['member_name'] = member_list[1]

            yield item

        for member in member_lis_r:
            res = member.strip()  # 1  切赫
            member_list = re.findall('(\d+)\s+(.*)', res)[0]  # ('17', '奥布莱恩')
            item['bs_num_id'] = response.meta['bs_num_id']
            item['team_id'] = response.meta['r_team_id']
            item['member_id'] = member_list[0]
            item['member_name'] = member_list[1]
            yield item

    # 球队信息
    def team_data(self, response):
        # 第一行数据
        teamDetail = re.findall('var teamDetail = \[(\d+.*)\]', response.text)
        teamDetail_lis = eval(teamDetail[0])
        # 获取教练
        var_coach = re.findall("var coach = \[\['\d+','','(.*?)','.*','.*',\d\]\];", response.text)
        item = Team_DataItem()
        #
        item['team_id'] = teamDetail_lis[0]
        item['team_name'] = teamDetail_lis[1]
        item['Eng_name'] = teamDetail_lis[3]
        item['team_city'] = teamDetail_lis[5]
        item['team_home'] = teamDetail_lis[8]
        item['build_team_time'] = teamDetail_lis[12]
        try:
            item['var_coach'] = var_coach[0]
        except:
            item['var_coach'] = 'NULL'

        # 球队特点
        item['team_youshi'] = str(re.findall('\[1,\d,"(.*?)\^', response.text))
        item['team_ruodian'] = str(re.findall('\[2,\d,"(.*?)\^', response.text))
        item['team_style'] = str(re.findall('\[3,\d,"(.*?)\^', response.text))
        team_stats_lis = re.findall('var countSum = \[\[(\'.*?)\]', response.text)[0]
        stats_tuple = eval(team_stats_lis)

        s = stats_tuple

        winrate = int(s[2]) / (int(s[2]) + int(s[3]) + int(s[4]))
        data = (s[2], s[3], s[4], winrate, s[5], s[6], s[7], s[8], s[9], (s[10]), s[11], (s[12]), s[13], s[14], s[24])
        str_stats = '全部:胜:%s,平:%s,负:%s,胜率:%.3f,犯规:%s,黄牌:%s,红牌:%s,' \
                    '控球率:%s,射门(射正):%s(%s),传球(成功):%s(%s),传球成功率:%s,过人次数:%s,评分:%s'
        item['team_stats'] = str_stats % (data)
        yield item

    def rangqiu(self, num_rang):
        if num_rang == '0':
            return '平手'
        elif num_rang == '0.25':
            return '平/半'
        elif num_rang == '0.5':
            return '半球'
        elif num_rang == '0.75':
            return '半/一'
        elif num_rang == '1':
            return '一球'
        elif num_rang == '1.25':
            return '一/球半'
        elif num_rang == '1.5':
            return '球半'
        elif num_rang == '1.75':
            return '半/二'
        elif num_rang == '2':
            return '二球'
        elif num_rang == '2.25':
            return '二/半'
        elif num_rang == '-0.25':
            return '*平/半'
        elif num_rang == '-0.5':
            return '*半球'
        elif num_rang == '-0.75':
            return '*半/一'
        elif num_rang == '-1':
            return '*一球'
        elif num_rang == '-1.25':
            return '*一/球半'
        elif num_rang == '-1.5':
            return '*球半'
        else:
            return '暂未收录'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        