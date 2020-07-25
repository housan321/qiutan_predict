#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality to transform the raw data into recency features.
Recency Features: https://link.springer.com/article/10.1007/s10994-018-5747-8

Created on Tue Apr  2 22:49:30 2019
@author: Juan Beleño
"""
import numpy as np
import tensorflow as tf

from math import exp, pow, fabs, log
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from collections import Counter

def convert_raw_data_to_features(match_data: dict) -> Tuple[dict, dict]:
    '''Transform the raw data from matches to features

    Args:
        match_data: The data about a soccer match.

    Returns:
        A tuple with features of the home and away teams.
    '''
    league = match_data['league']
    score_home_team = match_data['score_home_team']
    score_away_team = match_data['score_away_team']
    match_date = match_data['match_date']
    match_id = match_data['match_id']

    home_data = {
        'date': match_date,
        # 'match_id':match_id,
        'home_advantage': 1,
        'attacking_strength': score_home_team,
        'defensive_strength': score_away_team,
        # 'oz_0_mean': match_data['oz_home0_mean'], ##初盘欧赔赔率作为主队特征
        # 'oz_1_mean': match_data['oz_draw0_mean'],
        # 'oz_3_mean': match_data['oz_away0_mean'],
        # 'oz_0_std': match_data['oz_home0_std'],
        # 'oz_1_std': match_data['oz_draw0_std'],
        # 'oz_3_std': match_data['oz_away0_std']
    }

    away_data = {
        'date': match_date,
        # 'match_id': match_id,
        'home_advantage': 0,
        'attacking_strength': score_away_team,
        'defensive_strength': score_home_team,
        # 'oz_0_mean': match_data['oz_home9_mean'], ##收盘欧赔赔率作为客队特征
        # 'oz_1_mean': match_data['oz_draw9_mean'],
        # 'oz_3_mean': match_data['oz_away9_mean'],
        # 'oz_0_std': match_data['oz_home9_std'],
        # 'oz_1_std': match_data['oz_draw9_std'],
        # 'oz_3_std': match_data['oz_away9_std']
    }
    return (home_data, away_data)


def get_feature_names(
        team_type: str = 'home'
) -> Tuple[str, str, str, str, str, str, str, str, str]:
    '''Get feature names given a team type.

    Args:
        team_type: The team type: home or away.

    Returns:
        A tuple with feature names.
    '''
    attack_key = '{}_attacking_strength'.format(team_type)
    defense_key = '{}_defensive_strength'.format(team_type)
    opposition_key = '{}_strength_opposition'.format(team_type)
    h_advantage_key = '{}_home_advantage'.format(team_type)
    league_key = '{}_league'.format(team_type)
    month_key = '{}_month'.format(team_type)
    weekday_key = '{}_weekday'.format(team_type)
    fatigue_key = '{}_fatigue'.format(team_type)
    # wins_key = '{}_wins'.format(team_type)
    # draws_key = '{}_draws'.format(team_type)
    # loses_key = '{}_loses'.format(team_type)
    rating_key = '{}_pi_rating'.format(team_type)
    # Give more expressiveness power allowing super linear functions
    # Source: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45530.pdf
    # attack_key_x2 = '{}_attacking_strength_x2'.format(team_type)
    # defense_key_x2 = '{}_defensive_strength_x2'.format(team_type)
    wins_key_x2 = '{}_wins_x2'.format(team_type)
    draws_key_x2 = '{}_draws_x2'.format(team_type)
    loses_key_x2 = '{}_loses_x2'.format(team_type)
    goals_pred_key = '{}_goals_pred'.format(team_type)
    team_key = '{}_team_id'.format(team_type)
    # Match stats
    fouls_key = '{}_fouls'.format(team_type)
    yellow_cards_key = '{}_yellow_cards'.format(team_type)
    red_cards_key = '{}_red_cards'.format(team_type)
    offsides_key = '{}_offsides'.format(team_type)
    corners_key = '{}_corners'.format(team_type)
    saves_key = '{}_saves'.format(team_type)
    possession_key = '{}_possession'.format(team_type)
    shots_key = '{}_shots'.format(team_type)
    shots_on_goal_key = '{}_shots_on_goal'.format(team_type)
    '''fouls_key, yellow_cards_key, red_cards_key, offsides_key,
            corners_key, saves_key, possession_key, shots_key,
            shots_on_goal_key'''
    # return [attack_key, defense_key, opposition_key, h_advantage_key,
    #         league_key, month_key, weekday_key, fatigue_key, wins_key,
    #         draws_key, loses_key, rating_key, attack_key_x2,
    #         defense_key_x2, wins_key_x2, draws_key_x2, loses_key_x2,
    #         goals_pred_key, team_key]
    return [rating_key, goals_pred_key]


def get_pi_ratings(match: Dict, pi_ratings: Dict) -> Tuple[float, float]:
    '''Get pi ratings for each team

    Args:
        match: A dictionary with information of the team.
        pi_ratings: The pi-rating of each team per league.

    Returns:
        The home background rating for home team.
        The away background rating for away team.
    '''
    home_team_id = match['id_home_team']
    away_team_id = match['id_away_team']
    league = match['league']
    VTFormPtsStr = match['VTFormPtsStr']
    HTFormPtsStr = match['HTFormPtsStr']
    ATFormPtsStr = match['ATFormPtsStr']
    br_home_4_home_team = 0
    br_away_4_home_team = 0
    br_home_4_away_team = 0
    br_away_4_away_team = 0

    pr_home_4_home_team = 0
    pr_away_4_away_team = 0

    if home_team_id in pi_ratings:
        if league in pi_ratings[home_team_id]:
            br_home_4_home_team = pi_ratings[home_team_id][league]['home']
            br_away_4_home_team = pi_ratings[home_team_id][league]['away']

    if away_team_id in pi_ratings:
        if league in pi_ratings[away_team_id]:
            br_home_4_away_team = pi_ratings[away_team_id][league]['home']
            br_away_4_away_team = pi_ratings[away_team_id][league]['away']

    home_sita = get_sita2(HTFormPtsStr)
    away_sita = get_sita2(ATFormPtsStr)

    if home_sita > 1:
        pr_home_4_home_team = br_home_4_home_team + (0.01 * (home_sita - 1) / ((home_sita - 1) ** 2.5))
    elif home_sita < -1:
        pr_home_4_home_team = br_home_4_home_team + (-0.01 * (-home_sita - 1) / ((-home_sita - 1) ** 2.5))
    else:
        pr_home_4_home_team = br_home_4_home_team

    if away_sita > 1:
        pr_away_4_away_team = br_away_4_away_team + (0.01 * (away_sita - 1) / ((away_sita - 1) ** 2.5))
    elif away_sita < -1:
        pr_away_4_away_team = br_away_4_away_team + (-0.01 * (-away_sita - 1) / ((-away_sita - 1) ** 2.5))
    else:
        pr_away_4_away_team = br_away_4_away_team

    return pr_home_4_home_team, pr_away_4_away_team



def get_recency_features(
        team_id: int, league: int, time_steps: int,
        matches_per_team: dict, team_type: str = 'home',
        has_padding: bool = False
) -> dict:
    '''Get the features of the last n matches of a team

    Recency features are derived using the procedure explained in the paper
    "Incorporating domain knowledge in machine learning for soccer outcome
    prediction" by Berrar, D. et al.
    Source: https://link.springer.com/article/10.1007/s10994-018-5747-8

    Args:
        team_id: A team identifier.
        time_steps: The number of matches to be analyzed.
        matches_per_team: The raw data of all the matches in the data source.
        team_type: The team type: home or away
        has_padding: Wheter the features has padding of time_steps or not

    Returns:
        A dictionary with a list of features of the last n games of a team.
    '''
    last_features = []
    if team_id in matches_per_team:
        if league in matches_per_team[team_id]:
            last_features = matches_per_team[team_id][league][-time_steps:]
    features_names = get_feature_names(team_type)
    recency_features = {}
    for name in features_names:
        recency_features[name] = []
    '''
    [attack_key, defense_key, opposition_key, h_advantage_key,
    league_key, month_key, weekday_key, fatigue_key, wins_key,
    draws_key, loses_key, rating_key, attack_key_x2,
    defense_key_x2, wins_key_x2, draws_key_x2, loses_key_x2,
    fouls_key, yellow_cards_key, red_cards_key, offsides_key,
    corners_key, saves_key, possession_key, shots_key,
    shots_on_goal_key] = features_names
    '''
    # [attack_key, defense_key, opposition_key, h_advantage_key,
    # league_key, month_key, weekday_key, fatigue_key, wins_key,
    # draws_key, loses_key, rating_key, attack_key_x2,
    # defense_key_x2, wins_key_x2, draws_key_x2, loses_key_x2,
    # goals_pred_key, team_key] = features_names
    [rating_key, goals_pred_key] = features_names

    old_date = None
    wins = 0
    draws = 0
    loses = 0
    for index, match in enumerate(last_features):
        # By default, I'm going to set 720 hours as rest time when no data
        # available. The fatigue function is 1 over the number of hours of
        # resting between matches.
        # fatigue = 1.0/720
        # if old_date is not None:
        #     seconds_in_hour = 3600.0
        #     difference = (match['date'] - old_date).total_seconds()
        #     difference = int(difference / seconds_in_hour)
        #     if difference > 0:
        #         fatigue = 1.0 / difference
        #     else:
        #         fatigue = 1.0
        # old_date = match['date']

        # if index == (len(last_features) - 1):
        #     recency_features[attack_key].append(-16.0)
        #     recency_features[defense_key].append(-16.0)
        #     recency_features[attack_key_x2].append(-16.0)
        #     recency_features[defense_key_x2].append(-16.0)
        #     # Match Stats
        #     '''
        #     recency_features[fouls_key].append(-16.0)
        #     recency_features[yellow_cards_key].append(-16.0)
        #     recency_features[red_cards_key].append(-16.0)
        #     recency_features[offsides_key].append(-16.0)
        #     recency_features[corners_key].append(-16.0)
        #     recency_features[saves_key].append(-16.0)
        #     recency_features[possession_key].append(-16.0)
        #     recency_features[shots_key].append(-16.0)
        #     recency_features[shots_on_goal_key].append(-16.0)
        #     '''
        # else:
        #     recency_features[attack_key].append(float(match['attacking_strength']))
        #     recency_features[defense_key].append(float(match['defensive_strength']))
        #     recency_features[attack_key_x2].append(float(match['attacking_strength'])**2)
        #     recency_features[defense_key_x2].append(float(match['defensive_strength'])**2)
        #     # Match stats
        #     '''
        #     recency_features[fouls_key].append(float(match['fouls']))
        #     recency_features[yellow_cards_key].append(float(match['yellow_cards']))
        #     recency_features[red_cards_key].append(float(match['red_cards']))
        #     recency_features[offsides_key].append(float(match['offsides']))
        #     recency_features[corners_key].append(float(match['corners']))
        #     recency_features[saves_key].append(float(match['saves']))
        #     recency_features[possession_key].append(float(match['possession']))
        #     recency_features[shots_key].append(float(match['shots']))
        #     recency_features[shots_on_goal_key].append(float(match['shots_on_goal']))
        #     '''

        # recency_features[attack_key].append(float(match['attacking_strength']))
        # recency_features[defense_key].append(float(match['defensive_strength']))
        # recency_features[opposition_key].append(float(match['strength_opposition']))
        # recency_features[h_advantage_key].append(float(match['home_advantage']))


        # recency_features[league_key].append(float(match['league']))
        # recency_features[month_key].append(float(match['month']))
        # recency_features[weekday_key].append(float(match['weekday']))
        # recency_features[fatigue_key].append(float(fatigue))
        # recency_features[wins_key].append(float(wins))
        # recency_features[draws_key].append(float(draws))
        # recency_features[loses_key].append(float(loses))
        # recency_features[wins_key_x2].append(float(wins)**2)
        # recency_features[draws_key_x2].append(float(draws)**2)
        # recency_features[loses_key_x2].append(float(loses)**2)
        recency_features[rating_key].append(float(match['pi_rating']))
        recency_features[goals_pred_key].append(float(match['goals_pred']))
        # recency_features[team_key].append(float(team_id))

        # goal_difference = int(match['attacking_strength'] - match['defensive_strength'])
        # if goal_difference == 0:
        #     draws = draws + 1
        # elif goal_difference > 0:
        #     wins = wins + 1
        # else:
        #     loses = loses + 1

    if has_padding:
        # Source: https://arxiv.org/abs/1903.07288
        for key in recency_features:
            padding = [-16.0] * (time_steps - len(recency_features[key]))
            recency_features[key] = padding + recency_features[key]

    return recency_features


def get_odds_features(match: Dict) -> Tuple[float]:
    '''Get pi ratings for each team
    Args:
        match: A dictionary with information of the team.

    Returns:
        开盘赔率的平均值，标准差
        收盘赔率的平均值，标准差
    '''
    oz_home0_mean = match['oz_home0_mean']
    oz_draw0_mean = match['oz_draw0_mean']
    oz_away0_mean = match['oz_away0_mean']
    oz_home0_std = match['oz_home0_std']
    oz_draw0_std = match['oz_draw0_std']
    oz_away0_std = match['oz_away0_std']

    oz_home9_mean = match['oz_home9_mean']
    oz_draw9_mean = match['oz_draw9_mean']
    oz_away9_mean = match['oz_away9_mean']
    oz_home9_std = match['oz_home9_std']
    oz_draw9_std = match['oz_draw9_std']
    oz_away9_std = match['oz_away9_std']

    oz_odds0 = [oz_home0_mean, oz_draw0_mean, oz_away0_mean, oz_home0_std, oz_draw0_std, oz_away0_std]
    oz_odds9 = [oz_home9_mean, oz_draw9_mean, oz_away9_mean, oz_home9_std, oz_draw9_std, oz_away9_std]

    return oz_odds0, oz_odds9



def get_strength_opposition(
        team_id: int, league: int, time_steps: int, matches_per_team: dict
) -> float:
    '''Get the strength of opposition of a team

    Args:
        team_id: A team identifier.
        time_steps: The number of matches to be analyzed.
        matches_per_team: The raw data of all the matches in the data source.

    Returns:
        A number that represents the strength of a team based on the goal
        difference.
    '''
    last_features = []
    if team_id in matches_per_team:
        if league in matches_per_team[team_id]:
            last_features = matches_per_team[team_id][league][-time_steps:]
    sum_goal_difference = 0.0
    if len(last_features) == 0:
        return sum_goal_difference

    for match in last_features:
        goals_scored = match['attacking_strength']
        goals_conceded = match['defensive_strength']
        goal_difference = goals_scored - goals_conceded
        sum_goal_difference = sum_goal_difference + goal_difference
    return sum_goal_difference * 1.0 / len(last_features)


def preprocess_labels(labels: tf.Tensor, n_classes: int) -> tf.Tensor:
    '''Generate one-hot encoding for class labels.'''
    return tf.one_hot(labels, n_classes)


def preprocess_features(
        recency_features: list, time_steps: int
) -> list:
    '''Get predictive features in a tensorflow compatible format'''
    home_feature_names = get_feature_names('home')
    away_feature_names = get_feature_names('away')
    feature_sequence = []
    for step in range(0, time_steps):
        feature_item = np.array([])
        for name in home_feature_names:
            feature_item = np.append(feature_item, recency_features[name][step])

        for name in away_feature_names:
            feature_item = np.append(feature_item, recency_features[name][step])

        feature_sequence.append(feature_item)
    return np.array(feature_sequence)


def train_test_split(
        dataset: list, test_size: float = 0.2
) -> Tuple[list, list]:
    '''Split a dataset in two: training and test datasets.

    Args:
        dataset: A list of features.
        test_size: The proportion of size for a test dataset.

    Returns:
        Training and test dataset
    '''
    dataset_size = len(dataset)
    num_rows_test = int(dataset_size * test_size)
    np.random.shuffle(dataset)
    training, test = dataset[num_rows_test:], dataset[:num_rows_test]
    return (training, test)


def get_bld_rating_predictions(match: Dict, bld_ratings: Dict) -> Tuple[float, float]:
    '''Predict goals per team

    Source: https://link.springer.com/article/10.1007/s10994-018-5747-8

    Args:
        match: A dictionary with information of the team.
        bld_ratings: The BLD (Berrar, Lopes, Dubitzki) rating of each team per league.

    Returns:
        The goal predictions per team.
    '''
    (hatt, hdef, aatt, adef) = get_bld_rating_features(match, bld_ratings)

    alpha_home = 5
    beta_home = 0.277
    # gamma_home = 4.269
    gamma_home = -1.183
    home_goals_pred = alpha_home / (1 + exp(- beta_home * (hatt + adef) - gamma_home))

    alpha_away = 5
    beta_away = 0.434
    # gamma_away = 3.419
    gamma_away = -1.485
    away_goals_pred = alpha_away / (1 + exp(- beta_away * (aatt + hdef) - gamma_away))

    return (home_goals_pred, away_goals_pred)


def get_bld_rating_features(match: Dict, bld_ratings: Dict) -> Tuple[float, float, float, float]:
    '''Get rating features

    Source: https://link.springer.com/article/10.1007/s10994-018-5747-8

    Args:
        match: A dictionary with information of the team.
        bld_ratings: The BLD (Berrar, Lopes, Dubitzki) rating of each team per league.

    Returns:
        The BLD (Berrar, Lopes, Dubitzki) rating features.
    '''
    home_team_id = match['id_home_team']
    away_team_id = match['id_away_team']
    league = match['league']
    hatt = 0.0
    hdef = 0.0
    aatt = 0.0
    adef = 0.0

    if home_team_id in bld_ratings:
        if league in bld_ratings[home_team_id]:
            hatt = bld_ratings[home_team_id][league]['hatt']
            hdef = bld_ratings[home_team_id][league]['hdef']

    if away_team_id in bld_ratings:
        if league in bld_ratings[away_team_id]:
            aatt = bld_ratings[away_team_id][league]['aatt']
            adef = bld_ratings[away_team_id][league]['adef']

    return (hatt, hdef, aatt, adef)


def update_bld_rating(match: Dict, bld_ratings: Dict) -> Dict:
    '''Update rating features

    Source: https://link.springer.com/article/10.1007/s10994-018-5747-8

    Args:
        match: A dictionary with information of the team.
        bld_ratings: The BLD (Berrar, Lopes, Dubitzki) rating of each team per league.

    Returns:
        The BLD (Berrar, Lopes, Dubitzki) rating dictionary updated.
    '''
    home_team_id = match['id_home_team']
    away_team_id = match['id_away_team']
    league = match['league']
    old_hatt = 0.0
    old_hdef = 0.0
    old_aatt = 0.0
    old_adef = 0.0

    if home_team_id in bld_ratings:
        if league in bld_ratings[home_team_id]:
            old_hatt = bld_ratings[home_team_id][league]['hatt']
            old_hdef = bld_ratings[home_team_id][league]['hdef']
        else:
            bld_ratings[home_team_id][league] = {
                'hatt': 0.0,
                'hdef': 0.0,
                'aatt': 0.0,
                'adef': 0.0
            }
    else:
        bld_ratings[home_team_id] = {
            league: {
                'hatt': 0.0,
                'hdef': 0.0,
                'aatt': 0.0,
                'adef': 0.0
            }
        }

    if away_team_id in bld_ratings:
        if league in bld_ratings[away_team_id]:
            old_aatt = bld_ratings[away_team_id][league]['aatt']
            old_adef = bld_ratings[away_team_id][league]['adef']
        else:
            bld_ratings[away_team_id][league] = {
                'hatt': 0.0,
                'hdef': 0.0,
                'aatt': 0.0,
                'adef': 0.0
            }
    else:
        bld_ratings[away_team_id] = {
            league: {
                'hatt': 0.0,
                'hdef': 0.0,
                'aatt': 0.0,
                'adef': 0.0
            }
        }

    w_hatt = (2.16 - 2.107) / (2 - 1.588)
    w_hdef = (-0.6 + 0.706) / (2 - 1.368)
    w_aatt = (1.94 - 1.877) / (2 - 1.368)
    w_adef = (-0.56 + 0.6) / (2 - 1.588)

    home_goals = match['score_home_team']
    away_goals = match['score_away_team']
    (home_goals_pred, away_goals_pred) = get_bld_rating_predictions(match, bld_ratings)

    new_hatt = old_hatt + w_hatt * (home_goals - home_goals_pred)
    new_hdef = old_hdef + w_hdef * (away_goals - away_goals_pred)
    new_aatt = old_aatt + w_aatt * (away_goals - away_goals_pred)
    new_adef = old_adef + w_adef * (home_goals - home_goals_pred)

    bld_ratings[home_team_id][league]['hatt'] = new_hatt
    bld_ratings[home_team_id][league]['hdef'] = new_hdef
    bld_ratings[away_team_id][league]['aatt'] = new_aatt
    bld_ratings[away_team_id][league]['adef'] = new_adef

    return bld_ratings


def update_pi_ratings(match: Dict, pi_ratings: Dict) -> Dict:
    '''Update pi-ratings of both teams given a match.

    Args:
        match: A dictionary with information of the team.
        pi_ratings: The pi-rating of each team per league.

    Returns:
        The pi-rating dictionary updated.
    '''
    home_team_id = match['id_home_team']
    away_team_id = match['id_away_team']
    league = match['league']
    old_br_home_4_home_team = 0
    old_br_away_4_home_team = 0
    old_br_home_4_away_team = 0
    old_br_away_4_away_team = 0

    if home_team_id in pi_ratings:
        if league in pi_ratings[home_team_id]:
            old_br_home_4_home_team = pi_ratings[home_team_id][league]['home']
            old_br_away_4_home_team = pi_ratings[home_team_id][league]['away']

    if away_team_id in pi_ratings:
        if league in pi_ratings[away_team_id]:
            old_br_home_4_away_team = pi_ratings[away_team_id][league]['home']
            old_br_away_4_away_team = pi_ratings[away_team_id][league]['away']

    # Observed goal difference
    go = match['score_home_team'] - match['score_away_team']

    # Expected goal difference
    base = 10
    coeff = 3
    # I removed absolute values in the formula to avoid Math overflow errors
    gpx = pow(base, old_br_home_4_home_team / coeff) - 1
    gpy = pow(base, old_br_away_4_away_team / coeff) - 1
    gp = gpx - gpy

    # Error
    error = fabs(go - gp)

    # Function that diminish the importance of the score difference error
    psi = coeff * log(1 + error, base)
    if gp < go:
        psi_x = psi
    else:
        psi_x = -1.0 * psi

    if gp > go:
        psi_y = psi
    else:
        psi_y = -1.0 * psi

    # Updating the pi-ratings
    nu = 0.054
    gamma = 0.79
    new_br_home_4_home_team = old_br_home_4_home_team + psi_x * nu
    home_delta_home_team = new_br_home_4_home_team - old_br_home_4_home_team
    new_br_away_4_home_team = old_br_away_4_home_team + home_delta_home_team * gamma
    new_br_away_4_away_team = old_br_away_4_away_team + psi_y * nu
    away_delta_away_team = new_br_away_4_away_team - old_br_away_4_away_team
    new_br_home_4_away_team = old_br_home_4_away_team + away_delta_away_team * gamma

    home_ratings = {
        'home': new_br_home_4_home_team,
        'away': new_br_away_4_home_team
    }
    away_ratings = {
        'home': new_br_away_4_away_team,
        'away': new_br_home_4_away_team
    }

    if home_team_id in pi_ratings:
        pi_ratings[home_team_id][league] = home_ratings
    else:
        pi_ratings[home_team_id] = {league: home_ratings}

    if away_team_id in pi_ratings:
        pi_ratings[away_team_id][league] = away_ratings
    else:
        pi_ratings[away_team_id] = {league: away_ratings}

    return pi_ratings


### 计算参数"θ"
def get_sita1(FormPtsStr:str)-> int:
    sita = 0
    cursor1 = FormPtsStr[0]
    if cursor1 == "W":
        sita += 1
    elif cursor1 == "L":  #最后一场是输的，就马上返回
        return sita

    for n in range(1,len(FormPtsStr)):
        cursor2 = FormPtsStr[n]
        if cursor2 == "L":
            break
        elif cursor1=="W" and cursor2=="W":
            sita += 1
            cursor1 = cursor2
        elif cursor1=="W" and cursor2=="D":
            sita += 0.5
            cursor1 = cursor2
        elif cursor1=="D" and cursor2=="W":
            sita += 0.5
            cursor1 = cursor2
        elif cursor1=="D" and cursor2=="D":
            cursor1 = cursor2

    return sita

### 计算参数"θ"
def get_sita2(FormPtsStr:str)-> int:
    sita = 0
    cursor1 = FormPtsStr[0]
    if cursor1 == "L":  #最后一场是输的，就马上返回
        for n in range(1, len(FormPtsStr)):
            cursor2 = FormPtsStr[n]
            if cursor2 == "W" or cursor2 == "D":
                sita -= 1
                break
            elif cursor2 == "L":
                sita -= 1

    else:
        for n in range(1, len(FormPtsStr)):
            cursor2 = FormPtsStr[n]
            if cursor2 == "L":
                sita += 1
                break
            elif cursor1 == "W" and cursor2 == "W":
                sita += 1
                cursor1 = cursor2
            elif cursor1 == "W" and cursor2 == "D":
                sita += 1
                cursor1 = cursor2
            elif cursor1 == "D" and cursor2 == "W":
                break
            elif cursor1 == "D" and cursor2 == "D":
                sita += 1
                cursor1 = cursor2

    return sita


#计算主、客队差距等级
def get_rank(dis_ratings):
    if dis_ratings > 2.1:
        rank = 0
    elif dis_ratings > 2 and dis_ratings <= 2.1:
        rank = 1
    elif dis_ratings > 1.9 and dis_ratings <= 2.0:
        rank = 2
    elif dis_ratings > 1.8 and dis_ratings <= 1.9:
        rank = 3
    elif dis_ratings > 1.7 and dis_ratings <= 1.8:
        rank = 4
    elif dis_ratings > 1.6 and dis_ratings <= 1.7:
        rank = 5
    elif dis_ratings > 1.5 and dis_ratings <= 1.6:
        rank = 6
    elif dis_ratings > 1.4 and dis_ratings <= 1.5:
        rank = 7
    elif dis_ratings > 1.3 and dis_ratings <= 1.4:
        rank = 8
    elif dis_ratings > 1.2 and dis_ratings <= 1.3:
        rank = 9
    elif dis_ratings > 1.1 and dis_ratings <= 1.2:
        rank = 10
    elif dis_ratings > 1.0 and dis_ratings <= 1.1:
        rank = 11
    elif dis_ratings > 0.9 and dis_ratings <= 1.0:
        rank = 12
    elif dis_ratings > 0.8 and dis_ratings <= 0.9:
        rank = 13
    elif dis_ratings > 0.7 and dis_ratings <= 0.8:
        rank = 14
    elif dis_ratings > 0.6 and dis_ratings <= 0.7:
        rank = 15
    elif dis_ratings > 0.5 and dis_ratings <= 0.6:
        rank = 16
    elif dis_ratings > 0.4 and dis_ratings <= 0.5:
        rank = 17
    elif dis_ratings > 0.3 and dis_ratings <= 0.4:
        rank = 18
    elif dis_ratings > 0.2 and dis_ratings <= 0.3:
        rank = 19
    elif dis_ratings > 0.1 and dis_ratings <= 0.2:
        rank = 20
    elif dis_ratings > 0 and dis_ratings <= 0.1:
        rank = 21
    elif dis_ratings > -0.1 and dis_ratings <= 0:
        rank = 22
    elif dis_ratings > -0.2 and dis_ratings <= -0.1:
        rank = 23
    elif dis_ratings > -0.3 and dis_ratings <= -0.2:
        rank = 24
    elif dis_ratings > -0.4 and dis_ratings <= -0.3:
        rank = 25
    elif dis_ratings > -0.5 and dis_ratings <= -0.4:
        rank = 26
    elif dis_ratings > -0.6 and dis_ratings <= -0.5:
        rank = 27
    elif dis_ratings > -0.7 and dis_ratings <= -0.6:
        rank = 28
    elif dis_ratings > -0.8 and dis_ratings <= -0.7:
        rank = 29
    elif dis_ratings > -0.9 and dis_ratings <= -0.8:
        rank = 30
    elif dis_ratings > -1.0 and dis_ratings <= -0.9:
        rank = 31
    elif dis_ratings > -1.1 and dis_ratings <= -1.0:
        rank = 32
    elif dis_ratings > -1.2 and dis_ratings <= -1.1:
        rank = 33
    elif dis_ratings > -1.3 and dis_ratings <= -1.2:
        rank = 34
    elif dis_ratings > -1.4 and dis_ratings <= -1.3:
        rank = 35
    elif dis_ratings > -1.5 and dis_ratings <= -1.4:
        rank = 36
    elif dis_ratings > -1.6 and dis_ratings <= -1.5:
        rank = 37
    elif dis_ratings > -1.7 and dis_ratings <= -1.6:
        rank = 38
    elif dis_ratings > -1.8 and dis_ratings <= -1.7:
        rank = 39
    elif dis_ratings > -1.9 and dis_ratings <= -1.8:
        rank = 40
    elif dis_ratings <= -1.9:
        rank = 41

    return rank


#计算样本的实力差距直方图和主、客队各进球数的场次数量
def get_histogram(dataset:Tuple[list]):
    histogram = []
    home_scores = {}
    away_scores = {}
    for data in dataset:
        dis_ratings = data['dis_ratings']
        score_home_team = data['score_home_team']
        score_away_team = data['score_away_team']
        rank = get_rank(dis_ratings)
        histogram.append(rank)
        #计算主、客队各进球数的场次数量
        if score_home_team >= 7:
            GH = 7
        else:
            GH = score_home_team
        if score_away_team >= 7:
            GA = 7
        else:
            GA = score_away_team

        if rank in home_scores:
            if GH in home_scores[rank]:
                home_scores[rank][GH] += 1
            else:
                home_scores[rank][GH] = 1
        else:
            home_scores[rank] = {GH: 1}

        if rank in away_scores:
            if GA in away_scores[rank]:
                away_scores[rank][GA] += 1
            else:
                away_scores[rank][GA] = 1
        else:
            away_scores[rank] = {GA: 1}

    p, bins, patches = plt.hist(histogram, 42, normed=True, facecolor='g', alpha=0.75)
    plt.show()

    return  histogram, home_scores, away_scores


#计算各个条件概率表
def get_CPT(dataset:Tuple[list]):
    histogram, home_scores, away_scores = get_histogram(dataset)
    ad_cpt = []
    result = Counter(histogram)
    for AD in range(1, 43):
        if result[AD] != 0:
            ad_cpt.append(result[AD])
        else:
            ad_cpt.append(0)
    ad_cpt = np.array(ad_cpt) / len(histogram)
    ad_cpt = ad_cpt.reshape((1, 42))

    gh_cpt = np.ones([8, 42]) * (1/8)
    ga_cpt = np.ones([8, 42]) * (1/8)
    for AD in range(1, 43):
        if AD in home_scores:
            matchs = sum(home_scores[AD].values())
            for GH in range(0, 8):
                if GH in home_scores[AD]:
                    gh_cpt[GH][AD-1] = home_scores[AD][GH] / matchs
                else:
                    gh_cpt[GH][AD-1] = 0

    for AD in range(1, 43):
        if AD in away_scores:
            matchs = sum(away_scores[AD].values())
            for GA in range(0, 8):
                if GA in away_scores[AD]:
                    ga_cpt[GA][AD-1] = away_scores[AD][GA] / matchs
                else:
                    ga_cpt[GA][AD-1] = 0

    prediction_cpt = np.array([[0,   0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	1,	0],
                      [1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1],
                      [0,	1,	1,	1,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0, 0, 0]])

    return (ad_cpt, gh_cpt, ga_cpt, prediction_cpt)
