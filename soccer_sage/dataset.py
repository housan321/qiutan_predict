#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building and loading the dataset for training and usage.

Created on Wed Apr  3 22:00:19 2019
@author: Juan Beleño
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import random

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Union
from soccer_sage.config import SoccerSageConfig
from soccer_sage.databases import get_mysql_engine
from soccer_sage.files import SoccerSageFiles
from soccer_sage.preprocess import (convert_raw_data_to_features, get_feature_names,
                         get_recency_features, get_strength_opposition,
                         preprocess_labels, preprocess_features,
                         train_test_split, get_pi_ratings, update_pi_ratings,
                         get_bld_rating_predictions, update_bld_rating,
                         get_odds_features, get_histogram, get_CPT)
from qiutan.league import leagueId

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

def _float_list_feature(value):
    '''Returns a float_list in TF from a float / double list.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> tf.train.Feature:
    '''int64 feature wrapper'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_sequence_example(
        sequence_example: tf.train.SequenceExample,
        config: SoccerSageConfig = SoccerSageConfig()
) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Decode a TFRecord example'''
    sequence_features = {
        'recency_features': tf.io.FixedLenSequenceFeature(
                                [SoccerSageConfig.feature_size], tf.float32)
    }
    context_features = {
        'result': tf.io.FixedLenFeature([], tf.int64)
    }
    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=sequence_example,
        context_features=context_features,
        sequence_features=sequence_features)
    return (sequence_parsed['recency_features'], context_parsed['result'])


def load_dataset(
        file_path: Path, config: SoccerSageConfig
) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Create a dataset generator from a TFRecord path.'''
    n_classes = config.num_results
    if config.classifier_type == 'total_goals':
        n_classes = config.num_goals

    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(
        _parse_sequence_example,
        num_parallel_calls=config.n_dataset_threads,
    )
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(config.batch_size)
    # iterator = dataset.make_one_shot_iterator()  #版本问题，改用下一行代码
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    (features, results) = iterator.get_next()

    ##测试代码
    # with tf.Session() as sess:
    #     fea = sess.run(features)
    #     print(sess.run(features))


    return (features, preprocess_labels(results, n_classes))


def have_fields_the_right_size(recency_features: dict, time_steps: int):
    '''Have the fields the size of the time steps?

    Args:
        recency_features: The recency features for a match.
        time_steps: The number of events per record.

    Returns:
        The answer to the question: "Have the fields the size of the time
        steps?"
    '''
    flag = True
    for key, value in recency_features.items():
        if key != 'result' and len(value) != time_steps:
            #print('Key: {0} => {1}'.format(key, len(value)))
            flag = False
            break
    return flag


def load_raw_data(sql_file: str, time_steps: int, classifier_type: str) -> Tuple[list, list]:
    '''Load and split the feature into training and test datasets

    Args:
        sql_file: A filepath of a sql command.
        time_steps: The number of events per record.

    Returns:
        Training and test dataset
    '''
    date_lis = ['{}-{}'.format(i, i + 1) for i in range(2011, 2021)]  # 赛季格式 2018-2019

    with open(sql_file) as file:
        sql_command = file.read()
    engine = get_mysql_engine()

    recency_features = []
    pi_ratings = {}

    for league in leagueId:
        for date in date_lis:
            matches = pd.read_sql_query(sql_command, engine, params=[league, date])
                                    #params=[time_steps, 'STATUS_FULL_TIME'])
            matches_json = matches.to_dict(orient='records')

            matches_per_team = {}
            # pi_ratings = {}
            bld_ratings = {}
            index = 0
            num = 0

            for match in matches_json:
                home_team_id = match['id_home_team']
                away_team_id = match['id_away_team']
                league = match['league']
                result = match['result']
                score_home_team = match['score_home_team']
                score_away_team = match['score_away_team']
                if classifier_type == 'total_goals':
                    result = match['num_goals']

                (home_data, away_data) = convert_raw_data_to_features(match)
                (home_goals_pred, away_goals_pred) = get_bld_rating_predictions(match, bld_ratings)
                pi_ratings_home, pi_ratings_away = get_pi_ratings(match, pi_ratings)
                dis_ratings = pi_ratings_home - pi_ratings_away
                #print('PREDICTED GOALS: {0} - {1}'.format(home_goals_pred, away_goals_pred), file=sys.stderr)
                #print('REAL GOALS: {0} - {1}'.format(match['score_home_team'], match['score_away_team']), file=sys.stderr)

                # home_data['strength_opposition'] = get_strength_opposition(home_team_id,
                #                                                            league,
                #                                                            time_steps,
                #                                                            matches_per_team)
                home_data['pi_rating'] = pi_ratings_home
                home_data['goals_pred'] = home_goals_pred
                # away_data['strength_opposition'] = get_strength_opposition(away_team_id,
                #                                                            league,
                #                                                            time_steps,
                #                                                            matches_per_team)
                away_data['pi_rating'] = pi_ratings_away
                away_data['goals_pred'] = away_goals_pred

                recency_features_home = get_recency_features(home_team_id, league, time_steps,
                                                             matches_per_team, 'home')
                recency_features_away = get_recency_features(away_team_id, league, time_steps,
                                                             matches_per_team, 'away')

                if home_team_id in matches_per_team:
                    if league in matches_per_team[home_team_id]:
                        matches_per_team[home_team_id][league].append(home_data)
                    else:
                        matches_per_team[home_team_id][league] = [home_data]
                else:
                    matches_per_team[home_team_id] = {league: [home_data]}

                if away_team_id in matches_per_team:
                    if league in matches_per_team[away_team_id]:
                        matches_per_team[away_team_id][league].append(away_data)
                    else:
                        matches_per_team[away_team_id][league] = [away_data]
                else:
                    matches_per_team[away_team_id] = {league: [away_data]}

                # oz_odds0_feature, oz_odds9_feature = get_odds_features(match)

                # Source: https://stackoverflow.com/a/26853961
                recency_features_per_match = {**recency_features_home,
                                              **recency_features_away}
                # recency_features_per_match['oz_odds0'] = oz_odds0_feature
                # recency_features_per_match['oz_odds9'] = oz_odds9_feature

                recency_features_per_match['result'] = result
                flag_sequence_size = have_fields_the_right_size(recency_features_per_match,
                                                                time_steps)
                recency_features_per_match['dis_ratings'] = dis_ratings
                recency_features_per_match['score_home_team'] = score_home_team
                recency_features_per_match['score_away_team'] = score_away_team
                # if flag_sequence_size: # 赛季开始5场以后的比赛才被选取
                #     recency_features.append(recency_features_per_match)
                #     index = index + 1
                    #print(recency_features_per_match['home_fouls'])
                    #flag_win = bool(random.getrandbits(3) >= 2)
                    #if (result != 0 or (flag_win and result == 0)):
                    #    recency_features.append(recency_features_per_match)
                recency_features.append(recency_features_per_match)
                pi_ratings = update_pi_ratings(match, pi_ratings)
                bld_ratings = update_bld_rating(match, bld_ratings)
                num = num + 1
    # train_samples, test_samples = train_test_split(recency_features,
    #                                                test_size=0.1)
    return recency_features, pi_ratings


def load_data_for_predictions(
        sql_file: str, time_steps: int, classifier_type: str
) -> Tuple[list, list]:
    '''Load and split the features datasets for prediction

    Args:
        sql_file: A filepath of a sql command.
        time_steps: The number of events per record.

    Returns:
        A tensor with the recency features and a list with match identifiers.
    '''
    with open(sql_file) as file:
        sql_command = file.read()
    engine = get_mysql_engine()
    matches = pd.read_sql_query(sql_command, engine)
                                # params=['STATUS_SCHEDULED'])
                                # params=[time_steps, 'STATUS_SCHEDULED'])
    matches_json = matches.to_dict(orient='records')
    recency_features = []
    matches_per_team = {}
    pi_ratings = {}
    bld_ratings = {}
    matchesIds = []
    thresholds = []
    today = datetime.now().date()

    for match in matches_json:
        home_team_id = match['id_home_team']
        away_team_id = match['id_away_team']
        league = match['league']
        match_date = match['match_date']

        (home_data, away_data) = convert_raw_data_to_features(match)
        (home_goals_pred, away_goals_pred) = get_bld_rating_predictions(match, bld_ratings)
        pi_ratings_home, pi_ratings_away = get_pi_ratings(match, pi_ratings)

        home_data['strength_opposition'] = get_strength_opposition(home_team_id,
                                                                   league,
                                                                   time_steps,
                                                                   matches_per_team)
        home_data['pi_rating'] = pi_ratings_home
        home_data['goals_pred'] = home_goals_pred
        away_data['strength_opposition'] = get_strength_opposition(away_team_id,
                                                                   league,
                                                                   time_steps,
                                                                   matches_per_team)
        away_data['pi_rating'] = pi_ratings_away
        away_data['goals_pred'] = away_goals_pred

        if home_team_id in matches_per_team:
            if league in matches_per_team[home_team_id]:
                matches_per_team[home_team_id][league].append(home_data)
            else:
                matches_per_team[home_team_id][league] = [home_data]
        else:
            matches_per_team[home_team_id] = {league: [home_data]}

        if away_team_id in matches_per_team:
            if league in matches_per_team[away_team_id]:
                matches_per_team[away_team_id][league].append(away_data)
            else:
                matches_per_team[away_team_id][league] = [away_data]
        else:
            matches_per_team[away_team_id] = {league: [away_data]}

        recency_features_home = get_recency_features(home_team_id, league, time_steps,
                                                     matches_per_team, 'home')
        recency_features_away = get_recency_features(away_team_id, league, time_steps,
                                                     matches_per_team, 'away')
        # Source: https://stackoverflow.com/a/26853961
        recency_features_per_match = {**recency_features_home,
                                      **recency_features_away}
        flag_sequence_size = have_fields_the_right_size(recency_features_per_match,
                                                        time_steps)
        if (match_date <= (today + timedelta(2)) and flag_sequence_size and match_date > date(2019, 6, 15)):
        # if (match_date >= today and match_date <= (today + timedelta(2)) and flag_sequence_size):
            pred_features = preprocess_features(recency_features_per_match,
                                                time_steps)
            recency_features.append(pred_features)
            matchesIds.append(match['match_id'])
            thresholds.append(0.5)
            # thresholds.append(match['betplay_threshold'])

        pi_ratings = update_pi_ratings(match, pi_ratings)
        bld_ratings = update_bld_rating(match, bld_ratings)

    return (recency_features, matchesIds, thresholds)


def save_dataset(
        config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> None:
    '''Store the train and test dataset.

    Args:
        config: A path of a file with config parameters or the default
                config parameters.
    '''
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)
    files = SoccerSageFiles(config)
    time_steps = SoccerSageConfig.time_steps

    recency_features, pi_ratings = load_raw_data(files.sql_query, time_steps,
                                                config.classifier_type)
    ad_cpt, gh_cpt, ga_cpt, prediction_cpt = get_CPT(recency_features)
    # print('TRAIN_SIZE: {}'.format(len(train_dataset)), file=sys.stderr)
    # print('TEST SIZE: {}'.format(len(test_dataset)), file=sys.stderr)
    # write_holdout(train_dataset[:config.n_train_samples],
    #               files.train_dataset, time_steps)
    # write_holdout(test_dataset[:config.n_test_samples],
    #               files.test_dataset, time_steps)
    # bayesian_network_prediction(recency_features[100:300], ad_cpt, gh_cpt, ga_cpt, prediction_cpt)

    return (ad_cpt, gh_cpt, ga_cpt, prediction_cpt, pi_ratings)

def write_holdout(
        samples: List[Dict], file_path: str, time_steps: int
) -> None:
    '''Write a single TFRecord file for either train or test.

    Args:
        samples: The samples from train or test dataset.
        file_path: Where is going to be stored the dataset.
        time_steps: The size of the time serie window.
    '''
    home_feature_names = get_feature_names('home')
    away_feature_names = get_feature_names('away')
    with tf.io.TFRecordWriter(file_path) as writer:
        for sample in samples:
            feature_sequence = []
            for step in range(0, time_steps):
                feature_item = []
                for name in home_feature_names:
                    feature_item.append(sample[name][step])

                for name in away_feature_names:
                    feature_item.append(sample[name][step])

                feature_sequence.append(_float_list_feature(feature_item))

            sequence_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'result': _int64_feature(int(sample['result']))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'recency_features': tf.train.FeatureList(feature=feature_sequence)
                })
            )
            writer.write(sequence_example.SerializeToString())


def bayesian_network_prediction(dataset, ad_cpt, gh_cpt, ga_cpt, prediction_cpt):
    ###创建模型代码
    # coding: utf-8
    # In[16]:
    # Starting with defining the network structure

    dolores_model = BayesianModel([('ability_difference', 'goals_home'),
                                   ('ability_difference', 'goals_away'),
                                   ('goals_home', 'Prediction'),
                                   ('goals_away', 'Prediction')])
    cpd_AD = TabularCPD(variable='ability_difference', variable_card=42,
                          values=ad_cpt)
    cpd_GH = TabularCPD(variable='goals_home', variable_card=8,
                        values=gh_cpt,
                        evidence=['ability_difference'],
                        evidence_card=[42])
    cpd_GA = TabularCPD(variable='goals_away', variable_card=8,
                        values=ga_cpt,
                        evidence=['ability_difference'],
                        evidence_card=[42])
    cpd_P = TabularCPD(variable='Prediction', variable_card=3,
                            values=prediction_cpt,
                            evidence=['goals_home', 'goals_away'],
                            evidence_card=[8, 8])

    # Associating the parameters with the model structure.
    dolores_model.add_cpds(cpd_AD, cpd_GH, cpd_GA, cpd_P)
    # Checking if the cpds are valid for the model.
    dolores_model.check_model()
    dolores_model.get_independencies()
    from pgmpy.inference import VariableElimination
    inference = VariableElimination(dolores_model)

    histogram, home_scores, away_scores = get_histogram(dataset)
    predictions = []
    results = []
    for n in range(len(histogram)):
        rank = histogram[n]
        result = dataset[n]['result']
        pred = inference.query(variables=['Prediction'], evidence={'ability_difference': rank})
        predictions.append(pred.values)
        results.append(result)
    predictions = np.array(predictions)
    predictions = np.around(predictions, 2)
    results = np.array(results)
    results = results.reshape((results.shape[0], 1))
    results = np.around(results, 0)
    kk = np.concatenate((predictions, results), axis=1)

    pred0 = inference.query(variables=['Prediction'], evidence={'ability_difference': 0})
    pred1 = inference.query(variables=['Prediction'], evidence={'ability_difference': 5})
    pred2 = inference.query(variables=['Prediction'], evidence={'ability_difference': 10})
    pred3 = inference.query(variables=['Prediction'], evidence={'ability_difference': 15})
    pred4 = inference.query(variables=['Prediction'], evidence={'ability_difference': 20})
    pred5 = inference.query(variables=['Prediction'], evidence={'ability_difference': 21})
    pred6 = inference.query(variables=['Prediction'], evidence={'ability_difference': 22})
    pred7 = inference.query(variables=['Prediction'], evidence={'ability_difference': 23})
    pred8 = inference.query(variables=['Prediction'], evidence={'ability_difference': 24})
    pred9 = inference.query(variables=['Prediction'], evidence={'ability_difference': 25})

    return 0


def bayesian_network_prediction2(rank, ad_cpt, gh_cpt, ga_cpt, prediction_cpt):
    ###创建模型代码
    # coding: utf-8
    # In[16]:
    # Starting with defining the network structure

    dolores_model = BayesianModel([('ability_difference', 'goals_home'),
                                   ('ability_difference', 'goals_away'),
                                   ('goals_home', 'Prediction'),
                                   ('goals_away', 'Prediction')])
    cpd_AD = TabularCPD(variable='ability_difference', variable_card=42,
                          values=ad_cpt)
    cpd_GH = TabularCPD(variable='goals_home', variable_card=8,
                        values=gh_cpt,
                        evidence=['ability_difference'],
                        evidence_card=[42])
    cpd_GA = TabularCPD(variable='goals_away', variable_card=8,
                        values=ga_cpt,
                        evidence=['ability_difference'],
                        evidence_card=[42])
    cpd_P = TabularCPD(variable='Prediction', variable_card=3,
                            values=prediction_cpt,
                            evidence=['goals_home', 'goals_away'],
                            evidence_card=[8, 8])

    # Associating the parameters with the model structure.
    dolores_model.add_cpds(cpd_AD, cpd_GH, cpd_GA, cpd_P)
    # Checking if the cpds are valid for the model.
    dolores_model.check_model()
    dolores_model.get_independencies()
    from pgmpy.inference import VariableElimination
    inference = VariableElimination(dolores_model)
    pred = inference.query(variables=['Prediction'], evidence={'ability_difference': rank})

    pred_gh = inference.query(variables=['goals_home'], evidence={'ability_difference': rank})
    pred_ga = inference.query(variables=['goals_away'], evidence={'ability_difference': rank})

    return pred.values, pred_gh.values, pred_ga.values