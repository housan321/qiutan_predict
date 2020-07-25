#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using the model to predict soccer matches outcome.

Created on Sun Apr 14 17:04:41 2019
@author: Juan Beleño
"""
import numpy as np
import math

from typing import Union
from soccer_sage.config import SoccerSageConfig
from soccer_sage.databases import get_mysql_engine
from soccer_sage.dataset import load_data_for_predictions
from soccer_sage.files import SoccerSageFiles
from soccer_sage.mlstm_fcn import mlstm_fcn, reg_mlstm_fcn
import sys


def predict_results(
    config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> None:
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)
    config.pretrained_classifier = True
    files = SoccerSageFiles(config)
    time_steps = SoccerSageConfig.time_steps
    features, matches, thresholds = load_data_for_predictions(files.sql_query, time_steps,
                                                              config.classifier_type)
    features = np.array(features)
    model = mlstm_fcn(config)
    predictions = model.predict(features)

    parameters = []
    sql_code = ''

    if config.classifier_type == 'results':
        prediction_indexes = predictions.argmax(axis=1)
        sql_code = '''
        INSERT IGNORE INTO matches
        (id, tf_prediction_test, tf_confidence_test)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
          tf_prediction_test = VALUES(tf_prediction_test),
          tf_confidence_test = VALUES(tf_confidence_test)
        '''

        for index, prediction in enumerate(prediction_indexes):
            match_id = matches[index]
            confidence = predictions[index][prediction]
            params = (match_id, int(prediction), float(confidence))
            print('PARAMS: {}'.format(params), file=sys.stderr)
            parameters.append(params)
    else:
        prediction_indexes = predictions.argmax(axis=1)
        sql_code = '''
        INSERT IGNORE INTO matches
        (id, tf_total_goals, tf_tg_confidence)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
          tf_total_goals = VALUES(tf_total_goals),
          tf_tg_confidence = VALUES(tf_tg_confidence)
        '''
        for index, threshold in enumerate(thresholds):
            if threshold and not math.isnan(threshold):
                match_id = matches[index]
                threshold = int(threshold)
                under = float(np.array(predictions[index])[0:threshold + 1].sum())
                over = float(1.0 - under)
                if over > under:
                    params = (match_id, 1, over)
                else:
                    params = (match_id, 0, under)
                print('PARAMS: {}'.format(params), file=sys.stderr)
                parameters.append(params)
    '''
    if len(parameters) > 0:
        engine = get_mysql_engine()
        with engine.connect() as conn:
            conn.execute(sql_code, parameters)
    '''

def predict_regressions(
    config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> None:
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)
    config.pretrained_classifier = True
    files = SoccerSageFiles(config)
    time_steps = SoccerSageConfig.time_steps
    features, matches, thresholds = load_data_for_predictions(files.sql_query, time_steps,
                                                              config.classifier_type)
    features = np.array(features)
    model = reg_mlstm_fcn(config)
    predictions = model.predict(features)

    parameters = []
    sql_code = '''
    INSERT IGNORE INTO matches
    (id, tf_regression)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE
      tf_regression = VALUES(tf_regression)
    '''

    for index, prediction in enumerate(predictions):
        match_id = matches[index]
        params = (match_id, int(prediction))
        print('PARAMS: {}'.format(params), file=sys.stderr)
        parameters.append(params)

    if len(parameters) > 0:
        engine = get_mysql_engine()
        with engine.connect() as conn:
            conn.execute(sql_code, parameters)
