#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-FCN model.
Source: https://github.com/titu1994/MLSTM-FCN

Created on Sat Jun 22 17:15:37 2019
@author: Juan Beleño
"""
import tensorflow as tf

from soccer_sage.utils import (
    squeeze_excite_block, AttentionLSTM, ranked_probability_score
)
from tensorflow.python.keras.layers import (
    Input, Dense, Dropout, Masking, Permute, Conv1D, BatchNormalization,
    Activation, GlobalAveragePooling1D, concatenate, LSTM
)
from tensorflow.python.keras.models import Model
from typing import Union
from soccer_sage.config import SoccerSageConfig
from soccer_sage.files import SoccerSageFiles


def mlstm_fcn(
        config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> Model:
    '''Load and train the MLSTM-FCN model.
    Source: https://github.com/titu1994/MLSTM-FCN/blob/master/eeg_model.py

    Args:
        config: The configuration data.

    Returns:
        Model: A model in TensorFlow.
    '''
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)

    n_classes = config.num_results
    if config.classifier_type == 'total_goals':
        n_classes = config.num_goals

    ip = Input(shape=(config.time_steps, config.feature_size))

    x = Permute((2, 1))(ip)
    #x = Masking(mask_value=config.masking_value)(x)
    x = LSTM(64)(ip)
    x = Dropout(0.8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(n_classes, activation='softmax')(x)

    classifier = Model(ip, out)

    classifier.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate),
        metrics=['accuracy', ranked_probability_score]
    )

    if config.pretrained_classifier:
        files = SoccerSageFiles(config)
        classifier.load_weights(files.model_weights)

    return classifier


def reg_mlstm_fcn(
        config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> Model:
    '''Load and train the regression MLSTM-FCN model.
    Source: https://github.com/titu1994/MLSTM-FCN/blob/master/eeg_model.py

    Args:
        config: The configuration data.

    Returns:
        Model: A model in TensorFlow.
    '''
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)

    ip = Input(shape=(config.time_steps, config.feature_size))

    x = Permute((2, 1))(ip)
    # x = Masking(mask_value=config.masking_value)(x)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(1, activation='linear')(x)

    classifier = Model(ip, out)

    classifier.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate),
        metrics=['mse', 'mae']
    )

    if config.pretrained_classifier:
        files = SoccerSageFiles(config)
        classifier.load_weights(files.regressor_weights)

    return classifier


