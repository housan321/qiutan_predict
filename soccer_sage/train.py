#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training the model.

Created on Wed Apr 10 20:21:16 2019
@author: Juan Beleño
"""
import sys
import tensorflow as tf

from typing import Union
from soccer_sage.config import SoccerSageConfig
from soccer_sage.dataset import load_dataset
from soccer_sage.files import SoccerSageFiles
from soccer_sage.mlstm_fcn import mlstm_fcn, reg_mlstm_fcn

tf.enable_eager_execution()
def train_model(
    config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> None:
    """Train the model and save classifier and feature weights."""
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)
    files = SoccerSageFiles(config)
    # train_en = r'D:\\DeepFootball\\soccer_sage\\soccer_sage\\assets\\processed\\train(36).tfrecord'
    # test_en = r'D:\\DeepFootball\\soccer_sage\\soccer_sage\\assets\\processed\\test(36).tfrecord'
    # train_it = r'D:\\DeepFootball\\soccer_sage\\soccer_sage\\assets\\processed\\train(34).tfrecord'
    # test_it = r'D:\\DeepFootball\\soccer_sage\\soccer_sage\\assets\\processed\\test(34).tfrecord'
    # x_train_en, y_train_en = load_dataset(train_en, config)
    # x_test_en, y_test_en = load_dataset(test_en, config)
    # x_train_it, y_train_it = load_dataset(train_it, config)
    # x_test_it, y_test_it = load_dataset(test_it, config)
    #
    # x_train = tf.concat([x_train_en, x_train_it], axis=0)
    # y_train = tf.concat([y_train_en, y_train_it], axis=0)
    # x_test = tf.concat([x_test_en, x_test_it], axis=0)
    # y_test = tf.concat([y_test_en, y_test_it], axis=0)
    #
    # print(x_train_en.shape)
    # print(y_train_en.shape)
    # print(x_test_en.shape)
    # print(y_test_en.shape)
    #
    # print(x_train_it.shape)
    # print(y_train_it.shape)
    # print(x_test_it.shape)
    # print(y_test_it.shape)
    #
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    x_train, y_train = load_dataset(files.train_dataset, config)
    x_test, y_test = load_dataset(files.test_dataset, config)
    model = mlstm_fcn(config)
    class_weight = {0: 1.0, 1: 1.6771, 2: 1.653}
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=config.verbose,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_steps=config.validation_steps
    )
    model.save_weights(files.model_weights, overwrite=True)
    return 'OK'


def train_regression_model(
    config: Union[str, SoccerSageConfig] = SoccerSageConfig()
) -> None:
    """Train the model and save classifier and feature weights."""
    if isinstance(config, str):
        config = SoccerSageConfig.from_yaml(config)
    files = SoccerSageFiles(config)
    x_train, y_train = load_dataset(files.train_dataset, config)
    x_test, y_test = load_dataset(files.test_dataset, config)
    y_train = tf.argmax(y_train, axis=1)
    y_test = tf.argmax(y_test, axis=1)
    print('EXAMPLE OUTPUT: {}'.format(y_test[0:10]), file=sys.stderr)
    model = reg_mlstm_fcn(config)
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=config.verbose,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_steps=config.validation_steps
    )
    model.save_weights(files.regressor_weights, overwrite=True)
    return 'OK'
