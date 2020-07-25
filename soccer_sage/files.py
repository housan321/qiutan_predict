#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formatter for the files.

Created on Thu Apr  4 19:26:19 2019
@author: Juan Beleño
"""
from typing import Union
from soccer_sage.config import SoccerSageConfig
from soccer_sage.__version__ import __version__


class SoccerSageFiles():
    _sql_query_filename: str = 'get_info_dolores.sql'
    # _sql_query_filename: str = 'get_info.sql'
    _train_dataset_filename: str = 'train(36).tfrecord'
    _test_dataset_filename: str = 'test(36).tfrecord'
    _train_tg_dataset_filename: str = 'train_goals.tfrecord'
    _test_tg_dataset_filename: str = 'test_goals.tfrecord'
    _model_weights_filename: str = f'soccer-classifier-{__version__}.h5'
    _model_tg_weights_filename: str = f'soccer-tg-classifier-{__version__}.h5'
    _reg_model_weights_filename: str = f'soccer-regressor-{__version__}.h5'

    def __init__(self,
                 config: Union[str, SoccerSageConfig] = SoccerSageConfig()):
        if isinstance(config, str):
            config = SoccerSageConfig.from_yaml(config)
        self._input_dir = config.input_directory_path
        self._processed_dir = config.processed_directory_path
        self._output_dir = config.output_directory_path

        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.classifier_type = config.classifier_type

    @property
    def sql_query(self) -> str:
        return str(self._input_dir/self._sql_query_filename)

    @property
    def train_dataset(self) -> str:
        dataset = str(self._processed_dir/self._train_dataset_filename)
        if self.classifier_type == 'total_goals':
            dataset = str(self._processed_dir/self._train_tg_dataset_filename)
        return dataset

    @property
    def test_dataset(self) -> str:
        dataset = str(self._processed_dir/self._test_dataset_filename)
        if self.classifier_type == 'total_goals':
            dataset = str(self._processed_dir/self._test_tg_dataset_filename)
        return dataset

    @property
    def model_weights(self) -> str:
        model = str(self._output_dir/self._model_weights_filename)
        if self.classifier_type == 'total_goals':
            model = str(self._output_dir/self._model_tg_weights_filename)
        return model

    @property
    def regressor_weights(self) -> str:
        model = str(self._output_dir/self._reg_model_weights_filename)
        return model
