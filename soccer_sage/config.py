#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file.

Created on Wed Apr  3 22:00:19 2019
@author: Juan Beleño
"""
import os
import yaml

from pathlib import Path


class SoccerSageConfig:
    # Model
    num_results: int = 3
    # Maximum number of goals in a match given our dataset
    num_goals: int = 16 
    feature_size: int = 2 #38 #54
    time_steps: int = 6
    masking_value: float = -16.0
    global_dropout: int = 0.2
    lstm_hidden_size: int = 28 # 32
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    pretrained_classifier: bool = False

    # Training
    verbose: bool = True
    batch_size: int = 28 #32
    n_epochs: int = 20
    # steps_per_epoch: int = 1000
    learning_rate: float = 0.0002
    clip_value: float = 0.0005

    # Nuisance parameters
    n_dataset_threads: int = 4
    shuffle_buffer_size: int = 1024

    # Fixed
    main_directory:str = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    input_directory: str = os.path.join(main_directory, 'assets', 'inputs')
    processed_directory: str = os.path.join(main_directory, 'assets', 'processed')
    output_directory: str = os.path.join(main_directory, 'assets', 'outputs')
    n_train_samples: int = 2843 #75119
    n_test_samples: int = 315 #8346

    def __init__(self, pretrained_classifier=False, classifier_type='results', n_epochs=12):
        self.pretrained_classifier = pretrained_classifier
        self.classifier_type = classifier_type
        self.n_epochs = n_epochs

    @classmethod
    def from_yaml(cls, path: str):
        """Load overrides from a YAML config file."""
        with open(path) as configfile:
            configdict = yaml.safe_load(configfile)
        return cls(**configdict)

    @property
    def steps_per_epoch(self) -> int:
        return self.n_train_samples // self.batch_size

    @property
    def validation_steps(self) -> int:
        return self.n_test_samples // self.batch_size

    @property
    def input_directory_path(self) -> Path:
        path_string = os.environ.get(
            'INPUT_DIRECTORY', self.input_directory)
        return Path(path_string)

    @property
    def processed_directory_path(self) -> Path:
        path_string = os.environ.get(
            'PROCESSED_DIRECTORY', self.processed_directory)
        return Path(path_string)

    @property
    def output_directory_path(self) -> Path:
        path_string = os.environ.get(
            'OUTPUT_DIRECTORY', self.output_directory)
        return Path(path_string)
