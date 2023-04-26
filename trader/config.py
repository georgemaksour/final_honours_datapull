##########################################################################################################
# Imports
##########################################################################################################
import gym
import json
import datetime as dt
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import math
import warnings
import sys
import os
import optuna
import shutup

shutup.please()

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines import PPO2, ACER,  DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common import env_checker
from stable_baselines.common.evaluation import evaluate_policy

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


from env.bettor_env import *

##########################################################################################################
# Warnings
##########################################################################################################
warnings.filterwarnings("ignore", category=RuntimeWarning)

##########################################################################################################
# Global Variables
##########################################################################################################
DATA_PATH = '../honours_datapull/preprocessed_files/*.csv'
MODEL_NAME = 'acer'
STAGING_DIR = 'staging'
OUTPUT_DIR = 'output'
TENSOR_LOGS = 'tensor_logs'
HYPERPARAMETER_DIR = 'hyperparameter_tuning'
FINAL_RESULTS_FILE = 'final_results_summarised'
DRL = 'Deep Reinforcement Learning'
ML = 'Machine Learning'
NEURAL_NETWORK = 'neural_network'
RANDOM_FOREST = 'random_forest'
ML_STAKE_SIZE = 10
BURN_IN_PCT = 60
RANDOM_SEED = 42
VERBOSITY = 2
VERBOSE = 0
NUMBER_OF_SIMULATIONS = 2
MULTI_RUNS = 'multiple_runs'
DEEP_REINFORCEMENT_LEARNING = True
MACHINE_LEARNING = False

MAX_REPEATS = 30
SESSION_LENGTH = 241
TOTAL_TIMESTEPS = 500000
LARGE_TOTAL_TIMESTEPS = 1000000

HYPERPARAMETER_TIMESTEPS = 1500

LIST_OF_MODELS_RUNS = ['a2c', 'acer', 'ddpg', 'dqn', 'ppo2',
                       'a2c_lstm', 'acer_lstm']

TEST_LIST = ['acer_lstm', 'acer', 'a2c_lstm', 'a2c', 'ppo2', 'dqn']

LIST_OF_MODELS_SINGLE = ['a2c', 'acer', 'ddpg', 'dqn', 'ppo2']

LSTM_MODELS = ['a2c', 'acer']


FINAL_COLS = ['model_name',
              'average_net_worth_mean', 'average_net_worth_std',
              'average_number_of_wins_mean', 'average_number_of_wins_std',
              'average_number_of_losses_mean', 'average_number_of_losses_std',
              'average_profit_per_trade_mean', 'average_profit_per_trade_std',
              'max_wins_mean',
              'min_wins_mean',
                'average_roi_mean', 'average_roi_std',
              'sharpe_ratio_mean', 'sharpe_ratio_std']

OPTIMISE_MODEL = 'acer'

##########################################################################################################
# Dictionaries
##########################################################################################################
MODEL_NAMES = {
    'ppo2': 'Proximal Policy Optimization',
    'a2c': 'Advantage Actor Critic',
    'acer': 'Actor Critic with Experience Replay',
    'dqn': 'Deep Q Network',
    'ddpg': 'Deep Deterministic Policy Gradient',
    RANDOM_FOREST: "Random Forest Classifier",
    NEURAL_NETWORK: 'Multi-layer perceptron classifier'
}


NEURAL_NETWORK_PARAMS = {
    'solver': ['sgd'],
    'hidden_layer_sizes': [(5, 5, 3)],
    'activation': ['relu', 'tanh'],
    'max_iter': [2000],
    'learning_rate': ['adaptive', 'constant'],
    'verbose': [False],
    'momentum': [0.90]
}


RANDOM_FOREST_PARAMS = {
    'n_estimators': [100],
    'criterion': ['gini', 'entropy'],
}


##########################################################################################################
# Functions
##########################################################################################################

def optimize_ppo2(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'gae_lambda': trial.suggest_loguniform('gae_lambda', 0.8, 0.9999),
        'clip_range': trial.suggest_loguniform('clip_range', 0.1, 0.9),
        'vf_coef': trial.suggest_loguniform('vf_coef', 1e-8, 1e-1),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.9),
    }


def optimize_ppo2_lstm(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'vf_coef': trial.suggest_loguniform('vf_coef', 1e-8, 1e-1),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.9),
    }

def optimise_a2c(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'vf_coef': trial.suggest_loguniform('vf_coef', 1e-8, 1e-1),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.9),
    }


def optimise_acer(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'q_coef': trial.suggest_loguniform('q_coef', 1e-8, 1e-1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.9),
        'replay_ratio': trial.suggest_loguniform('replay_ratio', 0.1, 0.9),
        'rprop_alpha': trial.suggest_loguniform('rprop_alpha', 0.1, 0.9),
        'rprop_epsilon': trial.suggest_loguniform('rprop_epsilon', 0.1, 0.9),
        'correction_term': trial.suggest_loguniform('correction_term', 5, 20),
    }


def optimise_dqn(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'train_freq': trial.suggest_int('train_freq', 1, 5),
        'exploration_fraction': trial.suggest_loguniform('exploration_fraction', 0.1, 0.9),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.3, 0.9),
        'tau': trial.suggest_loguniform('tau', 0.1, 0.9),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
    }


def optimise_dqn_lstm(trial):
    """ Learning hyper-paramters we want to optimise"""

    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'train_freq': trial.suggest_int('train_freq', 1, 5),
        'exploration_fraction': trial.suggest_loguniform('exploration_fraction', 0.1, 0.9),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
    }


def optimise_a2c_lstm(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.01),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', 0.01, 1),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'vf_coef': trial.suggest_loguniform('vf_coef', 0.5, 1),
        'alpha': trial.suggest_loguniform('alpha', 0.9, 0.999),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-5, 1e-1),
    }

