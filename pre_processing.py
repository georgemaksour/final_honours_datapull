import pandas as pd

from config import *

def pre_process_files():
    for file_name in glob.glob(DATA_PATH):
        data = pd.read_csv(file_name)


def time_until_jump(data: pd.DataFrame) -> pd.DataFrame:
    """Given datasheet will calculate the time until the start of the race in 5 second intervals"""
    number_of_intervals = len(data)
    time_to = list(reversed(range(start=0, stop=5*number_of_intervals, step=5)))
    data['time'] = time_to
    return data


def generate_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    """Creates probabilites for the dataframe"""
    data[BACK_PROBABILITY] = 1/data[BETFAIR_BEST_BACK]
    data[LAY_PROBABILITY] = 1/data[BETFAIR_BEST_LAY]
    return data


def n_minute_average_prob(data: pd.DataFrame, n: float) -> pd.DataFrame:
    """Calculates the average probability over n periods. Each period is 5 seconds
    :param data: dataframe being manipulated
    :param n: number of look back periods, each period is 5 seconds. eg. n=6 is 30s"""

    if n > len(data) - 1: return data
    data[BACK_LOOK_BACK] = data[BETFAIR_BEST_BACK].rolling(window=n).mean()
    data[LAY_LOOK_BACK] = data[BETFAIR_BEST_LAY].rolling(window=n).mean()
    return data


def average_lay_back_lookback(data: pd.DataFrame) -> pd.DataFrame:
    data[AVG_LAY_BACK] = (data[BACK_LOOK_BACK] + data[LAY_LOOK_BACK])/2
    return data


def moving_average_convergence_divergence(data: pd.DataFrame) -> pd.DataFrame:

