from config import *


def pre_process_files():
    for file_name in glob.glob(DATA_PATH):
        data = pd.read_csv(file_name)
        for name, group in data.groupby(SELECTION_ID):
            group_df = time_until_jump(group)
            group_df = generate_probabilities(group_df)
            group_df = n_minute_average_prob(group_df, n=6)
            group_df = average_lay_back_lookback(group_df)
            group_df = moving_average_convergence_divergence(group_df, p1=12, p2=6)
            fn = file_name.replace('.csv', '')
            fn = fn.replace('data_files/', '')
            group_df.to_csv(f'{PROCESSED_PATH}/{fn}_{name}.csv', index=False)

def time_until_jump(data: pd.DataFrame) -> pd.DataFrame:
    """Given datasheet will calculate the time until the start of the race in 5 second intervals"""
    number_of_intervals = len(data)
    time_to = list(reversed(range(0, 5*number_of_intervals, 5)))
    data[TIME_DELTA] = time_to
    return data


def generate_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    """Creates probabilites for the dataframe"""
    data[BACK_PROBABILITY] = 1/data[BETFAIR_BEST_BACK]
    data[LAY_PROBABILITY] = 1/data[BETFAIR_BEST_LAY]
    return data


def n_minute_average_prob(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Calculates the average probability over n periods. Each period is 5 seconds
    :param data: dataframe being manipulated
    :param n: number of look back periods, each period is 5 seconds. eg. n=6 is 30s"""

    if n > len(data) - 1: return data
    data[BACK_LOOK_BACK] = data[BACK_PROBABILITY].rolling(window=n).mean()
    data[LAY_LOOK_BACK] = data[LAY_PROBABILITY].rolling(window=n).mean()
    return data


def average_lay_back_lookback(data: pd.DataFrame) -> pd.DataFrame:
    data[AVG_LAY_BACK] = (data[BACK_LOOK_BACK] + data[LAY_LOOK_BACK])/2
    return data


def moving_average_convergence_divergence(data: pd.DataFrame, p1: int, p2: int) -> pd.DataFrame:
    assert p1 > p2, 'Make sure period one is bigger than period 2'
    data[EMA_ONE_BACK] = data[BACK_PROBABILITY].ewm(span=p1, adjust=False).mean()
    data[EMA_TWO_BACK] = data[BACK_PROBABILITY].ewm(span=p2, adjust=False).mean()
    data[EMA_ONE_LAY] = data[LAY_PROBABILITY].ewm(span=p1, adjust=False).mean()
    data[EMA_TWO_LAY] = data[LAY_PROBABILITY].ewm(span=p2, adjust=False).mean()
    data[MACD_BACK] = data[EMA_ONE_BACK] - data[EMA_TWO_BACK]
    data[MACD_LAY] = data[EMA_ONE_LAY] - data[EMA_TWO_LAY]
    return data

def volatility_demeaned_returns(data: pd.DataFrame) -> pd.DataFrame:
    ann_factor = len(data.index) / WINDOW
    data[VDI_BACK] = np.sqrt(np.log(data[BETFAIR_BEST_BACK]).diff().rolling(window).var() * ann_factor)
    data[VDI_LAY] = np.sqrt(np.log(data[BETFAIR_BEST_LAY]).diff().rolling(window).var() * ann_factor)
    return data


def weight_of_money(data: pd.DataFrame) -> pd.DataFrame:
    data[WOM] = (data[BETFAIR_BEST_BACK_SIZE]*data[BETFAIR_BEST_BACK] + data[BETFAIR_BEST_LAY_SIZE]*data[BETFAIR_BEST_LAY])/(data[BETFAIR_BEST_BACK_SIZE] + data[BETFAIR_BEST_LAY_SIZE])
    return data

def min_max_normalisation(data: pd.DataFrame) -> pd.DataFrame:
    data[MIN_MAX_BACK] = (data[BETFAIR_BEST_BACK] - data[BETFAIR_BEST_BACK].min()) / (data[BETFAIR_BEST_BACK].max() - data[BETFAIR_BEST_BACK].min())
    data[MIN_MAX_LAY] = (data[BETFAIR_BEST_LAY] - data[BETFAIR_BEST_LAY].min()) / (data[BETFAIR_BEST_LAY].max() - data[BETFAIR_BEST_LAY].min())
    return data
