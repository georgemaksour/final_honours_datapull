from datapull import data_pull
from market import *

if __name__ == '__main__':
    trading = setup_trading_login()
    data_pull(trading, {})
