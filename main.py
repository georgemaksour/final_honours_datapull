from config import *
from comms import *
from market import *
from strategy import *
from account_calls import *
from classes import *
from data_pull import *
from pre_processing import *
from models import *

logging.basicConfig(filename=LOGGING_FILE_PATH, level=logging.INFO)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    event_tracker = TRACKER()
    trading = setup_trading_login()
    host = setup_text_pass()

    if DATA_PULL:
        while True:
            data_pull(trading=trading, event_tracker=event_tracker)
    else:
        pre_process_files()


