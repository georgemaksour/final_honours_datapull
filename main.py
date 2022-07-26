from config import *
from comms import *
from market import *
from strategy import *
from account_calls import *
from classes import *

logging.basicConfig(filename=LOGGING_FILE_PATH, level=logging.INFO)
pd.options.mode.chained_assignment = None

# Hello World

if __name__ == '__main__':
    event_tracker = TRACKER()
    trading = setup_trading_login()
    host = setup_text_pass()

    while True:
        start_time = time.time()
        # Retrieve upcoming events in the next 5 minutes, returns dataframe with events
        number, events_df = retrieve_upcoming_races(host=trading, minutes_ahead=MINUTES_AHEAD)

        # If the number is 0, there are no events then and exit for now
        if number == 0:
            logging.info('no events found')

        # Retrieve events in australia with more than one market in the event
        filtered_ev = events_df[events_df['Market Count'] >= 1]
        event_list = list(filtered_ev['Event ID'])

        # Check there exists events
        if len(event_list) == 0:
            continue

        # Go through the Events and check the market books
        market_catalog_df = retrieve_market_catalogs(trading, event_list)

        # Get the win markets
        market_catalog_df = market_catalog_df[market_catalog_df['Market Name'].str.match(r'(^(R|A)[0-9].*$)')]

        # Get markets within normal time interval
        market_catalog_df['seconds to start'] = (market_catalog_df['Start Time'] - datetime.datetime.now() + datetime.timedelta(hours=10)).dt.total_seconds()
        print(market_catalog_df)
        market_catalog_df = market_catalog_df[market_catalog_df['seconds to start'] <= 1205]

        # Get new markets
        new_markets_df = market_catalog_df[(market_catalog_df['seconds to start'] <= 1205) & (market_catalog_df['seconds to start'] >= 1200)]
        event_tracker.add_new_market(list(new_markets_df['Market ID']))

        # Remove expired markets
        expired_markets = market_catalog_df[market_catalog_df['seconds to start'] <= 0]
        event_tracker.delete_market(list(expired_markets['Market ID']))

        # Remove expired markets from the search space
        market_catalog_df = market_catalog_df[~market_catalog_df['Market ID'].isin(list(expired_markets['Market ID']))]

        # Retrieve market books
        for market_id in event_tracker.get_keys():
            market_books = retrieve_market_books(host=trading, market_id=market_id)
            print(market_books)
            for market in market_books:
                temp_df = process_runner_books(market.runners)
                event_tracker.update_market(market_id=market_id, frame=temp_df)

        print("One iteration took", time.time() - start_time, "seconds to run")
        print('RAM memory % used:', psutil.virtual_memory()[2])

        event_tracker.debug()

        end_time = time.time() - start_time
        time.sleep(5-end_time)


