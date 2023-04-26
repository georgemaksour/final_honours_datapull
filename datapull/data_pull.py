from config import *
from market import *

def data_pull(trading, event_tracker):
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
        return 0

    # Go through the Events and check the market books
    market_catalog_df = retrieve_market_catalogs(trading, event_list)

    # Get the win markets
    market_catalog_df = market_catalog_df[market_catalog_df['Market Name'].str.match(r'(^(R|A)[0-9].*$)')]

    # Get markets within normal time interval
    market_catalog_df['seconds to start'] = (
                market_catalog_df['Start Time'] - datetime.datetime.now() + datetime.timedelta(hours=10)).dt.total_seconds()

    market_catalog_df = market_catalog_df[market_catalog_df['seconds to start'] <= 1205]

    # Get new markets
    new_markets_df = market_catalog_df[
        (market_catalog_df['seconds to start'] <= 1205) & (market_catalog_df['seconds to start'] >= 1200)]
    event_tracker.add_new_market(list(new_markets_df['Market ID']))

    # Remove expired markets
    expired_markets = market_catalog_df[market_catalog_df['seconds to start'] <= 0]
    event_tracker.delete_market(list(expired_markets['Market ID']))

    # Retrieve market books
    for market_id in event_tracker.get_keys():
        market_books = retrieve_market_books(host=trading, market_id=market_id)
        for market in market_books:
            if check_runner_book(market.runners):
                break
            temp_df = process_runner_books(market.runners)
            event_tracker.update_market(market_id=market_id, frame=temp_df)

    event_tracker.debug()
    print("One iteration took", time.time() - start_time, "seconds to run")
    print('RAM memory % used:', psutil.virtual_memory()[2])

    end_time = time.time() - start_time
    if 5 - end_time > 0:
        time.sleep(5 - end_time)
    return 1
