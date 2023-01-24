from config import *


def setup_trading_login():
    """returns betfairlightweight api client object that allows connection to bet"""

    trading = betfairlightweight.APIClient(username=USERNAME, password=PASSWORD, app_key=APP_KEY, certs=f'{CERTS_PATH}')
    # trading.Betfair(username=USERNAME, password=PASSWORD, app_key=APP_KEY, certs=f'{CERTS_PATH}')
    trading.login_interactive()
    return trading


def retrieve_upcoming_races(host, minutes_ahead=MINUTES_AHEAD):
    """Retrieve the upcoming horse and greyhound races from the betfair api

    :param host:
    :param minutes_ahead: how
    :return:
    """

    racing_event_filter = betfairlightweight.filters.market_filter(
        event_type_ids=[GREYHOUND_RACING_ID, HORSE_RACING_ID],
        market_start_time={
            'to': (datetime.datetime.utcnow() + datetime.timedelta(minutes=minutes_ahead)).strftime("%Y-%m-%dT%TZ"),
        }
    )

    upcoming_racing_events = host.betting.list_events(
        filter=racing_event_filter
    )

    upcoming_racing_events_df = pd.DataFrame({
        'Event Name': [event_object.event.name for event_object in upcoming_racing_events],
        'Event ID': [event_object.event.id for event_object in upcoming_racing_events],
        'Event Venue': [event_object.event.venue for event_object in upcoming_racing_events],
        'Country Code': [event_object.event.country_code for event_object in upcoming_racing_events],
        'Time Zone': [event_object.event.time_zone for event_object in upcoming_racing_events],
        'Open Date': [event_object.event.open_date for event_object in upcoming_racing_events],
        'Market Count': [event_object.market_count for event_object in upcoming_racing_events]
    })

    if upcoming_racing_events_df.empty:
        logging.info('no events in the next 5 minutes')
        return 0, upcoming_racing_events_df

    return upcoming_racing_events_df.shape[0], upcoming_racing_events_df


def retrieve_market_catalogs(host, event_id, minutes_ahead=MINUTES_AHEAD):
    market_catalogue_filter = betfairlightweight.filters.market_filter(
        event_ids=event_id,
        market_start_time={
            'to': (datetime.datetime.utcnow() + datetime.timedelta(minutes=minutes_ahead)).strftime("%Y-%m-%dT%TZ")
        },
        in_play_only=False
    )

    market_catalogues = host.betting.list_market_catalogue(
        filter=market_catalogue_filter,
        max_results='1000',
        sort='FIRST_TO_START',
        market_projection=['MARKET_START_TIME', 'MARKET_DESCRIPTION']
    )

    # Create a DataFrame for each market catalogue
    market_catalogues_df = pd.DataFrame({
        'Market Name': [market_cat_object.market_name for market_cat_object in market_catalogues],
        'Market ID': [market_cat_object.market_id for market_cat_object in market_catalogues],
        'Total Matched': [market_cat_object.total_matched for market_cat_object in market_catalogues],
        'Start Time': [market_cat_object.market_start_time for market_cat_object in market_catalogues]
    })

    return market_catalogues_df


def retrieve_market_types(host, event_id):
    """

    :param host:
    :param event_id:
    :return:
    """

    market_types_filter = betfairlightweight.filters.market_filter(event_ids=[event_id])

    # Request market types
    market_types = host.betting.list_market_types(
        filter=market_types_filter
    )

    # Create a DataFrame of market types
    market_types_df = pd.DataFrame({
        'Market Type': [market_type_object.market_type for market_type_object in market_types],
    })

    return market_types_df


def retrieve_market_books(host, market_id):
    """

    :param host:
    :param market_id:
    :return:
    """

    price_filter = betfairlightweight.filters.price_projection(
        price_data=['EX_BEST_OFFERS']
    )

    market_books = host.betting.list_market_book(
        price_projection=price_filter,
        market_ids=[market_id],
    )
    return market_books


def check_runner_book(runner_books):
    return any([len(runner.ex.available_to_back) == 0 for runner in runner_books]) or any([len(runner.ex.available_to_lay) == 0 for runner in runner_books])


def process_runner_books(runner_books):
    """
    This function processes the runner books and returns a DataFrame with the best back/lay prices + vol for each runner
    :param runner_books:
    :return:
    """

    best_back_prices = [runner_book.ex.available_to_back[0].price
                        if runner_book.ex.available_to_back[0].price
                        else 1.01
                        for runner_book
                        in runner_books]
    best_back_sizes = [runner_book.ex.available_to_back[0].size
                       if runner_book.ex.available_to_back[0].size
                       else 1.01
                       for runner_book
                       in runner_books]

    best_lay_prices = [runner_book.ex.available_to_lay[0].price
                       if runner_book.ex.available_to_lay[0].price
                       else 1000.0
                       for runner_book
                       in runner_books]
    best_lay_sizes = [runner_book.ex.available_to_lay[0].size
                      if runner_book.ex.available_to_lay[0].size
                      else 1.01
                      for runner_book
                      in runner_books]

    selection_ids = [runner_book.selection_id for runner_book in runner_books]
    last_prices_traded = [runner_book.last_price_traded for runner_book in runner_books]
    total_matched = [runner_book.total_matched for runner_book in runner_books]
    statuses = [runner_book.status for runner_book in runner_books]
    scratching_datetimes = [runner_book.removal_date for runner_book in runner_books]
    adjustment_factors = [runner_book.adjustment_factor for runner_book in runner_books]

    df = pd.DataFrame({
        'Selection ID': selection_ids,
        'Best Back Price': best_back_prices,
        'Best Back Size': best_back_sizes,
        'Best Lay Price': best_lay_prices,
        'Best Lay Size': best_lay_sizes,
        'Last Price Traded': last_prices_traded,
        'Total Matched': total_matched,
        'Status': statuses,
        'Removal Date': scratching_datetimes,
        'Adjustment Factor': adjustment_factors
    })

    return df

