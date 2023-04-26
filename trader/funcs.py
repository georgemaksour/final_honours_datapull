import numpy as np

from config import *

# Save statistics to a file
def save_finals(model_name: str) -> list:
    global NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, NET_WORTH, PCT_STAKED_MEAN

    a, b, c, d, e, f = NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, NET_WORTH, PCT_STAKED_MEAN

    np_arr = False
    for item in d:
        if type(item) == np.array:
            np_arr = True

    if not np_arr:
        net_worth_list = e
    else:
        net_worth_list = [l.tolist() for l in e]
        net_worth_list = [item for sublist in net_worth_list for item in sublist]

    # avg number of wins/loss

    # Average profit per trade
    ppt = c/np.mean(net_worth_list)

    # Max/Min number of wins and losses

    # Average ROI
    stake_list = [i1 * i2 for i1, i2 in zip(e, c)]
    roi = np.mean(net_worth_list)/len(net_worth_list)

    # Sharpe Ratio

    comb_list = [model_name,
                 np.mean(net_worth_list), np.std(net_worth_list),
                 np.mean(a), np.std(a),
                 np.mean(b), np.std(b),
                 np.mean(ppt), np.std(ppt),
                 max(a),
                 min(a),
                 np.mean(roi), np.std(roi),
                 np.mean(net_worth_list)/np.std(net_worth_list), np.std(f)]

    NUMBER_OF_WINS.clear()
    NUMBER_OF_LOSSES.clear()
    NUMBER_OF_TRADES.clear()
    NUMBER_OF_HOLDS.clear()
    NET_WORTH.clear()
    PCT_STAKED_MEAN.clear()
    return comb_list


def save_entire_document(model_name: str):
    global NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, NET_WORTH, PCT_STAKED_MEAN

    a, b, c, d, e, f = NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, NET_WORTH, PCT_STAKED_MEAN
    np_arr = False
    for item in d:
        if type(item) == np.array:
            np_arr = True

    if not np_arr:
        net_worth_list = e
    else:
        net_worth_list = [l.tolist() for l in e]
        net_worth_list = [item for sublist in net_worth_list for item in sublist]

    data_dict = {
        'number_of_wins': a,
        'number_of_losses': b,
        'number_of_trades': c,
        'number_of_holds': d,
        'net_worth': net_worth_list,
    }

    pd.DataFrame(data_dict).to_csv(f'{STAGING_DIR}/results_drl_{model_name}.csv', index=False)

    NUMBER_OF_WINS.clear()
    NUMBER_OF_LOSSES.clear()
    NUMBER_OF_TRADES.clear()
    NUMBER_OF_HOLDS.clear()
    NET_WORTH.clear()
    PCT_STAKED_MEAN.clear()


# Import all statistics from the output folder
def import_all_stagings():
    list_of_files = glob.glob(f'{STAGING_DIR}/results_drl*.csv')
    list_of_df = []
    for file_name in list_of_files:
        list_of_df.append(pd.read_csv(file_name))
    return pd.concat(list_of_df)

# calculate pessimistic return on margin
def calculate_pessimistic_return_on_margin(df: pd.DataFrame):
    return (df['net_worth'].mean()*(df['number_of_wins'].mean() - np.sqrt(df['number_of_wins'].mean())) - df['net_worth'].mean()*(df['number_of_losses'].mean() - np.sqrt(df['number_of_losses'].mean()))) / df['percent staked'].sum()

# Calculate the average profit per trade
def calculate_profit_per_trade(df: pd.DataFrame):
    return df['net_worth'].mean() / df['no_of_trades'].mean()

# Calculate the average win
def calculate_average_win(df: pd.DataFrame):
    win_filt = df[df['net_worth'] > 0]
    return win_filt['net_worth'].mean() / df['number_of_wins'].mean()

# Calculate the average loss
def calculate_average_loss(df: pd.DataFrame):
    loss_filt = df[df['net_worth'] < 0]
    return loss_filt['net_worth'].mean() / df['number_of_losses'].mean()

# Calculate the overall profit
def calculate_overall_profit(df: pd.DataFrame):
    return df['net_worth'].sum()

# Calculate sharpe ratio
def calculate_sharpe_ratio(df: pd.DataFrame):
    return df['net_worth'].mean() / df['net_worth'].std()


def calculate_average_return_on_investment(df: pd.DataFrame):
    return df['net_worth'].mean() / df['percent staked'].sum()*STAKE_SIZE


def calculate_average_number_of_wins(df: pd.DataFrame):
    return df['number_of_wins'].mean()


def calculate_average_number_of_losses(df: pd.DataFrame):
    return df['number_of_losses'].mean()


def calculate_min_max_wins(df: pd.DataFrame):
    return df['number_of_wins'].min(), df['number_of_wins'].max()


def calculate_min_max_losses(df: pd.DataFrame):
    return df['number_of_losses'].min(), df['number_of_losses'].max()


# Wrapper for performance statistics
def performance_statistics_wrapper(df: pd.DataFrame, burn_in: bool, method: str = 'ML', model_name: str = NEURAL_NETWORK) -> None:
    b = 'with_burn_in' if burn_in else 'without_burn_in'

    with open(f'{OUTPUT_DIR}/{method}/{model_name}_{FINAL_RESULTS_FILE}_{b}.txt', 'w') as sys.stdout:
        for group in df.groupby('model_name'):

            model_name = group[0]
            model_name = MODEL_NAMES[model_name]

            model_df = group[1]
            if burn_in:
                model_df = model_df.tail(int(len(df)*(BURN_IN_PCT/100)))

            print('#################################################################')
            print(f'Performance statistics for {model_name}')
            print('-----------------------------------------------------------------')
            print(f'Average profit per trade: {calculate_profit_per_trade(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Average win: {calculate_average_win(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Average loss: {calculate_average_loss(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Average number of wins: {calculate_average_number_of_wins(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Average number of losses: {calculate_average_number_of_losses(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Minimum number of wins: {calculate_min_max_wins(model_df)[0]}')
            print('-----------------------------------------------------------------')
            print(f'Maximum number of wins: {calculate_min_max_wins(model_df)[1]}')
            print('-----------------------------------------------------------------')
            print(f'Minimum number of losses: {calculate_min_max_losses(model_df)[0]}')
            print('-----------------------------------------------------------------')
            print(f'Maximum number of losses: {calculate_min_max_losses(model_df)[1]}')
            print('-----------------------------------------------------------------')
            print(f'Overall profit: {calculate_overall_profit(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Average return on investment: {calculate_average_return_on_investment(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Pessimistic return on margin: {calculate_pessimistic_return_on_margin(model_df)}')
            print('-----------------------------------------------------------------')
            print(f'Sharpe ratio: {calculate_sharpe_ratio(model_df)}')
            print('-----------------------------------------------------------------')
            print('#################################################################')
            print('\n')

    print(model_df)
    return calculate_sharpe_ratio(model_df)

############################################################################################################
# Neural Network and Random Forest
############################################################################################################
def get_average_price(back_list: pd.Series, lay_list: pd.Series):
    back_avg = back_list[:192].mean()
    lay_avg = lay_list[:192].mean()
    return (back_avg + lay_avg) / 2


def averge_price_change(back_list: pd.Series, lay_list: pd.Series):
    back_avg = back_list[132:192].mean() - back_list[0:60].mean()
    lay_avg = lay_list[132:192].mean() - lay_list[0:60].mean()
    return (back_avg + lay_avg) / 2


def weight_of_money(back_list: pd.Series, lay_list: pd.Series, back_volume: pd.Series, lay_volume: pd.Series):
    wom = (back_list[:192] * back_volume[:192] + lay_list[:192] * lay_volume[:192]) / (
                back_volume[:192] + lay_volume[:192])
    return wom.mean()


def weight_of_money_6m(back_list: pd.Series, lay_list: pd.Series, back_volume: pd.Series, lay_volume: pd.Series):
    wom = (back_list[120:192] * back_volume[120:192] + lay_list[120:192] * lay_volume[120:192]) / (
                back_volume[120:192] + lay_volume[120:192])
    return wom.mean()


def change_in_ema(back_list: pd.Series, lay_list: pd.Series):
    back_delta = (back_list[132:192].ewm(alpha=0.5).mean() + lay_list[132:192].ewm(alpha=0.5).mean()) / 2
    lay_delta = (back_list[0:60].ewm(alpha=0.5).mean() + lay_list[0:60].ewm(alpha=0.5).mean()) / 2
    return (back_delta.mean() + lay_delta.mean()) / 2


def calculate_macd_avg(back_list: pd.Series, lay_list: pd.Series):
    back_macd = back_list[62:192].ewm(alpha=0.5).mean() - back_list[132:192].ewm(alpha=0.5).mean()
    lay_macd = lay_list[62:192].ewm(alpha=0.5).mean() - lay_list[132:192].ewm(alpha=0.5).mean()
    return (back_macd.mean() + lay_macd.mean()) / 2


def create_triple_barrier_targets(df: pd.DataFrame, N: float, volatility: False) -> list:
    targets = []
    for index, row in df.iterrows():
        if volatility:
            stad_dev = row[132:192].std()
            if row[240] > row[192] + N * stad_dev:
                targets.append(2)
            elif row[240] < row[192] - N * stad_dev:
                targets.append(0)
            else:
                targets.append(1)
        else:
            if row[240] < row[192] + row[192] * N:
                targets.append(2)
            elif row[240] > row[192] - row[192] * N:
                targets.append(0)
            else:
                targets.append(1)
    return targets


def get_preprocessed_features():
    final_df = []
    list_of_dfs = []
    for file_name in glob.glob(DATA_PATH):
        data = pd.read_csv(file_name)
        if len(data['back_probability']) == 241:
            avg_price = get_average_price(data['Best Back Price'], data['Best Lay Price'])

            avg_price_change = averge_price_change(data['Best Back Price'], data['Best Lay Price'])

            wom = weight_of_money(data['Best Back Price'], data['Best Lay Price'], data['Best Back Size'],
                                  data['Best Lay Size'])

            wom_6m = weight_of_money_6m(data['Best Back Price'], data['Best Lay Price'], data['Best Back Size'],
                                        data['Best Lay Size'])

            ema_delta = change_in_ema(data['Best Back Price'], data['Best Lay Price'])

            macd_avg = calculate_macd_avg(data['Best Back Price'], data['Best Lay Price'])

            final_df.append([avg_price, avg_price_change, wom, wom_6m, ema_delta, macd_avg])

            list_of_dfs.append((1 / data['back_probability']).values.tolist())

    return pd.DataFrame(final_df, columns=['avg_price', 'averge_price_change', 'wom', 'wom_6m', 'ema_delta',
                                           'macd_avg']), pd.DataFrame(list_of_dfs)


def prepare_final_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test = standardise_features(X_train, X_test)
    return X_train, X_test, y_train, y_test


def standardise_features(X_train, X_test):
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_train.mean()) / X_test.std()
    return X_train, X_test


def get_feature_importance(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)

    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))


def perform_neural_network_gridsearch(X_train, y_train):
    parameters = NEURAL_NETWORK_PARAMS
    clf_fitted = GridSearchCV(MLPClassifier(random_state=RANDOM_SEED), parameters, verbose=VERBOSITY).fit(X_train, y_train)
    return clf_fitted


def perform_random_forest_gridsearch(X_train, y_train):
    parameters = RANDOM_FOREST_PARAMS
    clf_fitted = GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED), parameters, verbose=VERBOSITY).fit(X_train, y_train)
    return clf_fitted


def simulate_trades(diff_df, stake_size):
    number_of_trades = 0
    trades = []
    for index, row in diff_df.iterrows():
        if row['Target'] == 2:
            amount = (row[240] * stake_size) / row[192] - stake_size
            number_of_trades += 1
            trades.append(amount)
        if row['Target'] == 0:
            amount = (row[192] * stake_size) / row[240] - stake_size
            number_of_trades += 1
            trades.append(amount)

    print('Number of trades:', number_of_trades)
    print('Total Profit:', sum(trades) * 0.85)
    print('Average return on trades:', sum(trades) / len(trades))


def simulate_uncertain_trades(predict_probas, stake_size):
    trades = []
    pct_staked = []

    number_of_trades = []
    number_of_wins = []
    number_of_losses = []
    number_of_holds = []

    for index, row in predict_probas.iterrows():
        max_col = row[['L', 'H', 'B']].idxmax()

        if max_col == "H":
            number_of_wins.append(0)
            number_of_losses.append(0)
            number_of_holds.append(1)
            number_of_trades.append(0)
            pct_staked.append(0)
            trades.append(0)

        if max_col == "B":
            number_of_trades.append(1)
            amount = (row[240] * stake_size * row['H']) / row[192] - stake_size * row['H']
            pct_staked.append(row['H'])
            if amount > 0:
                number_of_wins.append(1)
                number_of_losses.append(0)
                number_of_holds.append(0)
            else:
                number_of_wins.append(0)
                number_of_losses.append(1)
                number_of_holds.append(0)
            trades.append(amount)

        if max_col == "L":
            number_of_trades.append(1)
            amount = (row[192] * stake_size * row['L']) / row[240] - stake_size * row['L']
            pct_staked.append(row['L'])
            if amount > 0:
                number_of_wins.append(1)
                number_of_losses.append(0)
                number_of_holds.append(0)
            else:
                number_of_wins.append(0)
                number_of_losses.append(1)
                number_of_holds.append(0)
            trades.append(amount)
    return number_of_wins, number_of_losses, number_of_trades, [i * 0.85 for i in trades], number_of_holds, pct_staked


def create_ml_staging_data(joint_df: pd.DataFrame, model_name: str = MODEL_NAME) -> None:
    a, b, c, d, e, f = simulate_uncertain_trades(joint_df, ML_STAKE_SIZE)

    dict_check = {'number_of_wins': a, 'number_of_losses': b,
                  'no_of_trades': c, 'net_worth': d,
                  'number_of_holds': e, 'percent staked': f}

    dict_df = pd.DataFrame(dict_check)
    dict_df['model_name'] = model_name
    dict_df.to_csv(f'{STAGING_DIR}/ML/results_ml_{model_name}.csv')

    list_of_files = glob.glob(f'{STAGING_DIR}/ML/results_ml*.csv')
    list_of_df = []
    for file_name in list_of_files:
        list_of_df.append(pd.read_csv(file_name))

    df = pd.concat(list_of_df)

    performance_statistics_wrapper(df, False, 'ML', model_name)


def machine_learning_simulation(model_name: str = 'neural_network') -> None:
    pre_df, df = get_preprocessed_features()
    targets_list = create_triple_barrier_targets(df, 2, True)

    X_train, X_test, y_train, y_test = prepare_final_data(pre_df, targets_list)
    get_feature_importance(X_train, y_train)

    # Perform Gridsearch
    if model_name == NEURAL_NETWORK:
        fitted_obj = perform_neural_network_gridsearch(X_train, y_train)
    else:
        fitted_obj = perform_random_forest_gridsearch(X_train, y_train)

    # Take best of gridsearch and fit predicted and predict probas
    best_model = fitted_obj.best_estimator_
    best_model_fitted = best_model.fit(X_train, y_train)

    # Make predictions
    predictions = best_model_fitted.predict(X_test)
    predict_probas = best_model_fitted.predict_proba(X_test)

    # Predict X values
    idx_vals = list(X_test.index.values)
    new_df = df.loc[idx_vals]
    diff_df = new_df[[192, 240]]
    diff_df['Target'] = predictions

    # Convert diff_df to csv
    diff_df.to_csv(f'{STAGING_DIR}/ML/{model_name}_predictions.csv')

    # Predict probabilities
    probas_df = diff_df.copy()
    probas_df.drop('Target', axis=1, inplace=True)
    probas_df.reset_index(inplace=True)
    predict_probas_df = pd.DataFrame(predict_probas, columns=['L', 'H', 'B'])
    joint_df = pd.merge(probas_df, predict_probas_df, left_index=True, right_index=True)
    joint_df.to_csv(f'{STAGING_DIR}/ML/{model_name}_predict_probas.csv')

    # Simulate Trades
    create_ml_staging_data(joint_df, model_name)


# Glob import data from PREPROCESSING FILES
def get_all_files():
    list_of_files = glob.glob(DATA_PATH)
    list_of_df = []
    for file_name in list_of_files:
        df = pd.read_csv(file_name, index_col=False)
        df.fillna(0, inplace=True)
        list_of_df.append(df)

    data = pd.concat(list_of_df)
    data.reset_index(inplace=True)
    print("Data is imported")
    return data


def perform_final_summary_statistics():
    for model_name in LIST_OF_MODELS:
        list_of_files = glob.glob(f'{STAGING_DIR}/DRL/{MULTI_RUNS}/{model_name}/*.csv')
        list_of_df = []
        for file_name in list_of_files:
            list_of_df.append(pd.read_csv(file_name))

        df = pd.concat(list_of_df)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        with open(f'{OUTPUT_DIR}/DRL/{MULTI_RUNS}/multiple_runs_{model_name}.txt', 'w') as sys.stdout:
            official_name = MODEL_NAMES[model_name]

            print('#################################################################')
            print(f'Overall statistics for {official_name} over 30 runs')
            print('#################################################################')
            print('Mean')
            print(df.mean())
            print('#################################################################')
            print('Standard Deviation')
            print(df.std())
            print('#################################################################')
