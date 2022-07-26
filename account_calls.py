from config import *


def retrieve_account_details(trading, get_bal=True):
    acc_funds = trading.account.get_account_funds()
    if get_bal:
        return acc_funds.available_to_bet_balance
    else:
        return {'wallet': acc_funds.wallet,
                'points_balance': acc_funds.points_balance,
                'exposure': acc_funds.exposure,
                'balance': acc_funds.available_to_bet_balance}

def retrieve_current_orders(trading):
    curr_orders = trading.betting.list_current_orders()
    print(curr_orders.orders)

    for orda in curr_orders.orders:
        print(orda)
    print(curr_orders.matches)
    print(curr_orders.more_available)


def retrieve_account_statement(trading):
     acc_stat = trading.account.get_account_statement()
     for statement in acc_stat.account_statement:
         #print(dir(statement))
         print(statement.amount)
         print(statement.balance)
         print(statement.legacy_data.commission_rate)
