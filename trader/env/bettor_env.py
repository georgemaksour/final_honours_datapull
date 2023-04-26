# Imports
import gym
from gym import spaces
import pandas as pd
import numpy as np
from config import *
import logging

LOGGING_PATH = 'logging/logs.txt'

# Scaling Variables
MAX_ACCOUNT_BALANCE = 20000
MAX_NETWORTH = 10
MIN_NETWORTH = -10
COMMISSION_RATE = 0.95
MAX_NO_OF_TRADES = 100

# Filters for the dataframe
MAX_BACK_PRICE = 5001
MAX_LAY_PRICE = 5001
MAX_TIME_DELTA = 1200

# Stake size
STAKE_SIZE = 10

BACK_PARAM = 1
INITIAL_ACCOUNT_BALANCE = 100
LOOK_AHEAD = 30
NUMBER_OF_DISCRETE = 4

NUMBER_OF_WINS = []
NUMBER_OF_LOSSES = []
NUMBER_OF_TRADES = []
NET_WORTH = []
NUMBER_OF_HOLDS = []
PCT_STAKED = []
PCT_STAKED_MEAN = []

DISCRETE_SPACES = ['dqn', 'acer', 'a2c', 'ppo2', 'acer_lstm', 'a2c_lstm']

logging.basicConfig(filename=f'{LOGGING_PATH}', level=logging.INFO)


class BettorTradingEnv(gym.Env):
    """A sports trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, model_name: str):
        super(BettorTradingEnv, self).__init__()
        self.model_name = model_name

        # Define dataframe which observations will be drawn
        self.df = df
        self.max_number_of_steps = 240

        # Action
        if self.model_name in DISCRETE_SPACES:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float16)

        # Observation
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float16)

        # Reward
        self.reward_range = (-10, 10)

        # Tracking bet and lay
        self.back_amount = STAKE_SIZE
        self.back_price = 0
        self.no_backs = 0

        # Tracking steps
        self.current_step = 0

        # Auxiliary
        self.net_worth = 0
        self.max_net_worth = MAX_NETWORTH
        self.no_of_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.held_no = 0
        self.amount = 0

    def _next_observation(self):
        # Get the data at the current step in time scaled between 0 and 1
        frame = np.array([
            self.df.loc[self.current_step, 'back_probability'],
            self.df.loc[self.current_step, 'back_look_back_probs'],
            self.df.loc[self.current_step, 'ema_back_one'],
            self.df.loc[self.current_step, 'back_look_back_probs'],
        ])

        return frame

    def step(self, action):
        """Step is a meta function that gives the course of action in sequences between two observations
        Args:
            action (object): the action of the step.
        Returns:
            np.ndarray: of the next observations to be passed into the model.
            int: the reward amount.
            bool: whether the observation is over.
            dict: empty dict"""

        global NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, PCT_STAKED_MEAN

        # Execute one time step within the environment
        done = False
        amount = self._take_action(action)
        self.current_step += 1

        reward = amount

        if (self.current_step - LOOK_AHEAD) % self.max_number_of_steps == 0 and (self.current_step - LOOK_AHEAD) != 0:
            done = True
            NUMBER_OF_WINS.append(self.winning_trades)
            NUMBER_OF_LOSSES.append(self.losing_trades)
            NUMBER_OF_TRADES.append(self.no_of_trades)
            NUMBER_OF_HOLDS.append(self.held_no)
            NET_WORTH.append(self.net_worth)
            PCT_STAKED_MEAN.append(np.mean(PCT_STAKED))

        if self.current_step == len(self.df) - LOOK_AHEAD:
            done = True
            self.current_step = 0
            NUMBER_OF_WINS.append(self.winning_trades)
            NUMBER_OF_LOSSES.append(self.losing_trades)
            NUMBER_OF_TRADES.append(self.no_of_trades)
            NUMBER_OF_HOLDS.append(self.held_no)
            NET_WORTH.append(self.net_worth)
            PCT_STAKED_MEAN.append(np.mean(PCT_STAKED))

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action: int) -> float:
        """The take action function is the main part of this object that performs a back, lay or hold dependent on
        the models decision given its environment.
        Args:
            action (int): a list of two items specifying the chosen action and the chosen amount.
        Returns:
            None"""
        global PCT_STAKED

        if self.model_name in DISCRETE_SPACES:
            # Get current stats
            future_back_price = round(1 / self.df.loc[self.current_step + LOOK_AHEAD, "back_probability"], 2)
            current_back_price = round(1 / self.df.loc[self.current_step, "back_probability"], 2)
            self.amount = amount = 0

            if action == 1:
                pct_staked = action/NUMBER_OF_DISCRETE

                PCT_STAKED.append(pct_staked)
                self.back_price = current_back_price
                if current_back_price != future_back_price:
                    amount = round(future_back_price/current_back_price, 2) - 1
                else:
                    amount = 0

                self.back_price = current_back_price

                if amount > 0:
                    self.winning_trades += 1
                elif amount < 0:
                    self.losing_trades += 1
                else:
                    self.held_no += 1
                self.no_of_trades += 1

            else:
                self.held_no += 1

            self.amount = amount
            self.net_worth += amount

            if type(amount) == np.ndarray:
                amount = np.asscalar(amount)
            return amount

        else:
            future_back_price = round(1 / self.df.loc[self.current_step + LOOK_AHEAD, "back_probability"], 2)
            current_back_price = round(1 / self.df.loc[self.current_step, "back_probability"], 2)
            self.amount = amount = 0

            if type(action) == np.ndarray:
                action = np.asscalar(action)

            if action <= 0.5:
                pct_staked = action/NUMBER_OF_DISCRETE

                PCT_STAKED.append(pct_staked)
                self.back_price = current_back_price
                if current_back_price != future_back_price:
                    amount = round(future_back_price/current_back_price, 2) - 1
                else:
                    amount = 0

                self.back_price = current_back_price

                if amount > 0:
                    self.winning_trades += 1
                elif amount < 0:
                    self.losing_trades += 1
                else:
                    self.held_no += 1
                self.no_of_trades += 1

            else:
                self.held_no += 1

            self.net_worth += amount
            self.amount = amount

            if type(amount) == np.ndarray:
                amount = np.asscalar(amount)
            return amount

    def reset(self, ):
        """Function that resets all variables in the object after each race
        Returns:
            ndarray of the next observation"""
        global PCT_STAKED

        self.net_worth = 0
        self.back_price = 0
        self.no_of_trades = 0
        self.no_backs = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.held_no = 0
        PCT_STAKED = []
        obs = self._next_observation()
        return obs

    def render(self, mode='human', close=False) -> None:
        """Render is run after each of the steps in an observation, it gives insight into the current model and
        is configurable to understand the models decisions
        Args:
            mode (str): Tells the mode that is being observed
            close (bool): Tells whether to add print when closing"""

        print(f'Step: {self.current_step}')
        print(f'Number of trades: {self.no_of_trades}')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Number of holds {self.held_no}')
        print(f'Winning trades: {self.winning_trades}')
        print(f'Losing trades: {self.losing_trades}')


def get_finals():
    """Function that returns the final"""
    global NUMBER_OF_WINS, NUMBER_OF_LOSSES, NUMBER_OF_TRADES, NUMBER_OF_HOLDS, NET_WORTH, PCT_STAKED_MEAN
    wins = NUMBER_OF_WINS
    losses = NUMBER_OF_LOSSES
    trades = NUMBER_OF_TRADES
    holds = NUMBER_OF_HOLDS
    net_worth = NET_WORTH
    pct_staked_mean = PCT_STAKED_MEAN
    NUMBER_OF_WINS.clear()
    NUMBER_OF_LOSSES.clear()
    NUMBER_OF_TRADES.clear()
    NUMBER_OF_HOLDS.clear()
    NET_WORTH.clear()
    PCT_STAKED_MEAN.clear()
    return wins, losses, trades, holds, net_worth, pct_staked_mean

