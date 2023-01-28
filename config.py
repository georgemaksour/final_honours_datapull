# Georges Betfair config file 3/3/22
#######################################################################
# Packages
import pandas as pd
import numpy as np
import os
import datetime
import json
import time
import sys
import logging
import math
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import glob
from os.path import exists


#######################################################################
# Client specific packages

# Twilio
from twilio.rest import Client

# Betfair
import betfairlightweight
from betfairlightweight import filters

# Email communication
import smtplib
import socket

#######################################################################
# Betfair Global Variables
CERTS_PATH = '/Users/georgemaksour/Docs/betfair_trader/certs/'
USERNAME = "gemaksour"
PASSWORD = "J#ackson1@"
APP_KEY = "XrF0OJ16HwxpJrIW"
# APP_KEY = "oy4o5ru8lQlmxjoL"
SSOID = '+cLLGOD2r3ShKXf86AUJIzDHfcOM1gG1VXo8Ft1eDmE='

#######################################################################
# Twilio Global Variables
TWILIO_ACC = 'AC20ad354a77fd68eebf188271af77758d'
TWILIO_AUTH_KEY = '2534e368231e0a7e15ff29ba9f87ad32'
TWILIO_PHONE_NUMBER = '+19107275923'
MY_NUMBER = '+61410446969'

#######################################################################
# Email global variables
EMAIL_ADDRESS = 'gmaksour@yahoo.com'
YAHOO_PASSCODE = 'rhvkukyclmbdttym'


#######################################################################
# Market Global Variables
DATA_PULL = True
DATA_PATH = 'data_files/*'
LOGGING_FILE_PATH: str = 'output/logs.txt'
HORSE_RACING_ID = 7
GREYHOUND_RACING_ID = 4339
MINUTES_AHEAD = 25
AVOID_MARKETS = ['To Be Placed', 'Exacta', 'Quinella']
NO_OF_SIMULATIONS = 1000000
STAKE_SIZE = 100
BASE_MARKET_ID = -1
MAX_RESULTS = 1000


#######################################################################
# Pre-processing global variables
PROCESSED_PATH = 'preprocessed_files/'
SELECTION_ID = 'Selection ID'
TIME_DELTA = 'time_delta'
BACK_PROBABILITY = 'back_probability'
LAY_PROBABILITY = 'lay_probability'
BETFAIR_BEST_BACK = 'Best Back Price'
BETFAIR_BEST_LAY = 'Best Lay Price'
BETFAIR_BEST_BACK_SIZE = 'Best Back Size'
BETFAIR_BEST_LAY_SIZE = 'Best Lay Size'
BACK_LOOK_BACK = 'back_look_back_probs'
LAY_LOOK_BACK = 'lay_look_back_probs'
AVG_LAY_BACK = 'lay_back_average'
EMA_ONE_BACK = 'ema_back_one'
EMA_ONE_LAY = 'ema_lay_one'
EMA_TWO_BACK = 'ema_back_two'
EMA_TWO_LAY = 'ema_lay_two'
MACD_BACK = 'macd_back'
MACD_LAY = 'macd_lay'
MIN_MAX_BACK = 'min_max_back'
MIN_MAX_LAY = 'min_max_lay'
VDI_BACK = 'vdi_back'
VDI_LAY = 'vdi_lay'
WINDOW = 15


#######################################################################
# Trading global variables
BET_SIZE = 1

TICK_SIZES = {
    1: 0.01,
    2: 0.02,
    3: 0.05,
    4: 0.1,
    6: 0.2,
    10: 0.5,
    20: 1.0,
    30: 2.0,
    50: 5.0,
    100: 10.0,
    1000: 1000
}

