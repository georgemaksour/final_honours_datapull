from runs import *
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def optimize_agent(trial):
    """ Train the model and optimize Optuna maximises the negative log likelihood, so we need to negate the reward here """

    data = pd.read_csv('data/data.csv')
    data = data.round(3)

    #if OPTIMISE_MODEL == 'a2c_lstm':


    #if OPTIMISE_MODEL == 'ppo2':
    #    model_params = optimize_ppo2(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='a2c')])
    #    model = PPO('MlpPolicy', env, verbose=0, **model_params)

    #elif OPTIMISE_MODEL == 'a2c':
    #    model_params = optimise_a2c(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='a2c')])
    #    model = A2C('MlpPolicy', env, verbose=0, **model_params)

    #elif OPTIMISE_MODEL == 'acer':
    #    model_params = optimise_acer(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='acer')])
    #    model = ACER('MlpPolicy', env, verbose=0, **model_params)

    #elif OPTIMISE_MODEL == 'dqn_lstm':
    #    model_params = optimise_dqn_lstm(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='dqn')])
    #    model = DQN('MlpLstmPolicy', env, verbose=0, **model_params)

    #elif OPTIMISE_MODEL == 'ppo2_lstm':
    #    model_params = optimize_ppo2_lstm(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='ppo2_lstm')])
    #    model = PPO2('MlpLstmPolicy', env, verbose=0, **model_params, nminibatches=1)

    #else:
    #    model_params = optimise_dqn(trial)
    #    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='dqn')])
    #    model = DQN('MlpPolicy', env, verbose=0, **model_params)

    global OPTIMISE_MODEL
    #model_params = optimise_a2c_lstm(trial)
    #env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='a2c_lstm')])
    #model = A2C(MlpLstmPolicy, env, verbose=VERBOSE, **model_params)

    OPTIMISE_MODEL = 'acer_lstm'
    model_params = optimise_acer(trial)
    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name=OPTIMISE_MODEL)])
    model = ACER('MlpLstmPolicy', env, verbose=0, **model_params)

    model.learn(HYPERPARAMETER_TIMESTEPS)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return -1 * mean_reward


def optimise_agent_wrapper():
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=4)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')

    df = study.trials_dataframe()
    df.to_csv(f'{HYPERPARAMETER_DIR}/{OPTIMISE_MODEL}.csv', index=False)
