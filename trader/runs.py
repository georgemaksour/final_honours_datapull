from funcs import *

def multiple_runs_wrapper():
    """Multiple runs to find out profitability"""

    list_of_lists = []
    data = pd.read_csv('data/data.csv')
    data = data.round(3)

    for model_name in TEST_LIST:
        for i in range(0, MAX_REPEATS):
            env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name=model_name)])
            if 'dqn' == model_name:
                print(model_name)
                model = DQN('MlpPolicy', env, verbose=VERBOSE)
            elif 'a2c' == model_name:
                print(model_name)
                model = A2C('MlpPolicy', env, verbose=VERBOSE)
            elif 'acer' == model_name:
                print(model_name)
                model = ACER('MlpPolicy', env, verbose=VERBOSE)
            elif 'a2c_lstm' == model_name:
                print(model_name)
                model = A2C('MlpLstmPolicy', env, verbose=VERBOSE)
            elif 'acer_lstm' == model_name:
                print(model_name)
                model = ACER('MlpLstmPolicy', env, verbose=VERBOSE)
            else:
                print(model_name)
                model = PPO2('MlpPolicy', env, verbose=VERBOSE)

            model.learn(total_timesteps=500000)

            obs = env.reset()
            for j in range(SESSION_LENGTH):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()

            if i == 29:
                list_of_lists.append(save_finals(model_name))
                break

    pd.DataFrame(list_of_lists, columns=FINAL_COLS).to_csv('output/DRL/check.csv')


def single_run_for_diagnostics():
    """Running diagnostics for all models and getting the results"""

    data = get_all_files()

    for model_name in LIST_OF_MODELS_SINGLE:
        env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name=model_name)])
        if 'dqn' == model_name:
            model = DQN('MlpPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/MLP/logging_{model_name}',
                        batch_size=240, learning_rate=0.000005)
        elif 'a2c' == model_name:
            model = A2C('MlpPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/MLP/logging_{model_name}',
                        vf_coef=0.4, ent_coef=0.02, learning_rate=0.000005)
        elif 'acer' == model_name:
            model = ACER('MlpPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/MLP/logging_{model_name}',
                         replay_ratio=0.3, replay_start=240, full_tensorboard_log=True, learning_rate=0.000005)
        else:
            model = PPO('MlpPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/MLP/logging_{model_name}',
                        n_steps=240, learning_rate=0.000005)

        model.learn(total_timesteps=LARGE_TOTAL_TIMESTEPS)

        obs = env.reset()
        for j in range(SESSION_LENGTH):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()

    for model_name in LSTM_MODELS:
        env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name=model_name)])
        if 'a2c' == model_name:
            model = A2C('MlpLstmPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/RNN/logging_{model_name}')
        else:
            model = ACER('MlpLstmPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/deep_run/RNN/logging_{model_name}')

        model.learn(total_timesteps=LARGE_TOTAL_TIMESTEPS)

        obs = env.reset()
        for j in range(SESSION_LENGTH):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()


def single_model():
    """Single run for diagnostics"""

    data = pd.read_csv('data/data.csv')
    data = data.round(3)
    env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name='a2c')])

    model = ACER('MlpLPolicy', env, verbose=1, tensorboard_log=f'{TENSOR_LOGS}/fixing/',
                learning_rate=0.021085628, batch_size=240, gamma=0.989404125,
                train_freq=2)

    model.learn(total_timesteps=5000000)

    obs = env.reset()
    for j in range(SESSION_LENGTH):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


# model = ACER('MlpLstmPolicy', env, verbose=VERBOSE, tensorboard_log=f'{TENSOR_LOGS}/fixing/')


def record_over_time():
    data = pd.read_csv('data/data.csv')
    data = data.round(3)

    for model_name in TEST_LIST:
        for i in range(0, MAX_REPEATS):
            env = DummyVecEnv([lambda: BettorTradingEnv(data, model_name=model_name)])
            if 'dqn' == model_name:
                print(model_name)
                model = DQN('MlpPolicy', env, verbose=VERBOSE)
            elif 'a2c' == model_name:
                print(model_name)
                model = A2C('MlpPolicy', env, verbose=VERBOSE)
            elif 'acer' == model_name:
                print(model_name)
                model = ACER('MlpPolicy', env, verbose=VERBOSE)
            elif 'a2c_lstm' == model_name:
                print(model_name)
                model = A2C('MlpLstmPolicy', env, verbose=VERBOSE)
            elif 'acer_lstm' == model_name:
                print(model_name)
                model = ACER('MlpLstmPolicy', env, verbose=VERBOSE)
            else:
                print(model_name)
                model = PPO2('MlpPolicy', env, verbose=VERBOSE)

            model.learn(total_timesteps=500000)

            obs = env.reset()
            for j in range(SESSION_LENGTH):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()

            if i == 29:
                save_entire_document(model_name)
                break


