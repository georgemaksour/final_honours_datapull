from config import *
from market import *


def inverse_func(dogs_prices):
    return list(map(lambda x: 1/x, dogs_prices))


def round_down(x, a):
    return math.ceil(x / a) * a


def get_increment(price: int):
    if 1 < price <= 2:
        return TICK_SIZES.get(1)
    if 2 < price <= 3:
        return TICK_SIZES.get(2)
    if 3 < price <= 5:
        return TICK_SIZES.get(3)


def round_number(x: float):
    first_num = int(x)
    rounder = TICK_SIZES.get(first_num)
    return round_down(x, rounder)


def all_places(dogs_list):
    """Returns 2d matrix of each dog and the place they are in"""
    probs_list = inverse_func(dogs_list)
    p = np.array(probs_list)
    x = np.power(p, 1/(len(dogs_list) - 1))
    matrix = list()
    matrix.append(probs_list)
    for i in range(2, len(dogs_list) + 1):
        num = (1 - p) * np.power(x, len(dogs_list) - i)
        p += num
        matrix.append(num)
    return np.transpose(np.array(matrix))


def dog_simulator(arr, dog_one: int, dog_two: int):
    probs_one = arr[dog_one]
    probs_two = arr[dog_two]
    num_list = list(range(1, len(arr[dog_one]) + 1))
    race_one = np.random.choice(num_list, NO_OF_SIMULATIONS, p=probs_one)
    race_two = np.random.choice(num_list, NO_OF_SIMULATIONS, p=probs_two)

    one_wins = 0
    two_wins = 0
    total_wins = 0
    uncertainty = 0
    for one, two in zip(race_one, race_two):
        if one < two:
            one_wins += 1
            total_wins += 1
        if two < one:
            two_wins += 1
            total_wins += 1
        if two == one:
            uncertainty += 1

    return inverse_func([one_wins/total_wins, two_wins/total_wins, NO_OF_SIMULATIONS/uncertainty])


def read_file(filename):
    temps = []
    file_exists = exists(f'../events/{filename}.txt')
    if file_exists:
        f = open(f'../events/{filename}.txt')
        for line in f.readlines():
            temps.append(float(line))
        f.close()

    return temps


def write_file(filename, number):
    file = open(f'../events/{filename}.txt', "a")
    file.write(f"{number}\n")
    file.close()


def caluclate_k(b, p):
    return p - ((1 - p) / b)


def dataframe_reshape(df_one, df_two):
    df_one = df_one[WIN_COLS]
    df_two = df_two[H2H_COLS]

    df_one.columns = WIN_RENAME
    df_two.columns = H2H_RENAME

    return df_one, df_two


def plot_function(i):
    global ax, ax1, cpu, ram
    # get data
    cpu.popleft()
    cpu.append(psutil.cpu_percent())
    ram.popleft()
    ram.append(psutil.virtual_memory().percent)

    # clear axis
    ax.cla()
    ax1.cla()

    # plot cpu
    ax.plot(cpu)
    ax.scatter(len(cpu)-1, cpu[-1])
    ax.text(len(cpu)-1, cpu[-1]+2, "{}%".format(cpu[-1]))
    ax.set_ylim(0,100)

    # plot memory
    ax1.plot(ram)
    ax1.scatter(len(ram)-1, ram[-1])
    ax1.text(len(ram)-1, ram[-1]+2, "{}%".format(ram[-1]))
    ax1.set_ylim(0,100)
