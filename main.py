import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from avg_tb import AvgLogger
from os import path
import argparse

from reward_machines.trm_environment_gym import TimedRewardMachineEnvGym, TimedRewardMachineWrapperGym
from reward_machines.trm_environment_gym_corner_abstraction import CornerAbstractionTimedRewardMachineEnvGym
from reward_machines.trm_environment_gym_corner_abstraction import CornerAbstractionTimedRewardMachineWrapperGym

from rl_algos.q_learn_ct_abs import learn_delay_successor_action, get_policy, get_best_action, get_best_delay_successor_action
from rl_algos.q_learn_ct import learn_delay_action


def add_elapsed_time_text(frame, elapsed_time):
    # Convert numpy array to PIL Image
    img = PIL.Image.fromarray(frame)
    draw = PIL.ImageDraw.Draw(img)
    try:
        font = PIL.ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = PIL.ImageFont.load_default()
    text = f"elapsed time: {elapsed_time}"
    x, y = 10, 10

    outline_range = 2
    for dx in range(-outline_range, outline_range+1):
        for dy in range(-outline_range, outline_range+1):
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=(255,255,255))
    return np.array(img)


def optimal_run(env, Q, actions, gamma):
    state, _ = env.reset(options={'random': False})
    done = False
    total_discounted_reward = 0
    step_count = 0
    elapsed_time = 0

    frames = [add_elapsed_time_text(env.env.render(), elapsed_time)]

    while not done:
        tup_state = tuple(state)
        action = get_best_action(Q, tup_state, actions)
        action_index = env.env.action_to_index[action]
        delay = action[0]

        # Simulate delay: add 'delay' waiting frames (before action is executed)
        for _ in range(delay):
            elapsed_time += 1
            frames.append(add_elapsed_time_text(env.env.render(), elapsed_time))

        # Take the action and render the resulting frame
        state, reward, term, trunc, _ = env.step(action_index)
        done = term or trunc
        elapsed_time += 1
        # print(f"s:{state} a:{action} r:{reward}")
        frames.append(add_elapsed_time_text(env.env.render(), elapsed_time))

        total_discounted_reward += (reward * (gamma ** elapsed_time))
        step_count += 1

    for _ in range(5):
        frames.append(add_elapsed_time_text(env.env.render(), elapsed_time))

    imageio.mimsave("trm_taxi_policy.mp4", frames, fps=4, codec="libx264")
    print("Saved video to trm_taxi_policy.mp4")


def discrete_q_learning(env_type, trm_name, add_crm, corner_abstraction, discretization_param, total_timesteps=300000, total_runs=1):

    total_timesteps = total_timesteps
    q_init = 10  # Initial Q-value for all state-action pairs
    epsilon = 0.9
    epsilon_decay = 0.999
    lr = 0.8
    lr_decay = 0.999
    crm_nums = 15
    print_freq = 100
    init_seed = 42 
    gamma = 0.999  # Discount factor for future rewards
    min_epsilon = 0
    learning_starts = 0  # Start decaying epsilon after this many steps
    crm_option = 3  # CRM option to use when add_crm is True
    
    if env_type == "taxi":
        gym.envs.registration.register(
        id='CustomTaxi-v0',
        entry_point='env.Taxi.taxi:TaxiEnv',  # assuming your file is taxi.py
        kwargs={'render_mode': 'rgb_array'},  # Specify render mode here
        max_episode_steps=100,
        )
        # trm = "env/Taxi/crm_vs_nocrm.txt"
        env_name = 'CustomTaxi-v0'
    
    elif env_type == "frozen_lake":
        gym.envs.registration.register(
        id='FrozenLakeEnv',
        entry_point='env.Frozen_Lake.frozen_lake:FrozenLakeEnv',  # assuming your file is frozen_lake.py
        kwargs={
            "map_name": "8x8_3Goals",
            "is_slippery": True,
            "success_rate": 0.8,
            "reward_schedule": (0, 0, 0),  # (Goal, Hole, Frozen)
        },
        max_episode_steps=100
        )
        # trm = "env/Frozen_Lake/crm_vs_nocrm.txt"
        env_name = 'FrozenLakeEnv'

    trm = trm_name
        
    # crms = [(True,3)]  # (add_crm, crm_option)
    # corner_abstractions = [(False,0)] # (corner_abstraction, discretization_param) & (False, 0) means (untimed) RM

    # options = [(ac, co, ca, dp) for ac, co in crms for ca, dp in corner_abstractions]

    # ---- MULTI-SEED LOOP + AVERAGE (exactly 10 runs + 1 avg) ----
    seeds = [init_seed + i for i in range(total_runs)]
    base_exp = f"q_learning_eps_{epsilon}_crm_{add_crm}_discretization_{discretization_param}_delay/"
    agg = AvgLogger()
    last_Q = None
    for s in seeds:
        print(f"Running with crm_option: {crm_option}, add_crm: {add_crm}, corner_abstraction: {corner_abstraction}, discretization_param: {discretization_param}")
        env = gym.make(env_name, render_mode='rgb_array')
        if corner_abstraction:
            rm_env1 = CornerAbstractionTimedRewardMachineEnvGym(env, [trm], gamma, seed=s)
            wrap_env1 = CornerAbstractionTimedRewardMachineWrapperGym(env=rm_env1, add_crm=add_crm, add_rs=False, gamma=gamma, rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option)
            q_learn = learn_delay_successor_action
        else:
            if discretization_param ==0:   # hack for running (untimed) RM
                add_crm = False
            rm_env1 = TimedRewardMachineEnvGym(env, [trm], gamma, discretization_param=discretization_param, seed=s)
            wrap_env1 = TimedRewardMachineWrapperGym(env=rm_env1, add_crm=add_crm, add_rs=False, gamma=gamma, rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option)    
            q_learn = learn_delay_action
        
        env = wrap_env1
        env_actions = env.env.action_combinations



        last_Q, episode_rewards, episode_times, episode_discounted_rewards = q_learn(
            env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            lr=lr,
            lr_decay=lr_decay,
            total_timesteps=total_timesteps,
            q_init=q_init,
            gamma=gamma,
            use_crm=add_crm,
            use_rs=False,
            print_freq=print_freq,
            min_epsilon=min_epsilon,
            seed=s,
            learning_starts=learning_starts,
            tensorboard_log="./q_learning_logs",
            exp_name=base_exp + f"seed_{s}_",
            aggregator=agg,        # NEW: mirror scalars into aggregator
        )
    # write averaged TB run
    # agg.write_avg(path.join("./q_learning_logs", base_exp, "avg_10seeds"))
    avg_run = (
        f"avg_10seeds_"
        f"{'cont' if corner_abstraction else 'disc'}"
        f"_use_crm_{add_crm}"
        f"_crm_{crm_option}"
        f"_discretization_{discretization_param}"
    )
    agg.write_avg(path.join("./q_learning_log_avg", base_exp, avg_run))

    # Use the last run’s Q for the demo video (unchanged behavior)
    optimal_run(env, last_Q, env_actions, gamma)

def main():
    parser = argparse.ArgumentParser(
        description="Run discrete Q-learning experiments."
    )
    parser.add_argument(
        "-e", "--env-type",
        choices=["taxi", "frozen_lake"],
        default="taxi",
        help="Environment type to run"
    )
    parser.add_argument(
        "-t", "--trm-name",
        default="env/Taxi/disc_vs_cont.txt",
        help="Path to the TRM file to load"
    )
    parser.add_argument(
        "-c", "--add-ci",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to add Counterfactual Imagining (default: 1)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["digital", "real"],
        default="digital",
        help="digital or real-time setting (default: digital) for TRM interpretation"
    )
    parser.add_argument(
        "-d", "--discretization-param",
        type=float,
        default=1.0,
        help="Discretization parameter for TRM (default: 1.0); use 0 for untimed RM"
    )
    parser.add_argument(
        "-tt", "--total-timesteps",
        type=int,
        default=300000,
        help="Total timesteps for each run (default: 300000)"
    )
    parser.add_argument(
        "-n", "--total-runs",
        type=int,
        default=1,
        help="Number of runs (default: 1) with different seeds"
    )

    args = parser.parse_args()

    corner_abstraction = True if args.mode == "real" else False
    discrete_q_learning(args.env_type, args.trm_name, bool(args.add_ci), corner_abstraction, args.discretization_param, args.total_timesteps, args.total_runs)
    
    
if __name__ == "__main__":
    main()
