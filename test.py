import matplotlib.pyplot as plt
import gymnasium as gym
import random
import shutil
import os
import tyro
import numpy as np
import imageio.v2 as imageio
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


from xml.parsers.expat import model
from dataclasses import dataclass, field

#from env.grid_env import GridWorldEnv
#from env.grid_env_ct import GridWorldEnvCT
#from env.grid_env_ct import GridWorldEnvCT
#from env.frozen_lake import FrozenLakeEnv
from env.grid_env_ca import ContinuousGridWorldEnv
from env.Taxi.taxi import TaxiEnv
from env.Frozen_Lake.frozen_lake import FrozenLakeEnv

#from reward_machines.reward_machine import RewardMachine
#from reward_machines.rm_environment_gym import RewardMachineEnvGym
from reward_machines.timed_reward_machine import TimedRewardMachine
from reward_machines.trm_environment_gym import TimedRewardMachineEnvGym, TimedRewardMachineWrapperGym
from reward_machines.trm_environment_gym_corner_abstraction import CornerAbstractionTimedRewardMachineEnvGym
from reward_machines.rm_environment_gym import RewardMachineEnvGym
from reward_machines.trm_environment_gym_corner_abstraction import CornerAbstractionTimedRewardMachineWrapperGym

#from rl_algos.q_learn_ct import learn_delay_action, get_policy
from rl_algos.q_learn_ct_abs import learn_delay_successor_action, get_policy, get_best_action, get_best_delay_successor_action
from rl_algos.q_learn_ct import learn_delay_action
from rl_algos.dqn import learn as dqn_learn
from rl_algos.dqn import Args as DQNArgs


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

    # Draw black outline
    outline_range = 2
    for dx in range(-outline_range, outline_range+1):
        for dy in range(-outline_range, outline_range+1):
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    # Draw white text
    draw.text((x, y), text, font=font, fill=(255,255,255))
    return np.array(img)
    # You can specify a TTF font file if you want a different font
    # font = PIL.ImageFont.truetype("arial.ttf", 24)
    # font = PIL.ImageFont.load_default()
    # text = f"elapsed time: {elapsed_time}"
    # # Draw text at top-left corner
    # draw.text((10, 10), text, font=font, fill=(255, 0, 0))
    # return np.array(img)

def action_symbol(action_tuple):
    time, successor, action = action_tuple
    arrow = {0: '↑', 1: '→', 2: '↓', 3: '←'}[action]
    return f"delay{time} succ{successor} move{arrow}"

def print_policy(policy, grid_size, action_map):
    """Print the policy in a grid-like pattern."""
    print("Optimal Policy:")
    grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]  # Initialize empty grid

    # Fill the grid with policy actions
    for state, action in policy.items():
        x, y = state[:2]  # Extract the position from the state
        grid[x][y] = action_symbol(action)

    # Print the grid row by row
    for row in grid:
        print(' '.join(row))

def optimal_run(env, Q, actions, gamma, video_path="trm_taxi_policy.mp4"):
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
        print(f"s:{state} a:{action} r:{reward}")
        frames.append(add_elapsed_time_text(env.env.render(), elapsed_time))

        total_discounted_reward += (reward * (gamma ** elapsed_time))
        step_count += 1

    for _ in range(5):
        frames.append(add_elapsed_time_text(env.env.render(), elapsed_time))
    
    # Ensure the videos directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, frames, fps=4, codec="libx264")
    print(f"Saved video to {video_path}")


    # imageio.mimsave("trm_taxi_policy.mp4", frames, fps=4, codec="libx264")
    # print("Saved video to trm_taxi_policy.mp4")

# def optimal_run(env, Q, actions, gamma):
#     """Run the environment using the optimal policy."""
#     state, _ = env.reset(options={'random': False})
#     done = False
#     total_discounted_reward = 0
#     step_count = 0
#     episode_time = 0
#     run = str(state)

#     #print(list(optimal_policy.keys()))
#     #print(f"s:{state[:2]}, u:{state[2]}, R:{env.env.index_to_region[state[3]]}, c:{state[4:]}")
#     frames = [env.env.render()]
#     while not done:
#         print('--------------------------------------------')
#         tup_state = tuple(state)
#         #print(tup_state)
#         #action = optimal_policy[tuple(state)]
#         action = get_best_action(Q,tup_state,actions)
#         #action = get_best_delay_successor_action(Q,tup_state,delays,successors,env_actions,max_constant_array)
#         #print('This is best action:', action_symbol(action), 'Q-value:', Q[tup_state][action])
#         #sorted_actions = sorted(Q[tuple(state)].items(), key=lambda kv: kv[1], reverse=True)
#         #for action1, q_value in sorted_actions[:5]:
#         #    print(f"Comparing action: {action_symbol(action1)}, Q-value: {q_value:.3f}")
#         #print(f"State: {state}, Action: {action_symbol(action)}, Prop: {env.env.env.get_events()}")

#         action_index = env.env.action_to_index[action]  # Ensure action is valid
#         delay = action[0]
#         state, reward, term, trunc, _ = env.step(action_index)
#         done = term or trunc
#         print(f"s:{state} a:{action} r:{reward}")
#         #print(f"{state[2]},current region: {env.env.index_to_region[state[2]]}")

#         #print Q values for different delays for the chosen action
#         # for d in range(env.env.max_constant):
#         #     try:
#         #         print(f"Delay {d}, action{action}: Q-value {Q[tup_state].get((d, action[1], action[2]), 0):.2f}")             
#         #     except:
#         #         print(f"Delay {d}: Q-value {Q[tup_state].get((d, action[1]), 0):.2f}")

#         for _ in range(delay+1):
#             frames.append(env.env.render())
#         # Print Q-values for all states and actions
    
#         #run += "-- "+ action_symbol(action)+' r='+str(f"{reward:.2f}")+' dis_r='+str(f"{reward * (gamma ** episode_time)}")+" -->"+str(state)
#         total_discounted_reward += (reward * (gamma ** episode_time))
#         episode_time += action[0]  # Assuming action[0] is the time step
#         step_count += 1
    
#     for _ in range(5):
#         frames.append(env.env.render())

#     imageio.mimsave("trm_taxi_policy.mp4", frames, fps=4, codec="libx264")
#     print("Saved video to trm_taxi_policy.mp4")
#     #print(f"State: {state}, Action: {action_symbol(action)}, Prop: {env.env.env.get_events()}")
    
#     #print(f"Optimal run completed in {step_count} steps with total reward: {total_discounted_reward}")
#     #print(f"Run details: {run}")

#     # # Initialize a list to store frames for rendering
#     # frames = [env.render()]

#     # # Run a random sequence of actions in this environment
#     # done = False
#     # while not done:
#     #     action = env.action_space.sample()  # your agent here (this takes random actions)
#     #     obs, reward, terminated, truncated, info = env.step(action)
#     #     done = terminated or truncated
#     #     frames.append(env.render())  # Append the frame to the list
#     #     print(f"Action: {action}, Reward: {reward}, Events: {env.unwrapped.get_events()}")

#     # # Save the frames as a video
#     # imageio.mimsave("random_taxi_policy.mp4", frames, fps=4, codec="libx264")
#     # print("Saved video to random_taxi_policy.mp4")

#     # env.close()


def discrete_q_learning():

    #total_runs = 1 # Number of runs to average over
    total_timesteps = 300000 # Total episodes for each run
    q_init = 10  # Initial Q-value for all state-action pairs
    epsilon = 0.9
    epsilon_decay = 0.999
    lr = 0.8
    lr_decay = 0.999
    add_crm = False
    crm_nums = 15
    crm_option = 2
    add_rs = False
    print_freq = 100
    seed = 42
    gamma = 0.999  # Discount factor for future rewards
    min_epsilon = 0
    learning_starts = 0  # Start decaying epsilon after this many steps
    
    # env1 = GridWorldEnvCT(grid_size=4,
    #     agent_start=(0, 0),
    #     goal_pos=[(1,1),(3,3)],
    #     obstacle_pos=[(2,2)]
    # )
    # env1 = FrozenLakeEnv(
    #     map_name="4x4_2Goals",         # Use the standard 4x4 map
    #     is_slippery=True,       # Enable slippery surface (default)
    #     success_rate=1.0/3.0,   # Probability of moving in intended direction
    #     reward_schedule=(0, 0, 0)  # (Goal, Hole, Frozen)
    # )
    # rm_env1 = TimedRewardMachineEnvGym(env1, ["example_trm_frozen_lake_2Goals.txt"], gamma)
    # rm_env2 = TimedRewardMachineEnvGym(env2, ["example_trm2.txt"], gamma)
    
    # gym.envs.registration.register(
    # id='CustomTaxi-v0',
    # entry_point='env.Taxi.taxi:TaxiEnv',  # assuming your file is taxi.py
    # kwargs={'render_mode': 'rgb_array'},  # Specify render mode here
    # max_episode_steps=100,
    # )
    # trm = "env/Taxi/trm1_running.txt"
    # env_name = 'CustomTaxi-v0'

    # gym.envs.registration.register(
    # id='FrozenLakeEnv',
    # entry_point='env.Frozen_Lake.frozen_lake:FrozenLakeEnv',  # assuming your file is frozen_lake.py
    # kwargs={
    #     "map_name": "4x4_2Goals",
    #     "is_slippery": True,
    #     "success_rate": 0.95,
    #     "reward_schedule": (0, 0, 0),  # (Goal, Hole, Frozen)
    # },
    # max_episode_steps=100
    # )

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
    trm = "env/Frozen_Lake/crm_vs_nocrm.txt"
    env_name = 'FrozenLakeEnv'


    crms = [(True,3)]  # (add_crm, crm_option)
    corner_abstractions = [(True,1), (True,1)] # (corner_abstraction, discretization_param) & (False, 0) means (untimed) RM

    options = [(ac, co, ca, dp) for ac, co in crms for ca, dp in corner_abstractions]

    for add_crm, crm_option, corner_abstraction, discretization_param in options:
        print(f"Running with crm_option: {crm_option}, add_crm: {add_crm}, corner_abstraction: {corner_abstraction}, discretization_param: {discretization_param}")
        env = gym.make(env_name, render_mode='rgb_array')
        if corner_abstraction:
            rm_env1 = CornerAbstractionTimedRewardMachineEnvGym(env, [trm], gamma, seed=seed)
            wrap_env1 = CornerAbstractionTimedRewardMachineWrapperGym(env=rm_env1, add_crm=add_crm, add_rs=add_rs, gamma=gamma, rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option)
            q_learn = learn_delay_successor_action
        else:
            rm_env1 = TimedRewardMachineEnvGym(env, [trm], gamma, discretization_param=discretization_param, seed=seed)
            wrap_env1 = TimedRewardMachineWrapperGym(env=rm_env1, add_crm=add_crm, add_rs=add_rs, gamma=gamma, rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option)    
            q_learn = learn_delay_action
        
        env = wrap_env1
        env_actions = env.env.action_combinations
        
        Q, episode_rewards, episode_times, episode_discounted_rewards = q_learn(
                        env,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        lr=lr,
                        lr_decay=lr_decay,
                        total_timesteps=total_timesteps,
                        q_init=q_init,
                        gamma=gamma,
                        use_crm=add_crm,
                        use_rs=add_rs,
                        print_freq=print_freq,
                        min_epsilon=min_epsilon,
                        seed=seed,
                        learning_starts=learning_starts,
                        tensorboard_log="./q_learning_logs",  # Add this line
                        exp_name=f"q_learning_eps_{epsilon}_crm_{add_crm}_discretization_{discretization_param}_delay"
                    )
        
        run_rewards = [episode_rewards]
        run_times = [episode_times]
        run_q_tables = [Q]
        run_discounted_rewards = [episode_discounted_rewards]
        optimal_policy = get_policy(Q, env_actions)
        #print("Optimal Policy:")
        # Construct a unique video filename for each run
        video_dir = "videos"
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{env_name}_crm{add_crm}_crmopt{crm_option}_corner{corner_abstraction}_disc{discretization_param}_seed{seed}.mp4"
        video_path = os.path.join(video_dir, video_name)
        if env_name == 'FrozenLakeEnv':
            # Create a deterministic (not slippery) env for optimal run
            det_env = FrozenLakeEnv(
                map_name="8x8_3Goals",
                is_slippery=False,
                success_rate=1,
                reward_schedule=(0, 0, 0),
                render_mode='rgb_array'
            )
            det_env.reset(seed=seed)
            if corner_abstraction:
                det_rm_env = CornerAbstractionTimedRewardMachineEnvGym(det_env, [trm], gamma, seed=seed)
                det_wrap_env = CornerAbstractionTimedRewardMachineWrapperGym(
                    env=det_rm_env, add_crm=add_crm, add_rs=add_rs, gamma=gamma,
                    rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option
                )
            else:
                det_rm_env = TimedRewardMachineEnvGym(det_env, [trm], gamma, discretization_param=discretization_param, seed=seed)
                det_wrap_env = TimedRewardMachineWrapperGym(
                    env=det_rm_env, add_crm=add_crm, add_rs=add_rs, gamma=gamma,
                    rs_gamma=gamma, crm_nums=crm_nums, crm_option=crm_option
                )
            # Use the deterministic env for optimal run
            optimal_run(det_wrap_env, Q, env_actions, gamma, video_path=video_path)
        else:
            # For other envs, just use the original
            optimal_run(env, Q, env_actions, gamma, video_path=video_path)
        # optimal_run(env, Q, env_actions, gamma)


# def clean_rl_dqn():
#     """Run DQN using CleanRL with similar structure to DDPG."""
#     delete_logs = False
#     if delete_logs:
#         for folder in ["runs", "videos"]:
#             if os.path.exists(folder):
#                 shutil.rmtree(folder)

#     # Define the list of runs with different configurations
#     run_list = [
#           # (add_crm, crm_num, seed, delay_clipping)
#         (False, 10)
#     ]

#     for l in run_list:
#         ac, s = l
#         rm_files = ["env/Taxi/example_trm1_fixed.txt"]
#         # env_args = {
#         #     "grid_size": 4,
#         #     "agent_start": (0, 0),
#         #     "goal_pos": [(1, 1),(3,3)],
#         #     "obstacle_pos": [(2,2)]
#         # }

#         @dataclass
#         class CustomArgs(DQNArgs):
#             env_id: str = "CustomTaxi-v0"
#             seed: int = s
#             add_crm: bool = ac
#             rm_files: list[str] = field(default_factory=lambda: rm_files)
#             #env_args: dict = field(default_factory=lambda: env_args)
#             total_timesteps: int = 1000000
#             capture_video: bool = False
#             crm_num: int = 2
#             crm_option: int = 2

#         # Parse arguments and run DQN
#         args = tyro.cli(CustomArgs)
#         dqn_learn(args)
    

def main():
    """Main function to run the Q-learning algorithm on the GridWorld environment."""
    # Initialize the GridWorld environment
    # env1 = GridWorldEnvCT(grid_size=4,
    #     agent_start=(0, 0),
    #     goal_pos=[(0, 2),(1,3)],
    #     obstacle_pos=[]
    # )
    # env2 = GridWorldEnvCT(grid_size=4,
    #     agent_start=(0, 0),
    #     goal_pos=[(0, 2),(1,3)],
    #     obstacle_pos=[]

    #################### Discrete Q-learning ########################
    # q-learning parameters
    
    discrete_q_learning()
    

    #################### CLEAN RL DQN ########################
    #clean_rl_dqn()
    ###################################################################

    
    
if __name__ == "__main__":
    main()
