import random, time
import matplotlib.pyplot as plt
import numpy as np
import torch
from avg_tb import SummaryWriter
import os


def action_symbol(action_tuple):
    time, action = action_tuple
    arrow = {0: '↑', 1: '→', 2: '↓', 3: '←'}[action]
    return f"{time}{arrow}"

def visualize_policy(env, policy):
    """Visualize the policy in a grid-based environment."""
    grid_size = env.grid_size

    # Create a blank grid
    grid = np.full((grid_size, grid_size), '', dtype=object)

    # Fill the grid with policy actions
    for state, action in policy.items():
        x, y = state  # Extract the position from the state
        grid[x, y] = action_symbol(action)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.imshow(np.ones((grid_size, grid_size)), cmap='Greys', alpha=0.1)  # Light background

    # Add policy symbols to the grid
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, grid[i, j], ha='center', va='center', fontsize=12)

    plt.title("Policy Visualization")
    plt.show()

def init_Q(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])

def get_qmax(Q,s,actions):
    assert(s in Q)
    return max(Q[s].values())

def get_best_action(Q,s,actions):
    qmax = get_qmax(Q,s,actions)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def init_Q_delay_action(Q, s, delays, env_actions, q_init):
    if s not in Q:
        Q[s] = dict([((d, a), q_init)
                     for d in range(0, delays.n)
                     for a in range(0, env_actions.n)])
        # for a in range(0, env_actions.n):
        #     Q[s][(0, a)] = q_init


def get_qmax_constant_action(Q, s, delays, env_actions, max_constant_array, discretization_param, num_clocks):
    if discretization_param == 1:
        clock_values = s[-num_clocks:]
    else:
        clock_values = np.asarray(s[-num_clocks:], dtype=float) * discretization_param  # Assuming clock values are in the state tuple starting from the last index
    max_possible_delay = max(max_constant_array - clock_values)  # Maximum delay considering the current clock values
    #print('Discretization Param', discretization_param, 'Clock values', clock_values, 'Max_constant', max_constant, 'Max possible delay:', max_possible_delay)
    #print(Q[s])
    return max([Q[s][(d, a)]
                for d in range(0, delays.n)
                for a in range(0, env_actions.n)
                if d <= max_possible_delay*discretization_param])

def get_best_delay_action(Q, s, delays, env_actions, max_constant_array, discretization_param, num_clocks):
    """
    Get the best (delay, successor, action) for a state based on the Q-table.
    """
    qmax = get_qmax_constant_action(Q, s, delays, env_actions, max_constant_array,discretization_param, num_clocks)
    if discretization_param == 1:
        clock_values = s[-num_clocks:]
    else:
        clock_values = np.asarray(s[-num_clocks:], dtype=float) * discretization_param  # Assuming clock values are in the state tuple starting from the last index
    max_possible_delay = max(max_constant_array - clock_values)  # Maximum delay considering the current clock values
    #print('Max possible delay:', max_possible_delay)
    best = [(d, a)
            for d in range(0, delays.n)
            for a in range(0, env_actions.n)
            if Q[s][(d, a)] == qmax and d <= max_possible_delay*(1/discretization_param)]
    return random.choice(best)

def get_policy(Q, actions):
    """Extract the policy from the Q-table."""
    policy = {}
    #print(Q)
    for s in Q:
        policy[s] = get_best_action(Q, s, actions)

    return policy


def learn_delay_action(env,
          lr,
          lr_decay,
          total_timesteps,
          epsilon,          
          epsilon_decay,
          gamma,
          q_init,
          use_crm,
          use_rs,
          print_freq,
          min_epsilon,
          learning_starts,
          seed,
          tensorboard_log=None,  # Add this parameter
          exp_name="q_learning_delay",
          aggregator=None):  # for averaging over seeds
    """Train a tabular Q-learning model with delay actions and log average rewards."""
    
    # Initialize TensorBoard writer
    random.seed(seed)

    writer = None
    # if tensorboard_log is not None:
    #     log_dir = os.path.join(tensorboard_log, exp_name+f"disc_clocks_{int(time.time())}"+f"use_crm_{env.add_crm}_crm_{env.crm_option}")
    #     writer = SummaryWriter(log_dir=log_dir)
    if tensorboard_log is not None:
        log_dir = os.path.join(
            tensorboard_log,
            exp_name + f"disc_clocks_{int(time.time())}" + f"use_crm_{env.add_crm}_crm_{env.crm_option}"
        )
        writer = SummaryWriter(log_dir=log_dir, aggregator=aggregator)
        print(f"TensorBoard logging to: {log_dir}")

    step = 0
    num_episodes = 0

    Q = {}
    
    delays = env.env.delay_space
    env_actions = env.env.env_action_space
    max_constant_dict = env.env.max_constant_dict
    max_constant_array = np.array(list(max_constant_dict.values()))
    num_clocks = env.env.num_clocks
    actions = env.env.action_combinations
    discretization_param = env.env.discretization_param

    
    # Logging variables
    episode_rewards = []
    episode_times = []
    episode_discounted_rewards = []
    
    epsilon_action = epsilon
    epsilon_delay = epsilon

    # Running averages
    running_reward = 0
    running_time = 0
    running_discounted_reward = 0
    current_episode_reward = 0
    current_episode_reward_discounted = 0
    
    # Initialize first episode
    options = {'random': False}  # Start with random actions to explore
    s, _ = env.reset(seed=seed, options=options)
    s = tuple(s.flatten()) if hasattr(s, 'shape') and s.ndim > 1 else tuple(s)
    init_state = s
    init_Q_delay_action(Q, s, delays, env_actions, q_init)
    #print('Init q values', Q)
    episode_reward = 0
    episode_time = 0
    episode_discounted_reward = 0
    current_episode_time = 0
    num_episodes = 0
    print("-----------------Strarting Q-learning with delay actions-----------------")
    for global_step in range(total_timesteps):
        
        explore_delay = random.random() < epsilon_delay
        explore_action = random.random() < epsilon_action
        best_delay, best_action = get_best_delay_action(Q, s, delays, env_actions, max_constant_array, discretization_param, num_clocks)
        
        if explore_delay or global_step < learning_starts:
            # Random action from enumerated action space
            a = random.choice(actions)
        else:
            a = (best_delay, best_action)  # Use the best delay and action

        action_index = env.env.action_to_index[a]
        sn, r, term, trunc, info = env.step(action_index)
        sn = tuple(sn)
        a = tuple(a)
        t = a[0]*discretization_param + 1  # Extracting the delay from the action tuple
        done = term or trunc
        
        #print('Step:', global_step, 'State:', s, 'Action:', a, 'Reward:', r, 'Next State:', sn, 'Done:', done, 'Prop', env.unwrapped.get_events())
        
        #print(s, a, r, env.unwrapped.get_events())

        # Updating the Q-values
        experiences = []
        if use_crm:
            experiences.append((s, sn, a, r, done, info))
            for _s, _sn, _a, _r, _done, _info in info["crm-experience"]:
                _s = tuple(_s)
                _sn = tuple(_sn)
                _a = tuple(env.env.index_to_action[_a])
                experiences.append((_s, _sn, _a, _r, _done, _info))
        elif use_rs:
            experiences = [(s, sn, a, r, done, info)]
        else:
            experiences = [(s, sn, a, r, done, info)]

        for _s, _sn, _a, _r, _done, _info in experiences:
            init_Q_delay_action(Q,_s,delays,env_actions,q_init)
            init_Q_delay_action(Q,_sn,delays,env_actions,q_init)

            if _done:
                _delta = _r - Q[_s][_a]
            else:
                _t = _a[0]*discretization_param + 1  # Extracting the delay from the action tuple
                _delta = _r + gamma**_t * get_qmax(Q, _sn, actions) - Q[_s][_a]
            
            Q[_s][_a] += lr * _delta
        
        # Episode stats
        episode_reward += r
        episode_discounted_reward += (r * (gamma ** episode_time)) 
        episode_time += t
        #print(f"-------------------------------")
        #print(Q)
        if done:
            # Start new episode
            #print('-----------------------------------------')

            s, _ = env.reset(seed=seed, options=options)
            s = tuple(s)
            init_Q_delay_action(Q, s, delays, env_actions, q_init)
            
            # Reset episode stats
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_time = 0.05 * episode_time + (1 - 0.05) * running_time
            running_discounted_reward = 0.05 * episode_discounted_reward + (1 - 0.05) * running_discounted_reward
            current_episode_reward = episode_reward
            current_episode_reward_discounted = episode_discounted_reward
            current_episode_time = episode_time 

            episode_reward = 0
            episode_time = 0
            episode_discounted_reward = 0
            num_episodes += 1
            #if env.unwrapped.get_events() != ():
            #    exit(0)
            if global_step > learning_starts:
                options = {'random': False}
                epsilon_delay = max(min_epsilon, epsilon_delay * epsilon_decay)
                epsilon_action = max(min_epsilon, epsilon_action * epsilon_decay)
                lr *= lr_decay
        else:    
            s = sn
            
        if global_step % print_freq == 0:    
            if writer is not None:
                #print(f"Step {global_step}, Running Reward: {running_reward:.2f}")
                writer.add_scalar("values/running_reward", running_reward, global_step)
                writer.add_scalar("values/running_time", running_time, global_step)
                writer.add_scalar("values/running_discounted_reward", running_discounted_reward, global_step)
                writer.add_scalar("values/episode_reward", current_episode_reward, global_step)
                writer.add_scalar("values/episode_discounted_reward", current_episode_reward_discounted, global_step)
                writer.add_scalar("values/episode_time", current_episode_time, global_step)
                writer.add_scalar("values/init_q", get_qmax(Q, init_state, actions), global_step)

                print('------------------------------------------------')
                print(f"Step {global_step}")
                print(f"Running Reward: {running_reward:.2f}")
                print(f"Running Time: {running_time}")
                print(f"Running Discounted Reward: {running_discounted_reward:.2f}")
                print("-------------------------------------------------")


                # Log exploration statistics

            #print(f"Episode {num_episodes}: Reward={running_reward:.2f}, Time={running_time:.2f}, Discounted={running_discounted_reward:.2f}")

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {log_dir}")
    
    return Q, episode_rewards, episode_times, episode_discounted_rewards