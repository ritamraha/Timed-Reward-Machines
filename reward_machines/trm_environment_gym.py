"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.

However, each environment must implement the following function:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.

Notes:
    - The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
    - The agent only gets the reward given by the RM.
    - Rewards coming from the environment are ignored.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reward_machines.timed_reward_machine import TimedRewardMachine, create_sub_space, find_corner_spaces, extract_bounds
import itertools
import random
from reward_machines.reward_machine_utils import evaluate_dnf


def clock_shift(clock_values, shift, max_constant_dict):
    """Shift all clock values by a given amount."""
    return {c: min(max(v + shift, 0), max_constant_dict[c]) for c, v in clock_values.items()}

def clock_shift_dict(clock_values, shift_dict, max_constant_dict):
    """Shift all clock values by a given amount."""
    return {c: min(max(v + shift_dict.get(c, 0), 0), max_constant_dict[c]) for c, v in clock_values.items()}

def clock_dict_combinations_same_bounds(clock_names, lower_bound, upper_bound):
    """
    Return list of dicts mapping each clock in clock_names to every combination
    of integer values in range(lower_bound, upper_bound) (or inclusive if set).
    """
    rng = range(lower_bound, upper_bound)
    return [dict(zip(clock_names, vals)) for vals in itertools.product(rng, repeat=len(clock_names))]

class TimedRewardMachineEnvGym(gym.Wrapper):
    def __init__(self, env, rm_files, gamma, global_dtype=np.int32, discretization_param=0.5, seed=None):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env)
        print('Runnin on rm files:', rm_files)
        # Loading the reward machines
        self.global_dtype = global_dtype
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        self.gamma = gamma # discount factor for the RM rewards
        for rm_file in rm_files:
            rm = TimedRewardMachine(rm_file, global_dtype=self.global_dtype)
            self.num_rm_states += len(rm.states)
            self.reward_machines.append(rm)
            self.max_constant = max(rm.max_constant for rm in self.reward_machines)  # Maximum delay 
            self.max_constant_dict = rm.max_constant_dict
            self.max_delay = max(rm.max_delay for rm in self.reward_machines)  # Maximum delay
            self.num_clocks = len(rm.clock_names)
            self.clock_names = rm.clock_names
        self.max_constant = int(self.max_constant) if self.global_dtype == np.int32 else float(self.max_constant)

        self.num_rms = len(self.reward_machines)
        self.add_crm = False  # Counterfactual experience is not used by default
        self.add_rs = False   # Reward shaping is not used by default
        self.rm_one_hot = False  # RM state is represented as a one-hot vector by default
        self.steps = 0
        self.max_steps = 1000
        self.discretization_param = discretization_param # 

        if self.discretization_param ==0:   # hack for running (untimed) RM
            self.max_delay = 0
            self.discretization_param = 1
        
        ###################### Creating observation space ######################
        self.features_space = spaces.Box(
            low=0 if hasattr(env.observation_space, 'n') else env.observation_space.low,
            high=env.observation_space.n-1 if hasattr(env.observation_space, 'n') else env.observation_space.high,
            shape=(1,) if hasattr(env.observation_space, 'n') else env.observation_space.shape,
            dtype=self.global_dtype
        )
        self.rm_state_space = spaces.MultiBinary(self.num_rm_states) if self.rm_one_hot else spaces.Box(
                low=0,
                high=self.num_rm_states - 1,
                shape=(1,),
                dtype=self.global_dtype
            )
        self.clock_space = spaces.Box(
            low=np.array([0 for clock in self.clock_names]),
            high=np.array([int(self.max_constant_dict[clock] * (1/self.discretization_param)) for clock in self.clock_names]),
            shape=(len(self.clock_names),),
            dtype=self.global_dtype
        )
        print('Clock space:', self.clock_space)
        self.observation_dict = spaces.Dict({
            'features': self.features_space,
            'rm-state': self.rm_state_space,
            'clock_values': self.clock_space
        })
        # Flattening the observation space for RL algorithms
        obs_flatdim = gym.spaces.flatdim(self.observation_dict)
        obs_low = np.concatenate([
            np.atleast_1d(self.features_space.low),
            np.full(self.num_rm_states, 0, dtype=self.global_dtype) if self.rm_one_hot else np.full(1, 0, dtype=self.global_dtype),
            self.clock_space.low
        ])
        obs_high = np.concatenate([
            np.atleast_1d(self.features_space.high),
            np.full(self.num_rm_states, 1, dtype=self.global_dtype) if self.rm_one_hot else np.full(1, self.num_rm_states - 1, dtype=self.global_dtype),
            self.clock_space.high
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(obs_flatdim,), dtype=self.global_dtype)
        ###################################################################

        ###################### Creating action space ######################
        self.env_action_space = env.action_space
        #self.successor_space = spaces.Discrete(2*self.num_clocks + 1)  # Including the option of not resetting any clock
        #self.delay_space = spaces.Discrete(int(self.max_constant*(1/self.discretization_param)))  # Including the option of not delaying
        self.delay_space = spaces.Discrete(int((self.max_delay)*(1/self.discretization_param))+1)  # Including the option of not delaying
        
        print('Delay upper bound:', self.delay_space.n)
        #print('Successor upper bound:', self.successor_space.n)
        print('Env action upper bound:', self.env_action_space.n)
       
        #print(f"Delay space: {self.delay_space}")
        # self.action_dict = spaces.Dict({
        #         "delay": self.delay_space,
        #         "successor": self.successor_space,
        #         "env_action": self.env_action_space
        #     })
        
        # self.action_combinations = list(product(
        #     range(self.delay_space.n),
        #     range(self.successor_space.n),
        #     range(self.env_action_space.n)
        # ))
        self.action_combinations = [(d, a)
                                    for a in range(self.env_action_space.n)
                                    for d in range(self.delay_space.n)
                                    ]
        print(self.action_combinations)
        self.action_to_index = {action: idx for idx, action in enumerate(self.action_combinations)}
        self.index_to_action = {idx: action for idx, action in enumerate(self.action_combinations)}
        self.num_actions = len(self.action_combinations)

        # Create a new Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)
        env.action_space.seed(seed)
        #print(f"Action space: {self.action_space}")
        #exit(0)
        # Computing normal encodings for the non-terminal RM states
        if self.rm_one_hot:
            print('Using one-hot encoding for RM states')
            self.rm_state_features = {}
            for rm_id, rm in enumerate(self.reward_machines):
                for u_id in rm.states:
                    one_hot = np.zeros(self.num_rm_states, dtype=np.float32)
                    one_hot[rm_id * len(rm.states) + u_id] = 1.0
                    self.rm_state_features[(rm_id, u_id)] = one_hot
            self.rm_done_feat = np.zeros(self.num_rm_states, dtype=np.float32)
        else:
            print('Using index encoding for RM states')
            self.rm_state_features = {}
            for rm_id, rm in enumerate(self.reward_machines):
                for u_id in rm.states:
                    self.rm_state_features[(rm_id,u_id)] = np.array([u_id])
            self.rm_done_feat = np.array([0]) # for terminal RM states, we give as features an array of zeros

        print('RM state features:', self.rm_state_features)
        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm    = None

    def reset(self, seed=None, options=None):
        # Reseting the environment and selecting the next RM tasks
        self.obs, info = self.env.reset(seed=seed, options=None)
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_config  = self.current_rm.reset(options=options)  # (u0, {clock_name: clock_value})
        self.steps = 0
        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_rm_id, self.current_config, False), info

    def step(self, action):
        
        # Extracting actions from tuple
        action = self.index_to_action[action]
        delay = action[0]*self.discretization_param
        env_action = int(action[1]) if self.global_dtype == np.int32 else action[1:]
        
        next_obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.steps += 1

        # print('----------------------------------')
        true_props = self.env.unwrapped.get_events()
        current_config = (self.current_config[0], self.current_config[1].copy())
        # print('Current config:', current_config)
        # print('Delay:', delay)
        self.crm_params = self.obs, current_config, action, next_obs, truncated, true_props, info
        

        # print(f"Current Config: {self.current_config}, True Props: {true_props}, Delay: {delay}")
        # update the RM state
        next_config, rm_rew_pair, rm_done = self.current_rm.step(current_config, true_props, delay+1, info)
        # bound the config of clocks to the maximum delay
        for clock in self.clock_names:
            if next_config[1][clock] > self.max_constant_dict[clock]:
                next_config[1][clock] = self.max_constant_dict[clock]

        # Calculating the reward
        state_rm_rew, transition_rm_rew = rm_rew_pair
        # if self.global_dtype == np.int32:
        #     rm_rew = sum([(self.gamma**t)*state_rm_rew for t in range(delay)]) + transition_rm_rew
        # else:
        #     rm_rew = ((self.gamma**delay - 1)*(1/np.log(self.gamma))*state_rm_rew) + transition_rm_rew
        
        rm_rew = (((self.gamma**delay - 1)/np.log(self.gamma))*state_rm_rew) + transition_rm_rew

        #rm_rew = (((self.gamma**delay - 1)/(self.gamma-1))*state_rm_rew) + transition_rm_rew
        #print('Reward inside TRM env:', rm_rew)
        # Setting the next observations
        self.obs = next_obs
        self.current_config = (next_config[0], next_config[1].copy())
        #print('Current config:', self.current_config, terminated)
        
        #print('Next config:', self.current_config)
        # Calculating if the episode is over
        terminated = rm_done or terminated
        truncated = truncated or self.steps >= self.max_steps
        
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_config, terminated)
        # if truncated:
        #     #print('Number of steps:', self.steps)
        #     info["final_observation"] = rm_obs  # Store the terminal observation in info



        return rm_obs, rm_rew, terminated, truncated, info

    def get_observation(self, next_obs, rm_id, config, done):

        u_id = config[0]
        clock_value_dict = config[1]

        next_obs = np.asarray(next_obs).flatten()  # Flatten the next observation
        #print('State features', self.rm_state_features)
        rm_feat = self.rm_done_feat.flatten() if done else self.rm_state_features[(rm_id, u_id)].flatten()
        clock_values = np.asarray([int((clock_value_dict[clock])*(1/self.discretization_param)) for clock in self.clock_names], dtype=self.global_dtype).flatten()

        
        # Concatenate all parts of the observation
        rm_obs = np.concatenate([next_obs, rm_feat, clock_values])

        return rm_obs 


class TimedRewardMachineWrapperGym(gym.Wrapper):
    def __init__(self, env, add_crm, gamma, crm_nums, crm_option=2, rs_gamma=0.99, add_rs=False):
        """
        RM wrapper
        --------------------
        It adds crm (counterfactual experience) and/or reward shaping to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_crm(bool):   if True, it will add a set of counterfactual experiences to info
            - add_rs(bool):    if True, it will add reward shaping to info
            - gamma(float):    Discount factor for the environment
            - rs_gamma(float): Discount factor for shaping the rewards in the RM
        """
        super().__init__(env)
        self.add_crm = add_crm
        self.add_rs  = add_rs
        self.env.add_crm = add_crm
        self.env.add_rs  = add_rs
        self.crm_option = crm_option
        self.crm_nums = crm_nums

        self.gamma = gamma  # Discount factor for the RM rewards
        #self.clock_spaces = self.env.clock_space
        self.clock_names = self.env.reward_machines[0].clock_names
        
        if add_rs:
            for rm in env.reward_machines:
                rm.add_reward_shaping(gamma, rs_gamma)

    def get_num_rm_states(self):
        return self.env.num_rm_states

    def reset(self, seed=None, options=None):
        self.valid_states = None # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!) 
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # RM and RM state before executing the action
        rm_id = self.env.current_rm_id
        rm    = self.env.current_rm
        config  = self.env.current_config

        # executing the action in the environment
        rm_obs, rm_rew, terminated, truncated, info = self.env.step(action)
        # adding crm if needed
        if self.add_crm:
            
            crm_experience = self._get_crm_experience(*self.env.crm_params)
            info["crm-experience"] = crm_experience
        elif self.add_rs:
            # Computing reward using reward shaping
            _, _, _, rs_env_done, rs_true_props, rs_info = self.crm_params
            _, rs_rm_rew, _ = rm.step(u_id, rs_true_props, rs_info, self.add_rs, rs_env_done)
            info["rs-reward"] = rs_rm_rew

        return rm_obs, rm_rew, terminated, truncated, info

    def _get_rm_experience(self, rm_id, rm, config, obs, action, next_obs, env_done, true_props, info):
        
        u_id, clock_values = config[0], config[1].copy()
        #print(f"State {u_id} - Clock values: {clock_values} - Action: {action}")
        delay = action[0]
        #print(f"Obs {obs}, Next Obs {next_obs}")
        #print(f"Current Config: {config}")
        rm_obs = self.env.get_observation(obs, rm_id, config, False)
        next_config, rm_rew_pair, rm_done = rm.step(config, true_props, delay+1, info, self.add_rs, env_done)
        #print(f"Next Config: {next_config}")
        state_rm_rew, transition_rm_rew = rm_rew_pair
        #rm_rew = sum([(self.gamma**t)*state_rm_rew for t in range(delay)]) + (self.gamma**(delay))*transition_rm_rew
        
        # if self.env.global_dtype == np.int32:
        #     rm_rew = sum([(self.gamma**t)*state_rm_rew for t in range(delay)]) + transition_rm_rew
        # else:
        # rm_rew = ((self.gamma**delay - 1)*(1/np.log(self.gamma))*state_rm_rew) + transition_rm_rew
        rm_rew = (((self.gamma**delay - 1)/(np.log(self.gamma)))*state_rm_rew) + transition_rm_rew
        
        #rm_rew = (((self.gamma**delay - 1)/(self.gamma-1))*state_rm_rew) + transition_rm_rew

        # bound the config of clocks to the maximum delay
        for clock in self.clock_names:
            if next_config[1][clock] > self.env.max_constant_dict[clock]:
                next_config[1][clock] = self.env.max_constant_dict[clock]

        #print(f"u:{config[0]} u':{next_config[0]} v:{config[1]} v':{next_config[1]} d:{action[0]} a:{action[1]} r:{rm_rew}")
        terminated = rm_done
        rm_next_obs = self.env.get_observation(next_obs, rm_id, next_config, terminated)
        info = {}
        if terminated:
            info["terminal_observation"] = rm_next_obs  # Store the terminal observation in info
        delay_index = int(action[0]*(1/self.env.discretization_param))
        env_action = int(action[1]) if self.env.global_dtype == np.int32 else action[1:]
        action = np.array([delay_index, env_action], dtype=self.env.global_dtype)
        action_index = self.env.action_to_index[tuple(action)]

        return (rm_obs, rm_next_obs, action_index, rm_rew, terminated, info), next_config[0]  # Return the experience and the next RM state id

    def _get_crm_experience(self, obs, config, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        reachable_states = set()

        current_state, clock_values = config[0], config[1].copy()
        delay, env_action = action[0], int(action[1])

        #print('Clock values in CRM:', clock_values)
        #print('Delay values in CRM:', delay)

        experiences = []
        #print('Inside function', action)
        # 0 for complete random clock values
        # 1 for random clock values that statisfy guard
        # 2 for random clock values that satisfy guard and also corner points
        #print('Current state:', current_state)
        #print('Clock values:', clock_values)
        if self.crm_option == 0:
            # Random clock values
            #print('Max delay:', self.env.max_delay)
            #for rm_state in [current_state]:
            #print('Current state:', current_state)
            #print('Clock values:', clock_values)
            all_possible_crms = dict()
            current_state, current_clock = config[0], config[1].copy()
            rm_states = [current_state]
            for rm_state in rm_states:
            #for rm_state in range(self.env.num_rm_states):
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                all_possible_crms[rm_state] = []
                
                max_upper_bound = 0 # 0 for no crm on v
                max_lower_bound = 4 # 0 for no crm on v
                                
                clock_upper_bounds = min(max_upper_bound,int(max([self.env.max_constant_dict[clock] - current_clock[clock] for clock in self.clock_names])))+1
                clock_lower_bounds = -min(max_lower_bound,int(min(current_clock.values())))

                shift_dict_list = clock_dict_combinations_same_bounds(self.clock_names, clock_lower_bounds, clock_upper_bounds)
                #print('Shift dict list:', shift_dict_list)


                crm_set = [clock_shift_dict(current_clock, shift, self.env.max_constant_dict) for shift in shift_dict_list]
                
                for crm_clock_values in crm_set:
                    assert min(crm_clock_values.values()) >= 0
                    min_current_clock_value = min(crm_clock_values.values())
                    all_possible_delays = np.arange(0, min(self.env.max_constant - min_current_clock_value, self.env.max_delay), self.env.discretization_param)
                    #print('All possible delays:', list(all_possible_delays))
                    all_possible_new_actions = [np.array([d,env_action]) for d in all_possible_delays]
                    #print(config)
                    all_possible_crms[rm_state].append((crm_clock_values, all_possible_new_actions))

        elif self.crm_option == 1:
            # Random clock values that satisfy the guard
            min_current_clock_value = min(clock_values.values())
            if min_current_clock_value >= self.env.max_constant:
                all_possible_delays = []   
            else:
                dnf_formulas = self.env.current_rm.known_transitions_dnf[(current_state, true_props)]
                state_guards = [theta[2] for dnf_formula in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[current_state][dnf_formula]]

                curr_state_guard = random.choice(state_guards)

                curr_integral = clock_values
                guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)
                #print(f"guard_bounds: {guard_bounds}, curr_integral: {curr_integral}")
                
                #print('Clock names', self.clock_names)
                delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0,self.env.max_delay) for clock in self.clock_names}
                # random sample one space from the list of possible clock spaces
                lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                
                if self.env.discretization_param == 1:
                    all_possible_delays = range(int(lower_delay), int(upper_delay)+1)
                else:
                    all_possible_delays = [d for d in np.arange(lower_delay, upper_delay+1, self.env.discretization_param)]
            
            #print('All possible delays:', list(all_possible_delays))
            all_possible_new_actions = [np.array([d,env_action]) for d in all_possible_delays]
            all_possible_crms = {config[0]: [(config[1], all_possible_new_actions)]}

        elif self.crm_option == 2:

            all_possible_crms = dict()
            for rm_state in range(self.env.num_rm_states):
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                min_current_clock_value = min(clock_values.values())
                if min_current_clock_value >= self.env.max_constant:
                    all_possible_delays = []   
                else:
                    
                    if (rm_state, true_props) in self.env.current_rm.known_transitions_dnf:
                        dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]
                    else:
                        for dnf_formula in self.env.current_rm.outgoing_transitions[rm_state]:
                            if not evaluate_dnf(dnf_formula, true_props):
                                continue
                            self.env.current_rm.known_transitions_dnf.setdefault((rm_state, true_props),[]).append(dnf_formula)
                        dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]

                    state_guards = [theta[2] for dnf_formula in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[rm_state][dnf_formula]]

                    curr_state_guard = random.choice(state_guards)

                    curr_integral = clock_values
                    guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)

                    delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0, self.env.max_delay) for clock in self.clock_names}
                    # random sample one space from the list of possible clock spaces
                    lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                    upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                    
                    all_possible_delays = range(int(lower_delay), int(upper_delay)+1)
                
                #print('All possible delays:', list(all_possible_delays))
                
                all_possible_new_actions = [np.array([d,env_action]) for d in all_possible_delays]
                all_possible_crms[rm_state] = [(config[1], all_possible_new_actions)]

        elif self.crm_option == 3:
            all_possible_crms = dict()
            current_state, current_clock = config[0], config[1].copy()
            rm_states = [current_state]

            for rm_state in rm_states:
            #for rm_state in range(self.env.num_rm_states):
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                all_possible_crms.setdefault(rm_state, [])
                max_upper_bound = 0 
                max_lower_bound = 4 
                                
                clock_upper_bounds = min(max_upper_bound,int(max([self.env.max_constant_dict[clock] - current_clock[clock] for clock in self.clock_names])))+1
                clock_lower_bounds = -min(max_lower_bound,int(min(current_clock.values())))
                
                shift_dict_list = clock_dict_combinations_same_bounds(self.clock_names, clock_lower_bounds, clock_upper_bounds)

                for s in shift_dict_list:
                    crm_clock = clock_shift_dict(current_clock, s, self.env.max_constant_dict)
                    if any(crm_clock[c] < 0 or crm_clock[c] > self.env.max_constant_dict[c] for c in self.clock_names):
                        continue

                    # collect DNF formulas that fire with current true_props
                    if (rm_state, true_props) in self.env.current_rm.known_transitions_dnf:
                        dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]
                    else:
                        for dnf in self.env.current_rm.outgoing_transitions[rm_state]:
                            if evaluate_dnf(dnf, true_props):
                                self.env.current_rm.known_transitions_dnf.setdefault(
                                    (rm_state, true_props), []
                                ).append(dnf)
                        dnf_formulas = self.env.current_rm.known_transitions_dnf.get((rm_state, true_props), [])

                    if not dnf_formulas:
                        continue

                    state_guards = [theta[2] for dnf in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[rm_state][dnf]]
                    if not state_guards:
                        continue

                    # Build union of feasible delay intervals over all guards
                    guard_intervals = []
                    for guard in state_guards:
                        bounds = extract_bounds(
                            guard, max_constant=self.env.max_constant,
                            global_dtype=np.float32, delta=0.0,
                            clock_names=self.clock_names,
                        )
                        delay_bounds = {
                            c: np.clip(np.array(bounds[c]) - crm_clock[c] - 1, 0, self.env.max_delay) for c in self.clock_names
                        }
                        lo = min(delay_bounds[c][0] for c in delay_bounds)
                        hi = max(delay_bounds[c][1] for c in delay_bounds)

                        # Respect remaining slack until any clock hits max
                        # d_max_clock = min(self.env.max_constant_dict[c] - crm_clock[c] for c in self.clock_names)
                        hi = min(hi, self.env.max_delay)


                        if hi >= lo:
                            guard_intervals.append((int(lo), int(hi)))

                    if not guard_intervals:
                        continue

                    guard_intervals.sort()
                    merged = []
                    for lo, hi in guard_intervals:
                        if not merged or lo > merged[-1][1] + 1:
                            merged.append([lo, hi])
                        else:
                            merged[-1][1] = max(merged[-1][1], hi)
                    #print('Merged intervals:', merged)
                    all_possible_delays = []
                    if self.env.discretization_param == 1:
                        for lo, hi in merged:
                            all_possible_delays.extend(range(int(lo), int(hi) + 1))
                    else:
                        step = self.env.discretization_param
                        for lo, hi in merged:
                            all_possible_delays.extend([float(d) for d in np.arange(lo, hi + step, step)])

                    if not all_possible_delays:
                        continue

                    actions = [np.array([d, env_action]) for d in all_possible_delays]
                    if actions:
                        all_possible_crms[rm_state].append((crm_clock, actions))

        else:
            return []


                # crm_set = [clock_shift_dict(clock_values, shift, self.env.max_constant_dict) for shift in shift_dict_list]
                # #print(crm_set)

                # for crm_clock_values in crm_set:
                #     assert min(crm_clock_values.values()) >= 0
                
                    # min_current_clock_value = min(crm_clock_values.values())
                    # if min_current_clock_value >= self.env.max_constant:
                    #     all_possible_delays = []   
                    # else:
                    #     if (rm_state, true_props) in self.env.current_rm.known_transitions_dnf:
                    #         dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]
                    #     else:
                    #         for dnf_formula in self.env.current_rm.outgoing_transitions[rm_state]:
                    #             if not evaluate_dnf(dnf_formula, true_props):
                    #                 continue
                    #             self.env.current_rm.known_transitions_dnf.setdefault((rm_state, true_props),[]).append(dnf_formula)
                    #         dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]

                    #     state_guards = [theta[2] for dnf_formula in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[rm_state][dnf_formula]]

                    #     curr_state_guard = random.choice(state_guards)

                    #     curr_integral = crm_clock_values
                    #     guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)

                    #     delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0, self.env.max_delay) for clock in self.clock_names}
                    #     # random sample one space from the list of possible clock spaces
                    #     lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                    #     upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                        
                    #     all_possible_delays = range(int(lower_delay), int(upper_delay)+1)
                    
                    # #print('All possible delays:', list(all_possible_delays))
                    # all_possible_new_actions = [np.array([d,env_action]) for d in all_possible_delays]
                    # all_possible_crms[rm_state] = [(crm_clock_values, all_possible_new_actions)]

        rm_id = self.env.current_rm_id
        rm = self.env.current_rm
        #print(all_possible_crms)
        for state in all_possible_crms:
            if state == rm.terminal_state:
                continue
            for crm_clock, actions in all_possible_crms[state]:
                config2 = (state, crm_clock)
                for new_action in actions:
                    exp, next_u = self._get_rm_experience(
                        rm_id, rm, config2, obs, new_action, next_obs, env_done, true_props, info
                    )
                    # if (next_u != state) or (exp[3] >= 0):
                    experiences.append(exp)
        
        # de-duplicate and cap
        seen = set()
        deduped = []
        for e in experiences:
            obs_e, _, action_idx_e, _, _, _ = e
            key = (obs_e.tobytes(), int(action_idx_e))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)

        deduped.sort(key=lambda x: x[3], reverse=True)
        # print(f"CRM experiences generated: {self.crm_nums}, deduped to: {len(deduped)}")
        # for item in deduped[: self.crm_nums]:
        #     print("Deduplicated experience:", item)
        return deduped[: self.crm_nums]

        # for rm_id, rm in enumerate(self.env.reward_machines):
        #     for state in all_possible_crms:
        #         if state == rm.terminal_state:
        #             continue
        #         for clock_values, action_list in all_possible_crms[state]:
        #             for new_action in action_list:
        #                 config = (state, clock_values.copy())
        #                 #print('Config:', config, 'Action:', new_action)
        #                 #print('CRM config:', config, 'Action:', new_action)
        #                 exp, next_u = self._get_rm_experience(rm_id, rm, config, obs, new_action, next_obs, env_done, true_props, info)
        #                 experiences.append(exp)


        # Add only the best N experiences
        # Sort by the highest reward and then by not done
        #experiences.sort(key=lambda x: x[3], reverse=True)

        #print('CRM nums:', self.crm_nums)
        #experiences = experiences[:self.crm_nums]
        #print(f"CRM experiences generated: {len(experiences)}")
        # for exp in experiences:
        #     print('Obs', exp[0], 'Action:', self.env.index_to_action[exp[2]], 'Reward:', exp[3], 'Next Obs:', exp[1], 'Done:', exp[4])
        #         # print(exp)
        # print(f"CRM experiences generated: {len(experiences)}")
        #self.valid_states = reachable_states
        #return experiences