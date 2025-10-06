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
import time
from itertools import product
import random
from reward_machines.timed_reward_machine_corner_abstraction import CornerAbstractionTimedRewardMachine, create_region_space, create_sub_space, find_corner_spaces, extract_bounds
from env.grid_env_ct import GridWorldEnvCT
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
    return [dict(zip(clock_names, vals)) for vals in product(rng, repeat=len(clock_names))]

class CornerAbstractionTimedRewardMachineEnvGym(gym.Wrapper):
    def __init__(self, env, rm_files, gamma, seed=None):
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

        # Loading the reward machines
        self.global_dtype = np.int32  # Global datatype for the RM states and actions
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        self.gamma = gamma # discount factor for the RM rewards
        for rm_file in rm_files:
            rm = CornerAbstractionTimedRewardMachine(rm_file, global_dtype=self.global_dtype)
            self.num_rm_states += len(rm.states)
            self.reward_machines.append(rm)
            self.max_constant_dict = rm.max_constant_dict
            self.num_clocks = max(len(rm.clock_names) for rm in self.reward_machines)  # Maximum number of clocks across all RMs
            self.max_delay = max(rm.max_delay for rm in self.reward_machines)  # Maximum delay
            self.clock_names = rm.clock_names  # Assuming all RMs have the same clock names
        self.max_constant = int(rm.max_constant) if self.global_dtype == np.int32 else float(self.max_constant)
        

        self.num_rms = len(self.reward_machines)
        self.add_crm = False  # Counterfactual experience is not used by default
        self.add_rs = False   # Reward shaping is not used by default
        self.rm_one_hot = False  # RM state is represented as a one-hot vector by default
        self.steps = 0
        self.max_steps = 1000

        self.region_list = create_region_space(self.reward_machines[0].clock_names, self.max_constant_dict)

        self.index_to_region = {}
        self.region_to_index = {}
        for i, region in enumerate(self.region_list):
            self.index_to_region[i] = region
            self.region_to_index[region] = i
        self.num_regions = i

        ###################### Creating observation space ######################
        self.features_space = spaces.Box(
            low=0 if hasattr(env.observation_space, 'n') else env.observation_space.low,
            high=env.observation_space.n-1 if hasattr(env.observation_space, 'n') else env.observation_space.high,
            shape=(1,) if hasattr(env.observation_space, 'n') else env.observation_space.shape,
            dtype=self.global_dtype
        )
        
        #self.feature_space = env.observation_space
        self.rm_state_space = spaces.MultiBinary(self.num_rm_states) if self.rm_one_hot else spaces.Box(
                low=0,
                high=self.num_rm_states - 1,
                shape=(1,),
                dtype=self.global_dtype
            )
        self.region_space = spaces.Box(
            low=0,
            high=self.num_regions,
            shape=(1,),
            dtype=self.global_dtype
        )
        self.corner_space = spaces.Box(
            low=np.array([0 for clock in self.clock_names], dtype=self.global_dtype),
            high=np.array([self.max_constant_dict[clock] for clock in self.clock_names], dtype=self.global_dtype),
            shape=(self.num_clocks,),
            dtype=self.global_dtype
        )

        self.observation_dict = spaces.Dict({
            'features': self.features_space,
            'rm-state': self.rm_state_space,
            'regions': self.region_space,
            'corners': self.corner_space,
        })
        print(self.observation_dict)
        # Flattening the observation space for RL algorithms
        obs_flatdim = gym.spaces.flatdim(self.observation_dict)
        obs_low = np.concatenate([
            np.atleast_1d(self.features_space.low),
            np.full(self.num_rm_states, 0, dtype=self.global_dtype) if self.rm_one_hot else np.full(1, 0, dtype=self.global_dtype),
            self.region_space.low,
            self.corner_space.low
        ])
        obs_high = np.concatenate([
            np.atleast_1d(self.features_space.high),
            #np.array([self.features_space.n-1]),
            np.full(self.num_rm_states, 1, dtype=self.global_dtype) if self.rm_one_hot else np.full(1, self.num_rm_states - 1, dtype=self.global_dtype),
            self.region_space.high,
            self.corner_space.high
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(obs_flatdim,), dtype=self.global_dtype)
        ###################################################################

        ###################### Creating action space ######################
        # self.env_action_space = spaces.Box(low=0, high=env.action_space.n-1, shape=(1,), dtype=self.global_dtype) if self.global_dtype == np.int32 else env.action_space
        # print(f"Env action space: {self.env_action_space}")
        # self.successor_space = spaces.Box(
        #     low=np.array([0], dtype=self.global_dtype),
        #     high=np.array([2*self.num_clocks], dtype=self.global_dtype),
        #     shape=(1,),
        #     dtype=self.global_dtype
        # )
        # self.delay_space = spaces.Box(
        #     low=np.array([0], dtype=self.global_dtype),
        #     high=np.array([self.max_constant], dtype=self.global_dtype),
        #     shape=(1,),
        #     dtype=self.global_dtype
        # )
        # action_flatdim = gym.spaces.flatdim(self.action_dict)
        # if self.global_dtype == np.int32:
        #     low_array = np.array([0, 0, 0])
        #     high_array = np.array([self.max_constant, 2*self.num_clocks, self.env.action_space.n - 1])
        # else:
        #     low_array = np.concatenate([self.delay_space.low,  self.env_action_space.low])
        #     high_array = np.concatenate([self.delay_space.high, self.env_action_space.high])
        # self.action_space = spaces.Box(low=low_array, high=high_array, shape=(action_flatdim,), dtype= self.global_dtype)

        self.env_action_space = env.action_space
        self.successor_space = spaces.Discrete(2*self.num_clocks + 1)  # Including the option of not resetting any clock
        #self.delay_space = spaces.Discrete(self.max_constant)  # Including the option of not delaying
        self.delay_space = spaces.Discrete(self.max_delay+1)

        print('Delay upper bound:', self.delay_space.n)
        print('Successor upper bound:', self.successor_space.n)
        print('Env action upper bound:', self.env_action_space.n)

        print(f"Delay space: {self.delay_space}")
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
        self.action_combinations = [(d, succ, a)
                                    for a in range(self.env_action_space.n)
                                    for d in range(self.delay_space.n)
                                    for succ in range(self.successor_space.n)
                                    ]

        self.action_to_index = {action: idx for idx, action in enumerate(self.action_combinations)}
        self.index_to_action = {idx: action for idx, action in enumerate(self.action_combinations)}
        self.num_actions = len(self.action_combinations)

        # Create a new Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)
        env.action_space.seed(seed)
        #print(f"Action space: {self.action_space}")

        # Computing normal encodings for the non-terminal RM states
        if self.rm_one_hot:
            self.rm_state_features = {}
            for rm_id, rm in enumerate(self.reward_machines):
                for u_id in rm.states:
                    one_hot = np.zeros(self.num_rm_states, dtype=np.float32)
                    one_hot[rm_id * len(rm.states) + u_id] = 1.0
                    self.rm_state_features[(rm_id, u_id)] = one_hot
            self.rm_done_feat = np.zeros(self.num_rm_states, dtype=np.float32)
        else:
            self.rm_state_features = {}
            for rm_id, rm in enumerate(self.reward_machines):
                for u_id in rm.states:
                    self.rm_state_features[(rm_id,u_id)] = np.array([u_id])
            self.rm_done_feat = np.array([0]) # for terminal RM states, we give as features an array of zeros

        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm    = None

    def reset(self, seed=None, options=None):
        # Reseting the environment and selecting the next RM tasks
        self.obs, info = self.env.reset(seed=seed, options=None)
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_config  = self.current_rm.reset(options=options)
        self.steps = 0
        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_rm_id, self.current_config, False), info 

    def step(self, action):
        # executing the action in the environment
        action = self.index_to_action[action]
        delay = action[0]
        successor = action[1]
        env_action = int(action[2])

        #print(self.current_config, action)
        next_obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.steps += 1
        # print(f"Next Obs: {next_obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        # print(self.env)
        true_props = self.env.unwrapped.get_events()
        #print('Action before executing', env_action)
        #print(f"Current Obs: {self.obs}, Action: {action}, Next Obs: {next_obs}, True Props: {true_props}")
        #print(f"Current RM: {self.rm_files[self.current_rm_id]}, Current Config: {self.current_config}")

        current_config = (self.current_config[0], self.current_config[1].copy(), self.current_config[2].copy())
        self.crm_params = self.obs, current_config, action, next_obs, truncated, true_props, info
        

        # print(f"Current Config: {self.current_config}, True Props: {true_props}, Delay: {delay}")
        # update the RM state
        region_after_action = current_config[1].shift_region(1)
        corner_after_action = {clock: min(value + 1, self.max_constant_dict[clock]) for clock, value in current_config[2].items()}
        config_after_delay = (current_config[0], region_after_action, corner_after_action)
        
        next_config, rm_rew_pair, rm_done = self.current_rm.step(config_after_delay, true_props, delay, successor, info)

        # bound the config of clocks to the maximum delay
        # for clock_name in next_config[1]:
        #     if next_config[1][clock_name] > self.max_constant:
        #         next_config[1][clock_name] = self.max_constant

        # the total reward is computed as the sum of cost of delaying in a state and the reward of the transition
        state_rm_rew, transition_rm_rew = rm_rew_pair
        rm_rew = ((self.gamma**delay - 1)*(1/np.log(self.gamma))*state_rm_rew) + transition_rm_rew
        terminated = rm_done or terminated
        truncated = truncated or self.steps >= self.max_steps
        #print('RM done:', rm_done, 'Env done:', terminated)
        # if done:
        #     print(f"RM done {rm_done} Env done {env_done}")
        # if true_props==('b',):
        #     print(true_props, delay, rm_rew_pair,rm_rew, next_obs, self.current_config[1])
        self.obs = next_obs
        self.current_config = next_config
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_config, terminated)
        
            
        
        return rm_obs, rm_rew, terminated, truncated, info

    def get_observation(self, next_obs, rm_id, config, done):

        u_id = config[0]
        region_index = self.region_to_index[config[1]]
        corner_values_dict = config[2]

        next_obs = np.asarray(next_obs).flatten()  # Flatten the next observation
        rm_feat = self.rm_done_feat.flatten() if done else self.rm_state_features[(rm_id, u_id)].flatten()
        region_value = np.asarray([region_index], dtype=self.global_dtype).flatten()
        corner_values = np.asarray([corner_values_dict[clock_name] for clock_name in self.current_rm.clock_names], dtype=self.global_dtype).flatten()
        # Get the clock values from the current RM
        #print(f"Next obs: {next_obs}, RM feat: {rm_feat}, Region value: {region_value}, Corner values: {corner_values}")

        # Concatenate all parts of the observation
        rm_obs = np.concatenate([next_obs, rm_feat, region_value, corner_values]).astype(self.global_dtype)

        return rm_obs 


class CornerAbstractionTimedRewardMachineWrapperGym(gym.Wrapper):
    def __init__(self, env, add_crm, add_rs, gamma, rs_gamma, crm_nums, crm_option=1):
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

        return rm_obs, rm_rew, terminated, truncated, info

    def _get_rm_experience(self, rm_id, rm, config, obs, action, next_obs, env_done, true_props, info):
        
        #print(f"State {u_id} - Clock values: {clock_values} - Action: {action}")
        #print(action)
        delay = action[0]
        successor = action[1]

        #print(f"Current Config: {config}")
        
        rm_obs = self.env.get_observation(obs, rm_id, config, env_done)
        region_after_action = config[1].shift_region(1)
        corner_after_action = {clock: min(value + 1, self.env.max_constant_dict[clock]) for clock, value in config[2].items()}
        config_after_delay = (config[0], region_after_action, corner_after_action)
        #print('After action Config :', config_after_delay)
        next_config, rm_rew_pair, rm_done = rm.step(config_after_delay, true_props, delay, successor, info)

        state_rm_rew, transition_rm_rew = rm_rew_pair
        #rm_rew = sum([(self.gamma**t)*state_rm_rew for t in range(delay)]) + (self.gamma**(delay))*transition_rm_rew
        
        
        rm_rew = ((self.gamma**delay - 1)*(1/np.log(self.gamma))*state_rm_rew) + transition_rm_rew

        # bound the config of clocks to the maximum delay
        # for clock_name in next_config[1]:
        #     if next_config[1][clock_name] > self.env.max_constant:
        #         next_config[1][clock_name] = self.env.max_constant

        #print(f"u:{config[0]} u':{next_config[0]} v:{config[1]} v':{next_config[1]} d:{action[0]} a:{action[1]} r:{rm_rew}")
        terminated = rm_done
        rm_next_obs = self.env.get_observation(next_obs, rm_id, next_config, terminated)
        action_index = self.env.action_to_index[tuple(action)]
        info = {}
        # if terminated:
        #     info["final_obs"] = np.array([rm_next_obs])  # Store the terminal observation in info
        #     info["_final_obs"] = np.array([True])
        #     print('Terminated within CRM experience')
        #     print(rm_obs, rm_next_obs, action_index, rm_rew, terminated, info)

        #print(f"Experience: (s, a, r, s', done) = ({rm_obs}, {action}, {rm_rew}, {rm_next_obs}, {terminated})")
        
        return (rm_obs, rm_next_obs, action_index, rm_rew, terminated, info), next_config[0]  # Return the experience and the next RM state id

    def _get_crm_experience(self, obs, config, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        current_state, current_region, current_corner = config[0], config[1].copy(), config[2].copy()
        delay, successor, env_action = action[0], action[1], int(action[2])

        experiences = []

        #print('Inside function', action)
        # 0 for complete random clock values
        # 1 for random clock values that statisfy guard
        # 2 for random clock values that satisfy guard and also corner points

        if self.crm_option == 0:
            # Random clock values
            #print('Max delay:', self.env.max_delay)
            #print("CRM option 0: All possible clock values running")
            all_possible_crms = dict()
            for rm_state in [current_state]: # Activate this for no crm on state u
            #for rm_state in range(self.env.num_rm_states):
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                all_possible_crms[rm_state] = []
                max_upper_bound = 0 # 1 for no crm on R, alpha
                max_lower_bound = 4 # 0 for no crm on R, alpha
                clock_upper_bounds = min(max_upper_bound,max([self.env.max_constant_dict[clock] - current_corner[clock] for clock in self.clock_names]))+1
                clock_lower_bounds = - min(min(current_corner.values()),max_lower_bound)
                
                shift_dict_list = clock_dict_combinations_same_bounds(self.clock_names, clock_lower_bounds, clock_upper_bounds)
                # print('--------------------------------------------------')
                # print('Current region:', current_region)
                # print('Current corner:', current_corner)
                # print('Clock bounds:', clock_lower_bounds, clock_upper_bounds)
                # print('Number of shift combinations:', shift_dict_list)
                # print('Shift dict list:', shift_dict_list)
                #crm_set = [(current_region.shift_region(s), clock_shift(current_corner, s, self.env.max_constant)) for s in range(clock_lower_bounds, clock_upper_bounds)]
                
                #crm_set = [(current_region.shift_region_dict(s), clock_shift_dict(current_corner, s, self.env.max_constant_dict)) for s in shift_dict_list]


                for s in shift_dict_list:
                    crm_region = current_region.shift_region_dict(s)
                    crm_corner = clock_shift_dict(current_corner, s, self.env.max_constant_dict)
                    
                    assert all([(crm_corner[clock]>=0 and crm_region.I[clock]>=0) for clock in self.clock_names])
                    assert all([(0<=crm_corner[clock]-crm_region.I[clock]<=1) for clock in self.clock_names])
                    
                    if any(v!=0 for v in s.values()):
                        if all(crm_region.I[clock] != 0 for clock in self.clock_names):
                            new_I = dict()
                            for clock in self.clock_names:
                                new_I[clock] = crm_region.I[clock]-1 if crm_region.I[clock]!=self.env.max_constant_dict[clock] else crm_region.I[clock]
                            crm_region.I = new_I.copy()
                            # if all(crm_region.I[clock]!=self.env.max_constant_dict[clock] for clock in self.clock_names):
                            #     crm_region.Z = frozenset()
                            #     crm_region.L = [frozenset({'x'}), frozenset({'y'})]
                            # else:
                            #     
                            while not crm_region.upper_region_condition(new_I):
                                crm_region = crm_region.time_successor()

                    all_possible_delays = np.arange(0, min(max([self.env.max_constant_dict[clock] - crm_corner[clock] for clock in self.clock_names]), self.env.max_delay)+1)
                    all_possible_successors = range(0, 2 * len(self.clock_names) + 1)
                    #print('All possible delays:', list(all_possible_delays))
                    all_possible_new_actions = [np.array([d,s,env_action]) for d in all_possible_delays for s in all_possible_successors]
                    all_possible_crms[rm_state].append((crm_region, crm_corner, all_possible_new_actions))
        
        
        elif self.crm_option == 1:
            # Random clock values that satisfy the guard
            
            if all(current_corner[clock] >= self.env.max_constant_dict[clock] for clock in self.clock_names):
                all_possible_delays = []   
            else:
                dnf_formulas = self.env.current_rm.known_transitions_dnf[(current_state, true_props)]
                state_guards = [theta[2] for dnf_formula in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[current_state][dnf_formula]]

                curr_state_guard = random.choice(state_guards)

                curr_integral = current_corner
                guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)

                delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0, self.env.max_delay) for clock in self.clock_names}
                # random sample one space from the list of possible clock spaces
                lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                
                all_possible_delays = range(int(lower_delay), int(upper_delay)+1)
            
            all_possible_successors = range(0, 2 * len(self.clock_names) + 1)
            all_possible_new_actions = [np.array([d,s,env_action]) for d in all_possible_delays for s in all_possible_successors]
            all_possible_crms = {current_state: [(current_region, current_corner, all_possible_new_actions)]}

        elif self.crm_option == 2:

            all_possible_crms = dict()
            for rm_state in range(self.env.num_rm_states):
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                if all(current_corner[clock] >= self.env.max_constant_dict[clock] for clock in self.clock_names):
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

                    curr_integral = current_corner
                    guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)
                    
                    delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0, self.env.max_delay) for clock in self.clock_names}
                    # random sample one space from the list of possible clock spaces
                    lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                    upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                    
                    all_possible_delays = range(int(lower_delay), int(upper_delay)+1)
                
                #print('All possible delays:', list(all_possible_delays))
                all_possible_successors = range(0, 2 * len(self.clock_names) + 1)
                all_possible_new_actions = [np.array([d,s,env_action]) for d in all_possible_delays for s in all_possible_successors]
                all_possible_crms[rm_state] = [(current_region, current_corner, all_possible_new_actions)]

        elif self.crm_option == 3:
            # print("CRM option 3: zone-based running")
            all_possible_crms = dict()
            for rm_state in [current_state]: # Activate this for no crm on state u
            #for rm_state in range(self.env.num_rm_states): # Activate this for crm on all states
                if rm_state == self.env.current_rm.terminal_state:
                    continue
                # all_possible_crms[rm_state] = []
                all_possible_crms.setdefault(rm_state, [])
                max_upper_bound = 0 # 0 for no crm on R, alpha
                max_lower_bound = 4 # 0 for no crm on R, alpha
                clock_upper_bounds = min(max_upper_bound,max([self.env.max_constant_dict[clock] - current_corner[clock] for clock in self.clock_names]))+1
                clock_lower_bounds = -min(int(min(current_corner.values())), max_lower_bound)
                
                shift_dict_list = clock_dict_combinations_same_bounds(self.clock_names, clock_lower_bounds, clock_upper_bounds)

                for s in shift_dict_list:
                    crm_region = current_region.shift_region_dict(s)
                    crm_corner = clock_shift_dict(current_corner, s, self.env.max_constant_dict)
                    
                    # assert all([(crm_corner[clock]>=0 and crm_region.I[clock]>=0) for clock in self.clock_names])
                    # assert all([(0<=crm_corner[clock]-crm_region.I[clock]<=1) for clock in self.clock_names])
                    if any(crm_corner[c] < 0 for c in self.clock_names):
                        continue
                    if any(crm_region.I[c] < 0 for c in self.clock_names):
                        continue
                    if any(not (0 <= (crm_corner[c] - crm_region.I[c]) <= 1) for c in self.clock_names):
                        if any(v!=0 for v in s.values()):
                            if all(crm_region.I[clock] != 0 for clock in self.clock_names):
                                new_I = dict()
                                for clock in self.clock_names:
                                    new_I[clock] = crm_region.I[clock]-1 if crm_region.I[clock]!=self.env.max_constant_dict[clock] else crm_region.I[clock]
                                crm_region.I = new_I.copy()  
                                while not crm_region.upper_region_condition(new_I):
                                    crm_region = crm_region.time_successor()
                

                    # min_current_clock_value = min(crm_corner.values())
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
                    #     # if curr_state_guard == 'x<=9':
                    #     #     print('Selected region:', crm_region)
                    #     #     print('Selected guard:', curr_state_guard)
                    #     curr_integral = crm_corner
                    #     guard_bounds = extract_bounds(curr_state_guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names)

                    #     delay_bounds = {clock: np.clip(np.array(guard_bounds[clock])-curr_integral[clock]-1,0, self.env.max_delay) for clock in self.clock_names}
                    #     # random sample one space from the list of possible clock spaces
                    #     lower_delay = max([delay_bounds[clock][0] for clock in delay_bounds])
                    #     upper_delay = min([delay_bounds[clock][1] for clock in delay_bounds])
                        
                    #     all_possible_delays = range(int(lower_delay), int(upper_delay)+1) # To consider changing the delay action
                    #     all_possible_successors = range(0, 2 * len(self.clock_names) + 1) # To consider changing the successor action
                        
                    #     # all_possible_delays = [delay] # To avoid changing the delay action
                    #     # all_possible_successors = [successor] # To avoid changing the delay action
                        
                    #     all_possible_new_actions = [np.array([d,s,env_action]) for d in all_possible_delays for s in all_possible_successors]
                    #     all_possible_crms[rm_state] = [(crm_region, crm_corner, all_possible_new_actions)]
                                        # Collect all matching DNF formulas for this rm_state
                    if (rm_state, true_props) in self.env.current_rm.known_transitions_dnf:
                        dnf_formulas = self.env.current_rm.known_transitions_dnf[(rm_state, true_props)]
                    else:
                        for dnf in self.env.current_rm.outgoing_transitions[rm_state]:
                            if evaluate_dnf(dnf, true_props):
                                self.env.current_rm.known_transitions_dnf.setdefault((rm_state, true_props), []).append(dnf)
                        dnf_formulas = self.env.current_rm.known_transitions_dnf.get((rm_state, true_props), [])

                    if not dnf_formulas:
                        continue

                    state_guards = [theta[2] for dnf in dnf_formulas for theta in self.env.current_rm.outgoing_transitions[rm_state][dnf]]
                    if not state_guards:
                        continue

                    # Build union of feasible delay intervals over all guards
                    guard_intervals = []
                    for guard in state_guards:
                        guard_bounds = extract_bounds(
                            guard, max_constant=self.env.max_constant, global_dtype=np.float32, delta=0.0, clock_names=self.clock_names
                        )
                        delay_bounds = {
                            c: np.clip(np.array(guard_bounds[c]) - crm_corner[c] - 1, 0, self.env.max_delay) for c in self.clock_names
                        }
                        lo = min(delay_bounds[c][0] for c in delay_bounds)
                        hi = max(delay_bounds[c][1] for c in delay_bounds)

                        # Respect remaining slack until any clock hits max
                        # d_max_clock = min(self.env.max_constant_dict[c] - crm_corner[c] for c in self.clock_names)
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

                    all_possible_delays = []
                    for lo, hi in merged:
                        all_possible_delays.extend(range(int(lo), int(hi) + 1))

                    if not all_possible_delays:
                        continue

                    # Prune successors to reduce explosion: always keep 0 (no reset) + a few random resets
                    total_succ = 2 * len(self.clock_names) + 1
                    candidate_succs = list(range(1, total_succ))
                    k = min(len(self.clock_names), len(candidate_succs))
                    sampled = random.sample(candidate_succs, k) if k > 0 else []
                    succs = [0] + sampled
                    # succs = [successor]

                    actions = [np.array([d, s_idx, env_action]) for d in all_possible_delays for s_idx in succs]
                    if actions:
                        all_possible_crms[rm_state].append((crm_region, dict(crm_corner), actions))

        else:
            # Fallback: no CRM
            return []


        for rm_id, rm in enumerate(self.env.reward_machines):
            for state in all_possible_crms:
                if state == rm.terminal_state:
                    continue
                for crm_region, crm_corner, actions in all_possible_crms[state]:
                    config2 = (state, crm_region, crm_corner)
                    for new_action in actions:
                        exp, next_u = self._get_rm_experience(rm_id, rm, config2, obs, new_action, next_obs, env_done, true_props, info)
                        experiences.append(exp)
        #print(f"Total CRM experiences generated: {len(experiences)}")
        # Sort experiences based on reward
        
        # experiences.sort(key=lambda x: x[3], reverse=True)


        # Add only the best N experiences
        #print('CRM nums:', self.crm_nums)
        # experiences = experiences[:self.crm_nums]
        
        # ---- De-duplicate by (obs, action_idx), then sort and cap ----
        seen = set()
        deduped = []
        for e in experiences:
            obs_e, next_obs_e, action_idx_e, rew_e, done_e, info_e = e
            key = (obs_e.tobytes(), int(action_idx_e))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)

        deduped.sort(key=lambda x: x[3], reverse=True)
        return deduped[: self.crm_nums]
        # print('#################')
        # for exp in experiences:
        #     print('Observation:', exp[0], self.env.index_to_region[exp[0][2]], 'Action:', self.env.index_to_action[exp[2]], 'Reward:', exp[3], 'Next observation:', exp[1], 'Done:', exp[4])


        # print(f"CRM experiences returned: {len(experiences)}")
        # return experiences
    
# t0 = time.time()
# mdp_env = GridWorldEnvCT()
# trm_env = CornerAbstractionTimedRewardMachineEnvGym(mdp_env, rm_files=["example_trm2.txt"], gamma=0.99) 
# print(f"Time to create the environment: {time.time()-t0}")
# action = np.array([0, 0, 1])  # delay, successor, env_action
# first_s = trm_env.reset()
# print(first_s)
# second_s, r, done, truncated, info = trm_env.step(action)
# print('Coord', second_s[:2], 'State', second_s[2], trm_env.index_to_region[second_s[3]], 'Corner', second_s[4:], 'Reward', r)