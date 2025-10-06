#from reward_functions import *
#from reward_machine_utils import evaluate_dnf, value_iteration
import time
from reward_machines.reward_functions import RewardFunction, ConstantRewardFunction
from reward_machines.reward_machine_utils import evaluate_dnf, value_iteration
import sympy
from sympy import sympify, And
import re
from gymnasium import spaces
import numpy as np
import itertools


def integer_points_between(lower, upper, inclusive=True):
    start = int(np.floor(lower))
    end = int(np.ceil(upper))+1
    return list(range(start, end))

def create_sub_space(guard_space, delay, max_constant, global_dtype=np.float32) -> spaces.Box:
    """
    Creates a box space for guard space.
    """
    new_guard_space = substract_delay_from_bounds(guard_space, delay, max_constant)
    low = np.array([v[0] for v in new_guard_space.values()])
    high = np.array([v[1] for v in new_guard_space.values()])
    return spaces.Box(low=low, high=high, dtype=global_dtype)

def extract_bounds(constraint_str, max_constant, delta=0.01, global_dtype=np.float32, clock_names=None):
    """
    Extracts bounds from a constraint string of the form "x > 3 and y < 5 and z >= 2".
    Returns a dictionary with variable names as keys and their bounds as values.
    Adds a small delta for strict inequalities.
    If the constraint string is empty, returns full bounds for all clocks.
    """
    bounds = {}

    # If the constraint string is empty, return full bounds for all clocks
    if not constraint_str:
        if clock_names is None:
            raise ValueError("clock_names must be provided when constraint_str is empty.")
        return {clock: [0, max_constant] for clock in clock_names}

    # Split on 'and'
    clauses = [c.strip() for c in constraint_str.split('and')]
    for clause in clauses:
        # Match variable, operator, value
        m = re.match(r"(\w+)\s*(>=|<=|>|<|==)\s*([-\d.]+)", clause)
        if not m:
            continue
        var, op, val = m.groups()

        if global_dtype == np.float32:
            val = float(val)
            if var not in bounds:
                bounds[var] = [0, max_constant]
            if op == '>':
                bounds[var][0] = max(bounds[var][0], val + delta)
            elif op == '>=':
                bounds[var][0] = max(bounds[var][0], val)
            elif op == '<':
                bounds[var][1] = min(bounds[var][1], val - delta)
            elif op == '<=':
                bounds[var][1] = min(bounds[var][1], val)
            elif op == '==':
                bounds[var][0] = max(bounds[var][0], val)
                bounds[var][1] = min(bounds[var][1], val)

        elif global_dtype == np.int32:
            val = int(val)
            if var not in bounds:
                bounds[var] = [0, max_constant]
            if op == '>':
                bounds[var][0] = max(bounds[var][0], val + 1)
            elif op == '>=':
                bounds[var][0] = max(bounds[var][0], val)
            elif op == '<':
                bounds[var][1] = min(bounds[var][1], val - 1)
            elif op == '<=':
                bounds[var][1] = min(bounds[var][1], val)
            elif op == '==':
                bounds[var][0] = max(bounds[var][0], val)
                bounds[var][1] = min(bounds[var][1], val)
        
        # Add full bounds for any clock not mentioned in the constraints
    if clock_names:
        for clock in clock_names:
            if clock not in bounds:
                bounds[clock] = [0, max_constant]

    return bounds

def substract_delay_from_bounds(bounds, delay, max_constant):
    """
    Subtracts a delay from the bounds.
    """
    new_bounds = {}
    for var, (lower, upper) in bounds.items():
        new_lower = max(0, lower - delay)
        new_upper = upper if upper == max_constant else max(0, upper - delay)
        new_bounds[var] = [new_lower, new_upper]
    return new_bounds


def find_corner_spaces(bounds, corner_range=0.1):
    """
    Finds the corner spaces of a given box space.
    """
    var_corner_bounds = {}

    for var, (lower, upper) in bounds.items():
        #print(var, lower, upper)
        for integer_points in integer_points_between(lower, upper, inclusive=True):
            #print(f"Integer points for {var}: {integer_points}")
            if var not in var_corner_bounds:
                var_corner_bounds[var] = []
            var_corner_bounds[var].append([integer_points-corner_range, integer_points + corner_range])
        #print(f"Variable {var} corner bounds:", var_corner_bounds[var])
        var_corner_bounds[var][0][0] = lower
        var_corner_bounds[var][-1][1] = upper

    #print("Final Variable corner bounds:", var_corner_bounds)

    clock_names = list(var_corner_bounds.keys())
    values = [var_corner_bounds[k] for k in clock_names]

    # Generate all combinations
    combinations = list(itertools.product(*values))

    # Each combination is a tuple of intervals, one per variable
    # To get as dicts:
    all_clock_spaces = [dict(zip(clock_names, combo)) for combo in combinations]

    return all_clock_spaces


class TimedRewardMachine:
    def __init__(self, file, global_dtype=np.float32):
        # <states,init_state,transitions,(reward_state, reward_transition)>
        self.states  = []         # list of non-terminal RM states
        self.init_state = None       # initial state
        self.transitions    = [] # list of transitions
        self.outgoing_transitions = {} # list of outgoing transitions for each state and each dnf formula
        self.reward_state    = {} #  state-based reward function
        self.reward_transition = {} # transition-based reward function
        self.clock_names = []  # List of clock names
        self.global_dtype = global_dtype  # Data type for the clock values
        self.invalid_transition_reward = ConstantRewardFunction(-100)
        self._load_reward_machine(file)
        self.known_transitions_dnf = {} # Auxiliary variable to speed up computation of the next RM state

        #self.max_constant = 9  # Maximum delay for the RM transitions (in seconds)
         # Reward for invalid transitions (when no transition is defined for the current state and true_props)


    # Public methods -----------------------------------
    
    # def add_reward_shaping(self, gamma, rs_gamma):
    #     """
    #     It computes the potential values for shaping the reward function:
    #         - gamma(float):    this is the gamma from the environment
    #         - rs_gamma(float): this gamma that is used in the value iteration that compute the shaping potentials
    #     """
    #     self.gamma    = gamma
    #     self.potentials = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, rs_gamma)
    #     for u in self.potentials:
    #         self.potentials[u] = -self.potentials[u]


    def reset(self, options=None):
        # Returns the initial state and resets the clocks
        clock_values = {}
        for clock in self.clock_names:
            clock_values[clock] = 0
        self.known_transitions_dnf = {}

        # if options['random']:
        #     state = np.random.choice([s for s in self.states if s != self.terminal_state])
        # else:
        #     state = self.init_state
        
        state = self.init_state
        return (state, clock_values)  # Return the initial state and clock values
    
    # def _compute_next_state(self, u1, true_props):
    #     has_matching_dnf = False
    #     for u2 in self.delta_u[u1]:
    #         dnf_formula, guard, reset = self.delta_u[u1][u2]

    #         # Step 1: Check DNF formula first
    #         if not evaluate_dnf(dnf_formula, true_props):
    #             continue
    #         has_matching_dnf = True

    #         # Step 2: If DNF matched, now check guard
    #         if guard is None or self._evaluate_guard(guard):
    #             # Apply reset and return destination
    #             if reset:
    #                 self._apply_reset(reset)
    #             return u2

    #     # Step 3: If DNF matched but guard failed → sink state
    #     if has_matching_dnf:
    #         return self.terminal_u

    #     # Step 4: No matching DNF → sink state
    #     return self.terminal_u


    # def _compute_next_state(self, u1, true_props):
    #     for u2 in self.delta_u[u1]:
    #         dnf_formula, guard, reset = self.delta_u[u1][u2]
    #         if evaluate_dnf(dnf_formula, true_props):
    #             if guard is None or self._evaluate_guard(guard):
    #                 if reset:
    #                     self._apply_reset(reset)
    #                 return u2
    #     return self.terminal_u # no transition is defined for true_props

    # def get_next_state(self, u1, true_props):
    #     if (u1,true_props) not in self.known_transitions:
    #         u2 = self._compute_next_state(u1, true_props)
    #         self.known_transitions[(u1,true_props)] = u2
    #     return self.known_transitions[(u1,true_props)]

    
    
    def step(self, config, true_props, time_el, s_info, add_rs=False, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """
        u1 = config[0]  # Current state of the RM
        clock_values = config[1].copy()  # Clock values (if any)
        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_state, "the RM was set to a terminal state!"

        valid_transition = None
        if (u1, true_props) in self.known_transitions_dnf:
            dnf_formulas = self.known_transitions_dnf[(u1, true_props)]
        else:
            for dnf_formula in self.outgoing_transitions[u1]:
                #print(f"[RM CHECK] Matching DNF: {dnf_formula} against props: {true_props}")

                if not evaluate_dnf(dnf_formula, true_props):
                    continue
                # Incrementing the clocks
                self.known_transitions_dnf.setdefault((u1, true_props),[]).append(dnf_formula)
            dnf_formulas = self.known_transitions_dnf[(u1, true_props)]                

        for clock in self.clock_names:
            clock_values[clock] += time_el

        for dnf_formula in dnf_formulas:
            for trans in self.outgoing_transitions[u1][dnf_formula]:
                guard = trans[2]  # Guard condition
                reset = trans[3]  # Reset condition
                reward = trans[4].get_reward(s_info)
                if not self._evaluate_guard(guard, clock_values):
                    invalid_reward = (reward/2) if reward>0 else reward*2
                    continue
                # If the guard is satisfied, we apply the reset and return the next state
                if reset:
                    self._apply_reset(reset, clock_values)

                valid_transition = trans
                break
        
        state_rew = self.reward_state[u1]
        if valid_transition:
            # If no valid transition was found, we go to the terminal state with worst reward
            transition_rew = valid_transition[4].get_reward(s_info)
            u2 = valid_transition[5]
        else:
            transition_rew = self.invalid_transition_reward.get_reward(s_info)
            #transition_rew = invalid_reward 
            #u2 = u1  # stay in the same state if no valid transition is found
            u2 = self.terminal_state         
            
        #print(f"[RM REWARD] Transitioning: u1={u1} → u2={u2}, reward={rew}")
        done = (u2 == self.terminal_state)
        
        rew_pair = (state_rew, transition_rew)
        #print(f"u1: {u1}, u2: {u2}, true_props: {true_props}, done: {done}, rew: {rew}")  # Debugging line
        #print(f"u1: {u1}, u2: {u2}, true_props: {true_props}") # Debugging line
        
        # Getting the reward
        #rew = self._get_reward(u1,u2,s_info,add_rs, env_done)
        
        next_config = (u2, clock_values)  # Next state and updated clock values
        
        return next_config, rew_pair, done


    def get_states(self):
        return self.states

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]

    # def get_clock_values(self):
    #     """
    #     Returns the clock names and their current values.
    #     """
    #     return self.clocks
    
    # Private methods -----------------------------------

    # def _get_reward(self,u1,u2,s_info,add_rs,env_done):
    #     """
    #     Returns the reward associated to this transition.
    #     """
    #     # Getting reward from the RM
    #     reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
    #     if u1 in self.delta_r and u2 in self.delta_r[u1]:
    #         reward = self.delta_r[u1][u2].get_reward(s_info)
    #     # Adding the reward shaping (if needed)
    #     rs = 0.0
    #     if add_rs:
    #         un = self.terminal_u if env_done else u2 # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
    #         rs = self.gamma * self.potentials[un] - self.potentials[u1]
    #     # Returning final reward
    #     return reward + rs


    def _load_reward_machine(self, file):
        """
        Example:
        x,y
        0           # initial state
        [2]         # terminal state
        (0, '!w', "", "", ConstantRewardFunction(0), 0)
        (0, '!w&!g', "x > 2", "x", ConstantRewardFunction(-1), 1)
        (0, 'w', "", "", ConstantRewardFunction(-1), 1)
        (1, '!w&!g', "x < 1 and y > 5", "y", ConstantRewardFunction(-1), 0)
        (1, '!w&!g', "x > 3 and y < 2", "y", ConstantRewardFunction(-10), 1)
        (1, 'w', "", "", ConstantRewardFunction(-10), 1)
        (0, 'g', "x > 5", "x,y", ConstantRewardFunction(10), 2)
        (1, 'g', "", "", ConstantRewardFunction(10), 2)                      
        """  
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        
        # Extracting clock names from the first line
        self.clock_names = [clock.strip() for clock in lines[0].split(",") if clock.strip()]
        
        # setting the DFA
        self.init_state = eval(lines[1])
        self.terminal_state = 0
        
        # Adding the state-based reward function
        state_costs = eval(lines[2])
        self.reward_state = {u: c for (u,c) in state_costs}
        self.states = list(self.reward_state.keys())
        
        # adding transitions
        for e in lines[3:]:
            # Support 6 tuple transitions
            tup = eval(e)

            if len(tup) == 6:
                u1, dnf_formula, guard, reset, transition_reward, u2 = tup
            else:
                raise ValueError("Transition must have 4 or 6 elements")
            
            if u1 == self.terminal_state:
                self.invalid_transition_reward = transition_reward
                continue

            self.transitions.append(tup)
        
            if u1 not in self.outgoing_transitions:
                self.outgoing_transitions[u1] = {}
            if dnf_formula not in self.outgoing_transitions[u1]:
                self.outgoing_transitions[u1][dnf_formula] = []
            self.outgoing_transitions[u1][dnf_formula].append(tup)

            # Adding rewards to states
            self.reward_transition[tup] = transition_reward

        self.max_constant = 0 # Hardocing max_constant for now, can be changed later
        self.max_constant_dict = {clock: 0 for clock in self.clock_names}
        # Calculating spaces defined by the guard
        self.max_delay = 0
        self.guard_space = {}
        for tup in self.transitions:
            clauses = [c.strip() for c in tup[2].split('and')]
            for clause in clauses:
                # Match variable, operator, value
                m = re.match(r"(\w+)\s*(>=|<=|>|<|==)\s*([-\d.]+)", clause)
                if not m:
                    continue
                var, op, val = m.groups()
                self.max_constant = max(self.max_constant, int(val))
                self.max_constant_dict[var] = max(self.max_constant_dict[var], int(val))
                if op == '>':
                    self.max_delay = max(self.max_delay, int(val))
                elif op == '>=':
                    self.max_delay = max(self.max_delay, int(val)-1)
                elif op == '==':
                    self.max_delay = max(self.max_delay, int(val)-1)
                # elif op == '<':
                #     self.max_delay = max(self.max_delay, int(val) - 1)
                # elif op == '<=':
                #     self.max_delay = max(self.max_delay, int(val))

        self.max_constant = int(self.max_constant) + 1
        self.max_constant_dict = {clock: int(self.max_constant_dict[clock]) + 1 for clock in self.clock_names}
        #self.max_delay = int(self.max_constant)-1
        

        for tup in self.transitions:
            guard = tup[2]
            if guard:
                self.guard_space[guard] = extract_bounds(guard, max_constant=self.max_constant, delta=0.001, global_dtype=self.global_dtype)
            else:
                self.guard_space[guard] = {clock: [0, self.max_constant] for clock in self.clock_names}

    def __str__(self):
        return f"TimedRewardMachine(states={self.states}, init_state={self.init_state}, transitions={self.transitions}, reward_state={self.reward_state}, reward_transition={self.reward_transition}, clocks={self.clocks})"
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.U)


    def _evaluate_guard(self, guard, clock_values):
        # Example: guard = "x > 3 and y < 5"
        # Use self.clocks dict for variable values
        if not guard:
            return True
        return eval(guard, {}, clock_values)

    def _apply_reset(self, reset, clock_values):
        # Example: reset = "x,y"
        if reset is None or reset == "":
            return
        for clock in reset.split(","):
            clock = clock.strip()
            clock_values[clock] = 0


# Example usage
# t0 = time.time()
# rm = TimedRewardMachine("example_trm2.txt", global_dtype=np.int32)
# print("Clocks:", rm.clock_names)
# print("States:", rm.states)
# print("Max constant:", rm.max_constant)
# #print("Initial state:", rm.u0)
# print("Transitions:", rm.transitions)
# print("Outgoing transitions:", rm.outgoing_transitions)
# print("Reward state:", rm.reward_state)
# print("Reward transition:", rm.reward_transition)
# #print("Guard space:", rm.guard_space)
# print(rm.invalid_transition_reward)
# print("Time taken to load RM:", time.time() - t0)

# Example usageu1 = rm.reset()
# init_config = rm.reset()
# X_config, rew1, done1 = rm.step(init_config, "a", 3.5, {})
# XX_config, rew2, done2 = rm.step(X_config, "b", 2.0, {})
# print(f"Step: config1={init_config}, config2={X_config}, reward={rew1}, done={done1}")
# print(f"Step: config2={X_config}, config3={XX_config}, reward={rew2}, done={done2}")


# test_bounds = extract_bounds("x >= 3 and y < 5", max_constant=5, delta=0.001, global_dtype=np.int32)
# print("Extracted bounds:", test_bounds)
# new_test_bounds = substract_delay_from_bounds(test_bounds, 1, 5)
# print("Bounds after subtracting delay:", new_test_bounds)
#var_corner_points = find_corner_spaces(test_bounds, corner_range=0.1)
#print("Variable corner points:", var_corner_points)

