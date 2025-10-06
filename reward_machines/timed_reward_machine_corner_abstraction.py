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
from typing import Dict, Set, List, FrozenSet, Optional, Union, Tuple, Generator, Iterable
import copy


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


class Region:
    """
    Represents a region in timed automata with integral part, zero set, and ordering.
    
    A region is characterized by:
    - I: Integral part (mapping from clocks to their integer values)
    - Z: Zero set (set of clocks with zero fractional part) 
    - L: Ordering of clocks by fractional parts (list of frozensets)
    """
    
    def __init__(self, integral: Dict[str, int], zero_set: FrozenSet[str], ordering: List[FrozenSet[str]], max_constant_dict: Dict[str, int]):
        """
        Initialize a region.
        
        Args:
            integral: Dictionary mapping clock names to their integral parts
            zero_set: Frozenset of clocks with zero fractional part
            ordering: List of frozensets representing ordering of fractional parts
        """
        self.I = integral.copy()
        self.Z = zero_set
        self.L = [fs.copy() for fs in ordering]
        self.max_constant_dict = max_constant_dict
        
    @classmethod
    def from_dict(cls, region_dict: Dict):
        """Create region from dictionary representation"""
        return cls(
            integral=region_dict['I'],
            zero_set=region_dict['Z'], 
            ordering=region_dict['L']
        )
    
    @classmethod
    def initial_region(cls, clock_names: List[str], max_constant_dict: Dict[str, int]) -> 'Region':
        """
        Create initial region where all clocks are zero.
        
        Args:
            clock_names: List of all clock names
            
        Returns:
            Initial region with all clocks at zero
        """
        integral = {clock: 0 for clock in clock_names}
        zero_set = frozenset(clock_names)
        ordering = []
        return cls(integral, zero_set, ordering, max_constant_dict)


    def to_dict(self) -> Dict:
        """Convert region to dictionary representation"""
        return {
            'I': self.I.copy(),
            'Z': self.Z,
            'L': [fs.copy() for fs in self.L]
        }
    
    def copy(self) -> 'Region':
        """Create a deep copy of the region"""
        return Region(self.I, self.Z, self.L, self.max_constant_dict.copy())
    
    def time_successor(self) -> 'Region':
        """
        Compute the time successor region.
        
        Args:
            max_constant: Maximum constant used in constraints
            
        Returns:
            Time successor region
        """
        new_region = self.copy()
        
        if new_region.Z != frozenset():
            # Case 1: Z is not empty
            new_region.Z = frozenset()
            new_region.L = [self.Z] + new_region.L
        else:
            # Case 2: Z is empty
            new_zero_set = set()
            if new_region.L:
                Lk = new_region.L.pop()
                for clock in Lk:
                    if new_region.I[clock] < self.max_constant_dict[clock] - 1:
                        new_region.I[clock] += 1
                        new_zero_set.add(clock)
                    else:
                        new_region.I[clock] = self.max_constant_dict[clock]
                new_region.Z = frozenset(new_zero_set)
            
        return new_region
    
    def reset_clocks(self, clocks_to_reset: Set[str]) -> 'Region':
        """
        Reset specified clocks to zero.
        
        Args:
            clocks_to_reset: Set of clock names to reset
            
        Returns:
            New region with specified clocks reset
        """
        new_region = self.copy()
        
        # Reset integral parts
        for clock in clocks_to_reset:
            if clock in new_region.I:
                new_region.I[clock] = 0
        
        # Update zero set
        new_zero_set = set(new_region.Z)
        new_zero_set.update(clocks_to_reset)
        
        # Remove reset clocks from ordering
        new_ordering = []
        for group in new_region.L:
            remaining = group - clocks_to_reset
            if remaining:
                new_ordering.append(remaining)
        
        new_region.Z = frozenset(new_zero_set)
        new_region.L = new_ordering
        
        return new_region
    
    def sample_representative_value(self, delta: float = 0.1) -> Dict[str, float]:
        """
        Sample a representative point from the region.

        Args:
            delta: Small value to perturb the sampled point

        Returns:
            Dictionary with clock names as keys and sampled values as values
        """
        sampled_value = {}
        for clock, value in self.I.items():
            sampled_value[clock] = value

        for group in self.L:
            for clock in group:
                if clock in sampled_value:
                    sampled_value[clock] += delta
            delta += delta


        return sampled_value

    def satisfies_constraint(self, constraint: str) -> bool:
        """
        Check if region satisfies a given constraint.
        
        Args:
            constraint: String constraint like "x > 3 and y < 5"
            clock_values: Current clock values
            
        Returns:
            True if constraint is satisfied
        """
        if not constraint:
            return True
        
        try:
            sampled_value = self.sample_representative_value()
            #print(sampled_value)
            return eval(constraint, {}, sampled_value)
        except:
            return False
    
    def upper_region_condition(self, corner: Dict[str, float]) -> bool:
        """
        Check upper region condition for delay successor computation.
        
        Args:
            corner: Corner point
            
        Returns:
            True if condition is satisfied
        """
        if self.Z != frozenset():
            return False
        
        for clock in corner:
            if self.I[clock] != corner[clock]:
                return False
        return True
    
    def shift_region(self, delay: float) -> 'Region':
        """
        Shift the region by a given delay.

        Args:
            delay: Delay amount

        Returns:
            New region shifted by the delay
        """
        new_region = self.copy()
        if delay < 0:
            # introduce clocks to Z with clock_values == max_constant
            for clock in new_region.I:
                if new_region.I[clock] == self.max_constant_dict[clock]:
                    new_region.Z = new_region.Z.union({clock})

        new_region.I = {clock: min(value + delay, self.max_constant_dict[clock]) for clock, value in new_region.I.items()}

        for clock in new_region.I:
            if new_region.I[clock] < 0:
                new_region.I[clock] = 0
                # remove clocks from clock from L and add to Z if not already in Z
                if clock not in new_region.Z:
                    new_region.Z = new_region.Z.union({clock})
                    new_L = []
                    for group in new_region.L:
                        new_group = group - {clock}
                        if new_group:
                            new_L.append(new_group)
                    new_region.L = new_L

            if new_region.I[clock] == self.max_constant_dict[clock]:
                # drop clock from Z and L
                new_region.Z = new_region.Z - {clock}
                new_L = []
                for group in new_region.L:
                    new_group = group - {clock}
                    if new_group:
                        new_L.append(new_group)
                new_region.L = new_L

        return new_region

    def shift_region_dict(self, delay: dict) -> 'Region':
        """
        Shift the region by a given delay.

        Args:
            delay: Delay amount

        Returns:
            New region shifted by the delay
        """
        new_region = self.copy()
        for clock in new_region.I:
        
            if delay[clock] < 0:
                if new_region.I[clock] == self.max_constant_dict[clock]:
                    new_region.Z = new_region.Z.union({clock})

        new_region.I = {clock: min(value + delay[clock], self.max_constant_dict[clock]) for clock, value in new_region.I.items()}

        for clock in new_region.I:
            if new_region.I[clock] < 0:
                new_region.I[clock] = 0
                # remove clocks from clock from L and add to Z if not already in Z
                if clock not in new_region.Z:
                    new_region.Z = new_region.Z.union({clock})
                    new_L = []
                    for group in new_region.L:
                        new_group = group - {clock}
                        if new_group:
                            new_L.append(new_group)
                    new_region.L = new_L

            if new_region.I[clock] == self.max_constant_dict[clock]:
                # drop clock from Z and L
                new_region.Z = new_region.Z - {clock}
                new_L = []
                for group in new_region.L:
                    new_group = group - {clock}
                    if new_group:
                        new_L.append(new_group)
                new_region.L = new_L

        return new_region

    def delay_successor(self, corner: Dict[str, float], delay: float) -> List['Region']:
        """
        Compute delay successor regions.
        
        Args:
            corner: Starting corner point
            delay: Delay amount
            
        Returns:
            List of successor regions
        """
        successor_regions_list = []
        if delay == 0:
            curr_region = self.copy()
            while not curr_region.upper_region_condition(corner):
                #print('Entering upper region condition', curr_region)
                successor_regions_list.append(curr_region)
                curr_region = curr_region.time_successor()
            successor_regions_list.append(curr_region)
            #print('Successor regions:', successor_regions_list)
        else:

            shifted_region = self.shift_region(delay-1)
            shifted_corner = {clock: min(value + delay - 1, self.max_constant_dict[clock]) for clock, value in corner.items()}

            curr_region = shifted_region.copy()
            # upper region condition    
            while not curr_region.upper_region_condition(shifted_corner):
                curr_region = curr_region.time_successor()

            shifted_corner = {clock: min(value + 1, self.max_constant_dict[clock]) for clock, value in shifted_corner.items()}

            while not curr_region.upper_region_condition(shifted_corner):
                successor_regions_list.append(curr_region)
                curr_region = curr_region.time_successor()
                
            successor_regions_list.append(curr_region)
        
        return successor_regions_list
    
    def __eq__(self, other: 'Region') -> bool:
        """Check equality between regions"""
        if not isinstance(other, Region):
            return False
        return (self.I == other.I and 
                self.Z == other.Z and 
                self.L == other.L)
    
    def __hash__(self) -> int:
        """
        Compute a hash for the Region object.

        The hash is based on:
        - The sorted integral values (I) as a tuple of (clock, value).
        - The zero set (Z) as a frozenset.
        - The ordering (L) as a tuple of frozensets.

        Returns:
            int: The hash value of the Region object.
        """
        # Hash the integral values as a tuple of sorted (clock, value) pairs
        integral_hash = hash(tuple(sorted(self.I.items())))

        # Hash the zero set (Z) as a frozenset
        zero_set_hash = hash(self.Z)

        # Hash the ordering (L) as a tuple of frozensets
        ordering_hash = hash(tuple(self.L))

        # Combine the hashes using XOR to ensure uniqueness
        return integral_hash ^ zero_set_hash ^ ordering_hash
    
    def __str__(self) -> str:
        """String representation"""
        return f"Region(I={self.I}, Z={self.Z}, L={self.L}, M={self.max_constant_dict})"
    
    def __repr__(self) -> str:
        return self.__str__()

def ordered_partitions(elements: Iterable, k: int) -> Generator[Tuple[Tuple, ...], None, None]:
    """
    Generate all ordered partitions of the given elements into k non-empty blocks.
    - Order of blocks matters (block #1, block #2, ..., block #k).
    - No block is empty.
    - Within a block, elements appear in the same relative order as in `elements`.

    Yields:
        A tuple of k tuples, each inner tuple is one non-empty block.
    """
    elems = list(elements)
    n = len(elems)
    if not (1 <= k <= n):
        return  # generate nothing

    buckets: List[List] = [[] for _ in range(k)]

    def backtrack(i: int, empty_left: int):
        # Prune: not enough elements left to fill all still-empty buckets
        if n - i < empty_left:
            return
        if i == n:
            if empty_left == 0:
                yield tuple(tuple(b) for b in buckets)
            return

        x = elems[i]
        for j in range(k):
            was_empty = not buckets[j]
            buckets[j].append(x)
            yield from backtrack(i + 1, empty_left - (1 if was_empty else 0))
            buckets[j].pop()

    yield from backtrack(0, k)


def ordered_partitions_n_k(n: int, k: int):
    """Convenience wrapper using elements 0..n-1."""
    return ordered_partitions(range(n), k)


def create_region_space(clock_names, max_constant_dict):
    #print(max_constant)
    value_ranges = [range(max_constant_dict[clock] + 1) for clock in clock_names]

    # Compute the Cartesian product of all value ranges
    all_combinations = itertools.product(*value_ranges)

    # Map each combination to a dictionary with clock names as keys
    integral_values = [dict(zip(clock_names, combination)) for combination in all_combinations]

    num_clocks = len(clock_names)
    region_list = []
    
    for integral in integral_values:
        for i in range(1, num_clocks+1):
            
            clock_names_internal = [clock for clock in clock_names if integral[clock] < max_constant_dict[clock]]
            if len(clock_names_internal) < i:
                continue
            partitions = list(ordered_partitions(clock_names_internal, i))
            for part in partitions:
                zero_set = frozenset(part[0])
                ordering = [frozenset(p) for p in part[1:]]
                region1 = Region(integral, zero_set, ordering, max_constant_dict)
                
                zero_set = frozenset()
                ordering = [frozenset(p) for p in part]
                region2 = Region(integral, zero_set, ordering, max_constant_dict)

                region_list.append(region1)
                region_list.append(region2)

    final_region = Region({clock: max_constant_dict[clock] for clock in clock_names}, frozenset(), [], max_constant_dict)
    region_list.append(final_region)

    return region_list



class CornerAbstractionTimedRewardMachine:
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
        self.invalid_transition_reward = ConstantRewardFunction(-100) # Reward for invalid transitions (when no transition is defined for the current state and true_props)
        self._load_reward_machine(file)
        self.known_transitions_dnf = dict() # Auxiliary variable to speed up computation of the next RM state

        #self.max_constant = 9  # Maximum delay for the RM transitions (in seconds)
        


    def reset(self, options=None):
        # Returns the initial state and resets the clocks
        init_corner = {}
        init_region = Region.initial_region(self.clock_names, self.max_constant_dict)
        #print(f"Initial region: {init_region}")
        for clock in self.clock_names:
            init_corner[clock] = 0
        self.known_transitions_dnf = dict()

        # if options['random']:
        #     state = np.random.choice([s for s in self.states if s != self.terminal_state])
        # else:
        #     state = self.init_state
        
        state = self.init_state
        return (state, init_region, init_corner)
    
    def step(self, config, true_props, time_el, succ, s_info, add_rs=False, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """
        #print(f"[RM STEP] Current config: {config}, true_props: {true_props}, time_el: {time_el}, s_info: {s_info}, add_rs: {add_rs}, env_done: {env_done}")

        u1 = config[0]  # Current state of the RM
        region1 = config[1]
        corner1 = config[2]
        
        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_state, "the RM was set to a terminal state!"

        valid_transition = None
        if (u1, true_props) in self.known_transitions_dnf:
            dnf_formulas = self.known_transitions_dnf[(u1, true_props)]
        else:
            for dnf_formula in self.outgoing_transitions[u1]:
                if not evaluate_dnf(dnf_formula, true_props):
                    continue
                self.known_transitions_dnf.setdefault((u1, true_props),[]).append(dnf_formula)
            dnf_formulas = self.known_transitions_dnf[(u1, true_props)]                
        
        # Computing the time successor region
        #print('Time el:', time_el, 'corner1:', corner1)
        region_successors = region1.delay_successor(corner1, time_el)
        #print(f"Region successors (count={len(region_successors)}): {region_successors}")
        region2 = region_successors[succ % len(region_successors)]
        corner2 = {clock: min(corner1[clock] + time_el, self.max_constant_dict[clock]) for clock in corner1}
        
        #for clock in self.clock_names:
        #    clock_values[clock] += time_el
        #print('----------------------------')
        ##print('Current DNF formulas:', dnf_formulas)
        for dnf_formula in dnf_formulas:
            #print(self.outgoing_transitions[u1][dnf_formula])
            for trans in self.outgoing_transitions[u1][dnf_formula]:
                guard = trans[2]  # Guard condition
                reset = trans[3]  # Reset condition
                reward = trans[4].get_reward(s_info)  # Transition reward
                if not region2.satisfies_constraint(guard):
                    invalid_reward = reward/2 if reward > 0 else reward*2
                    continue
                # If the guard is satisfied, we apply the reset and return the next state
                
                if reset:
                    clocks_to_reset = set([c.strip() for c in reset.split(",") if c.strip()])
                    #print(reset, region2)
                    region2 = region2.reset_clocks(clocks_to_reset)
                    # reset clocks to zero in corner also
                    for clock in clocks_to_reset:
                        corner2[clock] = 0
                valid_transition = trans
                break
        
        state_rew = self.reward_state[u1]
        if valid_transition:
            transition_rew = valid_transition[4].get_reward(s_info)
            u2 = valid_transition[5]
        else:
            #print(f"Invalid transition########################################################")
            transition_rew = self.invalid_transition_reward.get_reward(s_info) # orginal code
            #transition_rew = invalid_reward
            #u2 = u1 # stay in the same state
            u2 = self.terminal_state         
            
        #print(f"[RM REWARD] Transitioning: u1={u1} → u2={u2}, reward={rew}")
        done = (u2 == self.terminal_state)
        
        rew_pair = (state_rew, transition_rew)
        #print(f"u1: {u1}, u2: {u2}, true_props: {true_props}, done: {done}, rew: {rew}")  # Debugging line
        #print(f"u1: {u1}, u2: {u2}, true_props: {true_props}") # Debugging line
        
        # Getting the reward
        #rew = self._get_reward(u1,u2,s_info,add_rs, env_done)
        next_config = (u2, region2, corner2)  # Next state and updated clock values

        return next_config, rew_pair, done


    def get_states(self):
        return self.states

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]


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

        self.max_constant = 0
        self.max_constant_dict = {clock: 0 for clock in self.clock_names}
        self.max_delay = 0
        # Calculating spaces defined by the guard
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
    
        self.max_constant = int(self.max_constant) + 1
        self.max_constant_dict = {clock: val+1 for clock, val in self.max_constant_dict.items()}
        #self.max_delay = int(self.max_constant) - 1 # original code

        # Calculates guard space
        # for tup in self.transitions:
        #     guard = tup[2]
        #     if guard:
        #         self.guard_space[guard] = extract_bounds(guard, max_constant=self.max_constant, delta=0.001, global_dtype=self.global_dtype)
        #     else:
        #         self.guard_space[guard] = {clock: [0, self.max_constant] for clock in self.clock_names}

    def __str__(self):
        return f"CornerAbstractionTimedRewardMachine(states={self.states}, init_state={self.init_state}, transitions={self.transitions}, reward_state={self.reward_state}, reward_transition={self.reward_transition}, clocks={self.clocks})"
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.U)




# t = CornerAbstractionTimedRewardMachine("example_trm2.txt", global_dtype=np.int32)
# starting_config = t.reset()
# print("Initial config:", starting_config)
# next_config, rew, done = t.step(starting_config, true_props=('a',), time_el=2, succ=2, s_info={})
# print("Next config:", next_config, "Reward:", rew, "Done:", done)
# next_config, rew, done = t.step(next_config, true_props=(), time_el=3, succ=1, s_info={})
# print("Next config:", next_config, "Reward:", rew, "Done:", done)
# next_config, rew, done = t.step(next_config, true_props=('b',), time_el=2, succ=2, s_info={})
# print("Next config:", next_config, "Reward:", rew, "Done:", done)

# print(extract_bounds("x==2", max_constant=10, clock_names=['x','y'], global_dtype=np.int32))
# Current region: Region(I={'x': 1, 'y': 3}, Z=frozenset(), L=[frozenset({'y'}), frozenset({'x'})], M=16)
# Current corner: {'x': 2, 'y': 4}
# Shift: -2
# New region: Region(I={'x': 0, 'y': 1}, Z=frozenset(), L=[frozenset({'y'}), frozenset({'x'})], M=16)
# New corner: {'x': 0, 'y': 2}

# r = Region(integral={'x': 1, 'y': 3},
#            zero_set=frozenset(),
#            ordering=[frozenset({'y'}), frozenset({'x'})],
#            max_constant=16)

# print("Original:", r)

# # shift by -2
# r_shifted = r.shift_region_dict({'x': -1, 'y': -2})
# print("Shifted by -2:", r_shifted)