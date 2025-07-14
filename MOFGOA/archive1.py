import math
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
#from nds import ndomsort
import nds


def dominates(obj1, obj2):
    """
    Fast Pareto dominance check with early stopping.
    """
    obj1 = np.asarray(obj1)
    obj2 = np.asarray(obj2)

    better_in_any = False
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False  # Worse in at least one objective → not dominating
        elif obj1[i] < obj2[i]:
            better_in_any = True
    return better_in_any

def non_dominated_filter(fitnesses):
    """
    Return indices of non-dominated solutions.
    """
    n = len(fitnesses)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j:
                if dominates(fitnesses[j], fitnesses[i]):
                    dominated[i] = True
                    break
                elif dominates(fitnesses[i], fitnesses[j]):
                    dominated[j] = True

    return np.where(~dominated)[0]
def density(items):
    """
    add_item_before_crowding should be False.

    Note: follows the format of items from class Archive, and
        add_item_before_crowding should be False in Archive.

    :param items:
    :return:
    """
    # First, calculate distance between points
    # Source: https://stackoverflow.com/questions/40996957/calculate-distance-between-numpy-arrays
    collected_objs = np.array([entry['fit'] for entry in items])

    euclidean_dist = cdist(collected_objs, collected_objs, metric='euclidean')

    # Sort distances in ascending order
    euclidean_dist = np.sort(euclidean_dist, axis=1)

    # k is set to sqrt of archive size
    k = int(math.sqrt(len(items)))

    tmp = 1 / (euclidean_dist[:, k] + 2)

    for i in range(len(items)):
        items[i]['dist'] = tmp[i]

    items.sort(key=lambda x: x["dist"], reverse=True)

    return items


def crowding_distance(items):
    if len(items) == 0:
        return items

    num_objectives = len(items[0]['fit'])
    fits = np.array([item['fit'] for item in items])
    
    # Hilangkan solusi duplikat
    _, unique_indices = np.unique(fits, axis=0, return_index=True)
    items = [items[i] for i in unique_indices]
    fits = fits[unique_indices]

    # Normalisasi
    scaler = MinMaxScaler()
    norm_fits = scaler.fit_transform(fits)

    for i, item in enumerate(items):
        item['dist'] = 0.0
        item['norm_fit'] = norm_fits[i]

    for i in range(num_objectives):
        items.sort(key=lambda x: x['norm_fit'][i])
        items[0]['dist'] = items[-1]['dist'] = float('inf')
        fmin = items[0]['norm_fit'][i]
        fmax = items[-1]['norm_fit'][i]
        if fmax - fmin < 1e-10:
            continue  # Tujuan ini tidak membedakan → skip
        for j in range(1, len(items) - 1):
            d = items[j + 1]['norm_fit'][i] - items[j - 1]['norm_fit'][i]
            items[j]['dist'] += d / (fmax - fmin)

    # Sort dari paling beragam ke paling padat
    items.sort(key=lambda x: -x['dist'])
    return items



def min_distance_indicator(items):
    """
    Calculates fitness of each solution in archive like explained in Section 3.1 of:

    Cui, Yingying, Xi Meng, and Junfei Qiao.
    "A multi-objective particle swarm optimization algorithm based on two-archive mechanism."
    Applied Soft Computing 119 (2022): 108532.
    """
    fitnesses = np.array([item['fit'] for item in items])
    scaler = MinMaxScaler(feature_range=(0, 1))
    fitnesses = scaler.fit_transform(fitnesses)

    for i in range(len(fitnesses)):
        shifted_fitnesses = fitnesses.copy()
        for j in range(len(fitnesses)):
            if i != j:
                shift_dimensions = np.where(shifted_fitnesses[j] > fitnesses[i])
                shifted_fitnesses[shift_dimensions] = fitnesses[shift_dimensions]


class Grid:
    def __init__(self, size):
        self.size = size
        self.item_div = {}
        self.adaptive = True
        self.focus_factor = 0.5  # Lebih fokus pada region yang lebih baik
        
    def calculate(self, items):
        objs = np.array([item['fit'] for item in items])
        
        if self.adaptive:
            min_ = np.min(objs, axis=0) - 1e-10
            max_ = np.max(objs, axis=0) + 1e-10
            
            # Non-linear spacing dengan fokus pada region yang lebih baik
            cutoffs = []
            for i in range(objs.shape[1]):
                # Exponential spacing untuk fokus pada nilai rendah
                space = np.geomspace(1.0, self.focus_factor, self.size + 1)
                space = min_[i] + (max_[i] - min_[i]) * (1 - space)
                cutoffs.append(space)
            cutoffs = np.array(cutoffs).T
        else:
            min_ = np.min(objs, axis=0) - 1e-10
            max_ = np.max(objs, axis=0) + 1e-10
            cutoffs = np.linspace(max_, min_, self.size + 1)

        self.item_div = {}
        item_grid_index = {}
        for i in range(len(items)):
            item_cutoff = [0] * cutoffs.shape[1]
            for row in cutoffs:
                for j in range(len(row)):
                    if items[i]['fit'][j] < row[j]:
                        item_cutoff[j] += 1
            tmp = 0
            for j in range(len(item_cutoff)):
                tmp += item_cutoff[j] * (self.size ** j)

            if str(tmp) not in item_grid_index:
                item_grid_index[str(tmp)] = [i]
            else:
                item_grid_index[str(tmp)].append(i)

            if str(tmp) not in self.item_div:
                self.item_div[str(tmp)] = [items[i]['pos']]
            else:
                self.item_div[str(tmp)].append(items[i]['pos'])

        # Get the most crowded grid index
        max_entry = []
        for k, v in item_grid_index.items():
            if len(v) > len(max_entry):
                max_entry = v

        # Randomly select index from most crowded grid
        rand_idx = random.sample(max_entry, 1)[0]

        # Swap first item in items with item in most crowded grid
        items[rand_idx], items[0] = items[0], items[rand_idx]

        return items


class Archive:
    def __init__(self,
                 capacity,
                 crowding_function=density,
                 track=None,
                 add_item_before_crowding=False,
                 allow_dominated=False):
        """
        Trackable information:
            - (default) 'pos': Position of solution
            - (default) 'fit': Fitness of solution
            - (default) 'dist': Crowding measure of solution (can be for density or crowding distance)
            - 'sum_fit': Sum of fitness
            - 'added_sol_idx': Returns the indices of solutions that were successfully added to archive (in add() function)

        Crowding functions:
            - Density
            - Crowding distance

        :param capacity: Maximum capacity of archive.
        :type capacity: int
        :param crowding_function: Crowding function used to remove archive entry when capacity is met.
        :param track: Additional information to track in archive
        :param add_item_before_crowding: True if item needs to be added to archive and then
            have crowding calculations complete. False if calculation is made first, and then solution replaced.
        :param allow_dominated: Whether to allow dominated solutions to be added to archive if not full,
            where non-dominated solutions have priority.
        """
        if track is None:
            track = []
        self.capacity = capacity
        self.items = []
        self.track = track
        self.crowding_function = crowding_function
        self.add_item_before_crowding = add_item_before_crowding
        self.allow_dominated = allow_dominated

    def add(self, swarm, fitneses):
        """
        This method is to be overridden by a super class to fit the
        wanted archive management system.
        """
        raise NotImplementedError


class ParetoArchive:
    def __init__(self, capacity, problem=None, minx=None, maxx=None, diversity_mechanism='crowding'):
        self.capacity = capacity
        self.items = []
        self.problem = problem
        self.minx = minx
        self.maxx = maxx
        self.diversity_mechanism = diversity_mechanism
        self.elitism_ratio = 0.15  # 15% solusi terbaik dipertahankan
        self.grid_size = 5 if diversity_mechanism == 'grid' else None
        self.grid_bounds = None
        self.grid_cells = {}

    def _update_grid(self):
        """Update grid structure with adaptive resolution."""
        if not self.items:
            return

        fits = np.array([item['fit'] for item in self.items])
        min_fit = np.min(fits, axis=0)- 1e-10
        max_fit = np.max(fits, axis=0)+ 1e-10

        self.grid_bounds = []
        for i in range(fits.shape[1]):
            low_res = np.linspace(min_fit[i], min_fit[i] + 0.2 * (max_fit[i] - min_fit[i]), int(0.8 * self.grid_size))
            high_res = np.linspace(min_fit[i] + 0.3 * (max_fit[i] - min_fit[i]), max_fit[i], self.grid_size - len(low_res) + 1)
            self.grid_bounds.append(np.unique(np.concatenate([low_res, high_res])))

        self.grid_cells = {}
        for idx, item in enumerate(self.items):
            cell_coords = []
            for obj_idx, obj_val in enumerate(item['fit']):
                cell = np.searchsorted(self.grid_bounds[obj_idx], obj_val) - 1
                cell = max(0, min(cell, self.grid_size - 1))
                cell_coords.append(cell)
            cell_key = tuple(cell_coords)
            if cell_key not in self.grid_cells:
                self.grid_cells[cell_key] = []
            self.grid_cells[cell_key].append(idx)

    def _maintain_diversity(self):
        if len(self.items) <= self.capacity:
            return
    
        # 1. Elitism (gunakan rank atau sum(fit))
        elite_size = max(1, int(0.1 * self.capacity))
        self.items.sort(key=lambda x: np.sum(x['fit']))
        elite = self.items[:elite_size]
        rest = self.items[elite_size:]
    
        # 2. Diversification
        if self.diversity_mechanism == 'crowding':
            rest = crowding_distance(rest)
        elif self.diversity_mechanism == 'grid':
            self._update_grid()
            rest = Grid(self.grid_size).calculate(rest)
    
        # 3. Gabungkan kembali
        self.items = elite + rest[:self.capacity - elite_size]


    def add(self, positions, fitnesses):
        """
        Add new solutions to the archive while maintaining:
        1. Non-dominance (only keep non-dominated solutions)
        2. Diversity (when archive is full)
        3. Local refinement on newly added non-dominated solutions
        """
        assert len(positions) == len(fitnesses)

        positions = np.asarray(positions)
        fitnesses = np.asarray(fitnesses)

        nd_indices = non_dominated_filter(fitnesses)
        non_dominated = [{
            'pos': positions[i].copy(),
            'fit': fitnesses[i].copy(),
            'dist': 0.0
        } for i in nd_indices]


        combined = self.items + non_dominated
        combined_fits = [item['fit'] for item in combined]
        nd_combined_indices = non_dominated_filter(combined_fits)
        self.items = [combined[i] for i in nd_combined_indices]


        self._maintain_diversity()

        # Local refinement for new additions
        for item in self.items[-len(non_dominated):]:
            self._local_refinement(item)

        return len(non_dominated)

    def _local_refinement(self, item, steps=10, step_size=0.001):
        if self.problem is None:
            return  # skip if problem evaluator not defined

        best_pos = item['pos'].copy()
        best_fit = item['fit'].copy()

        for _ in range(steps):
            new_pos = best_pos + np.random.normal(0, step_size, len(best_pos))
            new_pos = np.clip(new_pos, self.minx, self.maxx)
            new_fit = self.problem.evaluate(new_pos)

            if dominates(new_fit, best_fit):
                best_pos, best_fit = new_pos, new_fit

        item['pos'] = best_pos
        item['fit'] = best_fit