# =============================================================================
# import numpy as np
# 
# def dominates(obj1, obj2):
#     return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
# 
# def epsilon_dominates(obj1, obj2, epsilon=0.05):
#     return np.all(obj1 <= obj2 - epsilon)
# def crowding_distance(population):
#      """
#      Shift-Based Density Estimation (SDE) crowding distance for many objectives.
#      """
#      N, M = population.shape
#      dist = np.zeros(N)
#  
#      for m in range(M):
#          sorted_idx = np.argsort(population[:, m])
#          for i in range(1, N - 1):
#              dist[i] += abs(population[sorted_idx[i+1], m] - population[sorted_idx[i-1], m])
#  
#      return dist
# 
# class PlatEMOArchive:
#     def __init__(self, max_size, obj_clip_threshold=1e3, grid_size=200):
#         self.max_size = max_size
#         self.items = []
#         self.obj_clip_threshold = obj_clip_threshold
#         self.grid_size = grid_size
# 
#     def add(self, positions, objectives):
#         # Filter solusi ekstrem (outlier)
#         filtered = []
#         for pos, obj in zip(positions, objectives):
#             if np.all(np.abs(obj) < self.obj_clip_threshold):
#                 filtered.append({'pos': pos, 'fit': obj})
#         
#         candidates = self.items + filtered
#         self.items = self._non_dominated_filter(candidates)
#         self._filter_by_pareto_bounds()
#         self._limit_size_by_grid()
# 
#     def _non_dominated_filter(self, items):
#         non_dominated = []
#         for i, item in enumerate(items):
#             dominated = False
#             for j, other in enumerate(items):
#                 if i != j and epsilon_dominates(other['fit'], item['fit']):
#                     dominated = True
#                     break
#             if not dominated:
#                 non_dominated.append(item)
#         return non_dominated
# 
#     def _get_pareto_bounds(self):
#         if not self.items:
#             return None, None
#         fits = np.array([item['fit'] for item in self.items])
#         return np.min(fits, axis=0), np.max(fits, axis=0)
# 
#     def _filter_by_pareto_bounds(self, margin=0.1):
#         if len(self.items) < 2:
#             return
#         min_vals, max_vals = self._get_pareto_bounds()
#         refined = []
#         for item in self.items:
#             obj = item['fit']
#             if np.all(obj >= min_vals - margin) and np.all(obj <= max_vals + margin):
#                 refined.append(item)
#         self.items = refined
# 
#     def _limit_size_by_grid(self):
#         if len(self.items) <= self.max_size:
#             return
# 
#         fits = np.array([item['fit'] for item in self.items])
#         norm_fits = (fits - fits.min(axis=0)) / (fits.ptp(axis=0) + 1e-12)
#         grid_coords = np.floor(norm_fits * self.grid_size).astype(int)
# 
#         # Hitung frekuensi tiap grid
#         unique, counts = np.unique(grid_coords, axis=0, return_counts=True)
#         grid_density = dict(zip([tuple(u) for u in unique], counts))
# 
#         # Urutkan berdasarkan density terendah (preferensi ke sel jarang)
#         densities = [grid_density[tuple(coord)] for coord in grid_coords]
#         sorted_idx = np.argsort(densities)
# 
#         self.items = [self.items[i] for i in sorted_idx[:self.max_size]]
# 
#     def get_archive(self):
#         positions = np.array([item['pos'] for item in self.items])
#         objectives = np.array([item['fit'] for item in self.items])
#         return positions, objectives
# 
# =============================================================================

# =============================================================================
# import numpy as np
# from scipy.spatial.distance import cdist
# 
# def dominates(obj1, obj2):
#     """
#     Return True if obj1 dominates obj2 (for minimization problems).
#     """
#     return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
# 
# def epsilon_dominates(obj1, obj2, epsilon=0.05):
#     """
#     Epsilon dominance to improve selection in many-objective optimization.
#     """
#     return np.all(obj1 <= obj2 - epsilon)
# 
# def crowding_distance(fits):
#     """
#     Improved version of crowding distance calculation for many objectives.
#     """
#     N, M = fits.shape
#     if N <= 2:
#         return np.inf * np.ones(N)
# 
#     distance = np.zeros(N)
# 
#     for m in range(M):
#         sorted_idx = np.argsort(fits[:, m])
#         f = fits[sorted_idx, m]
#         distance[sorted_idx[0]] = np.inf
#         distance[sorted_idx[-1]] = np.inf
# 
#         max_f = f[-1]
#         min_f = f[0]
# 
#         if max_f == min_f:
#             continue
# 
#         for i in range(1, len(f) - 1):
#             distance[sorted_idx[i]] += (f[i+1] - f[i-1]) / (max_f - min_f)
# 
#     return distance
# 
# def crowding_distance(population):
#     """
#     Shift-Based Density Estimation (SDE) crowding distance for many objectives.
#     """
#     N, M = population.shape
#     dist = np.zeros(N)
# 
#     for m in range(M):
#         sorted_idx = np.argsort(population[:, m])
#         for i in range(1, N - 1):
#             dist[i] += abs(population[sorted_idx[i+1], m] - population[sorted_idx[i-1], m])
# 
#     return dist
# 
# class PlatEMOArchive:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.items = []
# 
#     def add(self, positions, objectives):
#         # Gabungkan semua solusi (lama dan baru)
#         candidates = self.items + [{'pos': pos, 'fit': fit} for pos, fit in zip(positions, objectives)]
# 
#         # Filter solusi non-dominated
#         self.items = self._non_dominated_filter(candidates)
# 
#         # Jaga ukuran maksimum dengan crowding distance
#         self._limit_size()
# 
#     def _non_dominated_filter(self, items):
#         """
#         Keep only non-dominated solutions from a list of candidate items.
#         Menggunakan fungsi dominates() atau epsilon_dominates()
#         """
#         non_dominated = []
#         for i, item in enumerate(items):
#             dominated = False
#             for j, other in enumerate(items):
#                 if i != j and dominates(other['fit'], item['fit']):
#                     dominated = True
#                     break
#             if not dominated:
#                 non_dominated.append(item)
#         return non_dominated
# 
#     def _limit_size(self):
#         """
#         Batasi jumlah solusi di archive menggunakan SDE crowding distance.
#         """
#         if len(self.items) <= self.max_size:
#             return
# 
#         fits = np.array([item['fit'] for item in self.items])
#         dist = crowding_distance(fits)
# 
#         # Urutkan berdasarkan crowding distance descending
#         idx_sorted = np.argsort(-dist)
# 
#         # Tie-breaker acak
#         temp_list = list(zip(idx_sorted, dist[idx_sorted]))
#         temp_list.sort(key=lambda x: (-x[1], np.random.rand()))  # Break ties secara random
#         idx_sorted = [x[0] for x in temp_list]
# 
#         self.items = [self.items[i] for i in idx_sorted[:self.max_size]]
# 
#     def get_archive(self):
#         """
#         Return arrays of positions and objectives from archive.
#         """
#         positions = np.array([item['pos'] for item in self.items])
#         objectives = np.array([item['fit'] for item in self.items])
#         return positions, objectives
# 
# =============================================================================
import numpy as np

def dominates(obj1, obj2):
    """
    Return True if obj1 dominates obj2 (for minimization problems).
    """
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

def crowding_distance(fits):
    """
    Improved version of crowding distance calculation for many objectives.
    """
    N, M = fits.shape
    if N <= 2:
        return np.inf * np.ones(N)

    distance = np.zeros(N)

    for m in range(M):
        sorted_idx = np.argsort(fits[:, m])
        f = fits[sorted_idx, m]
        distance[sorted_idx[0]] = np.inf  # Boundary points get infinite distance
        distance[sorted_idx[-1]] = np.inf

        max_f = f[-1]
        min_f = f[0]

        if max_f == min_f:
            continue

        # Normalized differences between adjacent solutions
        for i in range(1, len(f) - 1):
            distance[sorted_idx[i]] += (f[i+1] - f[i-1]) / (max_f - min_f)

    return distance

class Archive:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []

    def add(self, positions, objectives):
        """
        Add new candidate solutions to the archive and retain non-dominated ones.
        """
        # Gabungkan semua solusi (lama dan baru)
        candidates = self.items + [{'pos': pos, 'fit': fit} for pos, fit in zip(positions, objectives)]

        # Filter solusi non-dominated
        self.items = self._non_dominated_filter(candidates)

        # Jaga ukuran maksimum dengan crowding distance
        self._limit_size()

    def _non_dominated_filter(self, items):
        """
        Keep only non-dominated solutions from a list of candidate items.
        """
        non_dominated = []
        for i, item in enumerate(items):
            dominated = False
            for j, other in enumerate(items):
                if i != j and dominates(other['fit'], item['fit']):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(item)
        return non_dominated

    def _limit_size(self):
        """
        Limit archive size using improved crowding distance and sorting.
        """
        if len(self.items) <= self.max_size:
            return
    
        fits = np.array([item['fit'] for item in self.items])
        dist = crowding_distance(fits)
    
        # Urutkan berdasarkan crowding distance descending
        idx_sorted = np.argsort(-dist)
        
        # Jika ada tie-breaker (misalnya random), tambahkan
        # Misalnya: acak urutan solusi dengan crowding sama
        temp_list = list(zip(idx_sorted, dist[idx_sorted]))
        temp_list.sort(key=lambda x: (-x[1], np.random.rand()))  # Break ties randomly
        idx_sorted = [x[0] for x in temp_list]
    
        self.items = [self.items[i] for i in idx_sorted[:self.max_size]]

    def get_archive(self):
        """
        Return arrays of positions and objectives from archive.
        """
        positions = np.array([item['pos'] for item in self.items])
        objectives = np.array([item['fit'] for item in self.items])
        return positions, objectives
