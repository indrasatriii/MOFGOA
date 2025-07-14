import numpy as np
import random
#from archive import ParetoArchive,crowding_distance,dominates,Grid
from Archive import Archive, dominates,crowding_distance
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol
from Goose import Goose
'''
class FGOA:
    def __init__(self, problem, dim, size, minx, maxx, max_evals, incentive_threshold, archive_size, ngrid, max_iter):
        self.problem = problem
        self.dim = dim
        self.size = size
        self.minx = np.full(self.dim, minx)
        self.maxx = np.full(self.dim, maxx)
        self.max_iter = max_iter
        self.incentive_threshold = incentive_threshold
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)
        self.gbest_score = np.inf
        #self.archive = ParetoArchive(archive_size)
        self.archive = PlatEMOArchive(archive_size)
        self.best_scores = []
        #self.grid = Grid(ngrid)
        self.leader = None 
        self.positions, self.velocities = self.initialize_positions_and_velocities(dim, size, minx, maxx)
        self.p_bests = self.positions.copy()
        self.p_best_fits = self.problem.evaluate(self.positions).copy()
        self.archive.add(self.positions, self.p_best_fits)
        #self.grid.calculate(self.archive.items)
        self.gd_values=[]
        self.max_evals=max_evals
      
        
    def set_leader(self):
        scalar_scores = []
        for goose in self.geese:
            scalar_scores.append(np.sum(goose.score) if isinstance(goose.score, np.ndarray) else goose.score)
        
        leader_index = scalar_scores.index(min(scalar_scores))
        self.leader = self.geese[leader_index]
        self.gbest_score = self.leader.best_score
        self.gbest = np.copy(self.leader.best_position)
    
    def evaluate(self):
        # Ambil semua posisi dalam bentuk array 2D
        positions = np.array([goose.position for goose in self.geese])
    
        # Evaluasi SEKALIGUS (harus didukung oleh problem.evaluate())
        scores = self.problem.evaluate(positions)
    
        # Update tiap goose secara individual (masih perlu loop, tapi evaluasinya sudah cepat)
        for i, goose in enumerate(self.geese):
            goose.score = scores[i].flatten()
    
            # Inisialisasi best_score jika belum ada
            if not hasattr(goose, 'best_score') or isinstance(goose.best_score, (int, float)):
                goose.best_score = np.full_like(goose.score, np.inf)
    
            # Update best_position dan best_score jika skor lebih baik
            if dominates(goose.score, goose.best_score):
                goose.best_score = goose.score.copy()
                goose.best_position = goose.position.copy()
    
        # Update archive dengan seluruh populasi
        self.archive.add(positions, scores)
        """
        def evaluate(self):
            for goose in self.geese:
                goose.score = self.problem.evaluate(goose.position).flatten()
                if not hasattr(goose, 'best_score') or isinstance(goose.best_score, (int, float)):
                    goose.best_score = np.full_like(goose.score, np.inf)
                if dominates(goose.score, goose.best_score):
                    goose.best_score = goose.score.copy()
                    goose.best_position = goose.position.copy()
        
            # Update archive dengan semua posisi dan score
            positions = [goose.position for goose in self.geese]
            scores = [goose.score for goose in self.geese]
            self.archive.add(positions, scores)
        """
    def initialize_positions_and_velocities(self, dim, size, minx, maxx):
        sobol = Sobol(d=dim, scramble=True)
        samples = sobol.random(n=size)  # Lebih sederhana dan akurat
        positions = minx + samples * (maxx - minx)
        velocities = np.random.uniform(low=-0.1, high=0.5, size=(size, dim))
        return positions, velocities

    def assist_lagging_geese(self):
            archive_positions, _ = self.archive.get_archive()
            idx = np.random.choice(len(archive_positions), size=self.size // 2, replace=True)
            selected = archive_positions[idx]
        
            for i in range(self.size // 2):
                self.geese[i].position = selected[i].copy()

    
    def incentive(self):
        # 計算激勵閾值
        #incentive_threshold = np.abs(np.mean([p.score for p in self.geese]) - self.gbest_score)
        
        # 隨機選擇50%的粒子進行激勵
        num_incentives = int(0.5 * len(self.geese))  # 選擇50%的粒子進行激勵
        incentive_indices = np.random.choice(range(len(self.geese)), size=num_incentives, replace=False)
        
        # 對選中的粒子進行激勵
        for idx in incentive_indices:
            r = np.random.rand(self.dim)  # 隨機數
            # 更新激勵後的粒子位置
            self.geese[idx].position = r * self.gbest + (1 - r) * self.geese[idx].position
    
    def adaptive_inertia(self, iteration):
        si = 0.8
        send = iteration / self.max_iter
        if 0.3 <= send <= 0.9:
            return 0.8 + 0.2 * (send - 0.4) / 0.4 
        
        inertia = send + (si - send)*(1-iteration/self.max_iter)
        return inertia
        
    """
    def change_leader(self):
        if len(self.archive.items) == 0:
            # Gunakan goose terbaik dari populasi
            best_goose = min(self.geese, key=lambda x: np.sum(x.score) if isinstance(x.score, np.ndarray) else x.score)
            if self.leader is None:
                self.leader = Goose(self.dim, self.minx[0], self.maxx[0])
            self.leader.position = best_goose.position.copy()
            return
    
        # Hitung crowding distance untuk archive items
        fits = np.array([item['fit'] for item in self.archive.items])
        dist = cdist(fits, fits)
        np.fill_diagonal(dist, np.inf)
        nearest_indices = np.argmin(dist, axis=1)
    
        crowding = np.zeros(len(fits))
        for i in range(len(fits)):
            crowding[i] = np.linalg.norm(fits[i] - fits[nearest_indices[i]])
    
        # Tambahkan crowding ke item
        for i, item in enumerate(self.archive.items):
            item['crowding'] = crowding[i]
    
        # Pilih leader berdasarkan crowding tertinggi
        leader_item = max(self.archive.items, key=lambda x: x.get('crowding', 0))
    
        # Jika leader belum ada, buat dulu
        if self.leader is None:
            self.leader = Goose(self.dim, self.minx[0], self.maxx[0])
    
        self.leader.position = leader_item['pos'].copy()
    """
    def whiffling_exploitation(self, particle, experience_level=0.5):
        # Arah menuju gbest
        if np.random.rand() < 0.2:
            direction_to_target = -(self.gbest - particle.position)  # Reverse
        else:
            direction_to_target = self.gbest - particle.position
    
        experience_level = particle.experience / self.max_iter if hasattr(particle, 'experience') else 0.5

        # === EKSPLORASI AKTIF UNTUK ANGSA MUDA ===
        if experience_level < 0.2:
            # Gunakan random walk atau variasi besar
            exploration_step = np.random.uniform(-1, 1, size=self.dim) * (1 - experience_level)
            particle.position += exploration_step
        particle.position = np.clip(particle.position, self.minx, self.maxx)
    
    def update_geese(self, current_iter, exploitation_prob=5):
        for goose in self.geese:
            if np.random.rand() < exploitation_prob:
                new_position = goose.position + np.random.uniform(-0.1, 0.1, size=self.dim)
                new_position = np.clip(new_position, self.minx, self.maxx)
                new_score = self.problem.evaluate(new_position)
                if dominates(new_score, goose.score):
                    goose.position = new_position
                    goose.score = new_score
                    if dominates(new_score, goose.best_score):
                        goose.best_score = new_score
                        goose.best_position = np.copy(new_position)

            inertia = self.adaptive_inertia(current_iter)
            r1 = np.random.uniform(0.0, 1, size=self.dim)
            r2 = np.random.uniform(0.0, 1, size=self.dim)
            az = np.random.uniform(0, 2)
            A = 2 * (r1 * az) - az
            C = 2 * r2
            #goose.use_updraft()
            if np.random.rand() < 0.2:
                goose.use_updraft_front_neighbor(goose, self.gbest, self.geese)
            else:
                goose.use_updraft_front_neighbor(goose, self.gbest, self.geese)

            goose.velocity = goose.velocity + A * abs(C * (goose.best_position - goose.position))
            goose.velocity = (
                inertia * goose.velocity +
                1.7 * r1 * (goose.best_position - goose.position) +
                1.7 * r2 * (self.gbest - goose.position)
            )
            noise_factor = 0.1 * (1 - current_iter / self.max_iter)
            #particle.position = particle.velocity * ( noise_factor * np.random.rand(self.dim))
            #particle.position += particle.velocity * noise_factor * np.random.randn(self.dim)
            goose.position += goose.velocity * np.random.normal(loc=1, scale=noise_factor, size= self.dim) 
   
            # Batasi posisi
            goose.position = np.clip(goose.position, self.minx, self.maxx)
    """
    def update_geese(self, current_iter):
        positions = [goose.position for goose in self.geese]
        scores = [goose.score for goose in self.geese]
        self.archive.add(positions, scores)
   
        for particle in self.geese:
            # Hitung tingkat pengalaman berdasarkan umur atau fitness
            experience_level = particle.experience / self.max_iter if hasattr(particle, 'experience') else 0.5

            # === WHIFFLING EKSPLOITASI UNTUK ANGSA DEWASA ===
            if np.random.rand() < 0.2 + 0.3 * experience_level:
                self.whiffling_exploitation(particle,experience_level)
           
            r1 = np.random.uniform(0.0, 1, size=self.dim)
            r2 = np.random.uniform(0.0, 1, size=self.dim)
            #r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia = self.adaptive_inertia(current_iter)
            #inertia = 0.4 - 0.2 * (current_iter / self.max_iter)  # Inersia menurun sedikit
            cognitive = 1.5
            social = 1.6
            particle.velocity = inertia * particle.velocity \
                                + cognitive * r1 * (particle.best_position - particle.position) \
                                + social * r2 * (self.gbest - particle.position)
            az = np.random.uniform(0, 1)
            A = 0.2 * (r1 * az) -az
            C = 0.2 * r2
           # particle.velocity = (particle.velocity + A * abs(C * (particle.position - self.gbest)))
            # Tambahkan noise adaptif
            noise_factor = 0.1 * (1 - current_iter / self.max_iter)
            #particle.position = particle.velocity * ( noise_factor * np.random.rand(self.dim))
            #particle.position += particle.velocity * noise_factor * np.random.randn(self.dim)
            particle.position += particle.velocity * np.random.normal(loc=1, scale=noise_factor, size= self.dim) 
   
            # Batasi posisi
            particle.position = np.clip(particle.position, self.minx, self.maxx)
            
   
            # Update pengalaman (jika ada atribut ini)
            if hasattr(particle, 'experience'):
                particle.experience += 1
        """
            # Use Updraft
           

    def optimize(self, pareto_front):
        self.set_leader
        for i in range(self.max_iter):
            self.evaluate()
            self.set_leader()
            self.incentive()
            self.update_geese(i)
            #self.change_leader()
            if i % 40 == 0:
                self.assist_lagging_geese()
                
               
                print(f"Iteration {i} | Archive size: {len(self.archive.items)}")
            
        return self.archive.items
'''
class FGOA:
    def __init__(self, problem, dim, size, minx, maxx, max_evals, incentive_threshold, archive_size, ngrid, max_iter):
        self.problem = problem
        self.dim = dim
        self.size = size
        self.minx = np.full(self.dim, minx)
        self.maxx = np.full(self.dim, maxx)
        self.max_iter = max_iter
        self.incentive_threshold = incentive_threshold
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)
        self.gbest_score = np.inf
        #self.archive = ParetoArchive(archive_size)
        self.archive = Archive(archive_size)
        self.best_scores = []
        #self.grid = Grid(ngrid)
        self.leader = None 
        self.positions, self.velocities = self.initialize_positions_and_velocities(dim, size, minx, maxx)
        self.p_bests = self.positions.copy()
        self.p_best_fits = self.problem.evaluate(self.positions).copy()
        self.archive.add(self.positions, self.p_best_fits)
        #self.grid.calculate(self.archive.items)
        self.gd_values=[]
        self.max_evals=max_evals
      
        
    def set_leader(self):
        scalar_scores = []
        for goose in self.geese:
            scalar_scores.append(np.sum(goose.score) if isinstance(goose.score, np.ndarray) else goose.score)
        leader_index = scalar_scores.index(min(scalar_scores))
        
        # Jika self.leader belum ada, inisialisasi sebagai Goose baru
        if self.leader is None:
            self.leader = Goose(self.dim, self.minx[0], self.maxx[0])  # Buat Goose kosong
        
        # Update posisi leader dari goose terbaik
        best_goose = self.geese[leader_index]
        self.leader.position = best_goose.position.copy()
        self.leader.score = best_goose.score.copy()
        self.leader.best_position = best_goose.best_position.copy()
        self.leader.best_score = best_goose.best_score.copy()
    
        # Simpan juga ke gbest untuk reference
        self.gbest = self.leader.position.copy()
        self.gbest_score = self.leader.best_score.copy()
    
    def evaluate(self):
        for goose in self.geese:
            goose.score = self.problem.evaluate(goose.position).flatten()
            if not hasattr(goose, 'best_score') or isinstance(goose.best_score, (int, float)):
                goose.best_score = np.full_like(goose.score, np.inf)
            if dominates(goose.score, goose.best_score):
                goose.best_score = goose.score.copy()
                goose.best_position = goose.position.copy()
    
        # Update archive dengan semua posisi dan score
        positions = [goose.position for goose in self.geese]
        scores = [goose.score for goose in self.geese]
        self.archive.add(positions, scores)
        
    def initialize_positions_and_velocities(self, dim, size, minx, maxx):
        sobol = Sobol(d=dim, scramble=True)
        samples = sobol.random(n=size)  # Lebih sederhana dan akurat
        positions = minx + samples * (maxx - minx)
        velocities = np.random.uniform(low=-0.1, high=0.5, size=(size, dim))
        return positions, velocities

    def assist_lagging_geese(self):
        valid_scores = [goose.score for goose in self.geese if np.isfinite(np.sum(goose.score))]
        avg_score = np.mean(valid_scores)
        threshold = avg_score * 0.9
        for goose in self.geese:
            if np.sum(goose.score) > threshold:
                goose.position += 0.1 * (self.gbest - goose.position)
                goose.position = np.clip(goose.position, self.minx, self.maxx)


    def incentive(self):
        archive_positions, _ = self.archive.get_archive()
        idx = np.random.choice(len(archive_positions), size=self.size // 2, replace=True)
        selected = archive_positions[idx]
    
        for i in range(self.size // 2):
            self.geese[i].position = selected[i].copy()

    def adaptive_inertia(self, iteration):
        si = 0.8
        send = iteration / self.max_iter
        if 0.3 <= send <= 0.9:
            return 0.8 + 0.2 * (send - 0.4) / 0.4 
        
        inertia = send + (si - send)*(1-iteration/self.max_iter)
        return inertia
        
    
    def change_leader(self):
        if len(self.archive.items) == 0:
            # Gunakan goose terbaik dari populasi
            best_goose = min(self.geese, key=lambda x: np.sum(x.score) if isinstance(x.score, np.ndarray) else x.score)
            if self.leader is None:
                self.leader = Goose(self.dim, self.minx[0], self.maxx[0])
            self.leader.position = best_goose.position.copy()
            return
    
        # Hitung crowding distance untuk archive items
        fits = np.array([item['fit'] for item in self.archive.items])
        dist = cdist(fits, fits)
        np.fill_diagonal(dist, np.inf)
        nearest_indices = np.argmin(dist, axis=1)
    
        crowding = np.zeros(len(fits))
        for i in range(len(fits)):
            crowding[i] = np.linalg.norm(fits[i] - fits[nearest_indices[i]])
    
        # Tambahkan crowding ke item
        for i, item in enumerate(self.archive.items):
            item['crowding'] = crowding[i]
    
        # Pilih leader berdasarkan crowding tertinggi
        leader_item = max(self.archive.items, key=lambda x: x.get('crowding', 0))
    
        # Jika leader belum ada, buat dulu
        if self.leader is None:
            self.leader = Goose(self.dim, self.minx[0], self.maxx[0])
    
        self.leader.position = leader_item['pos'].copy()
        
    def whiffling_exploitation(self, particle, experience_level=0.5):
        # Arah menuju gbest
        if np.random.rand() < 0.5:
            direction_to_target = -(self.gbest - particle.position)  # Reverse
        else:
            direction_to_target = self.gbest - particle.position
    
        dir_norm = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
    
        # Random noise & variation
        random_factor = np.random.rand() * np.abs(direction_to_target)
        strong_noise = np.random.randn(self.dim) * 0.5 * (1 - experience_level)
    
        # Whiffling-like random direction change in N-dim
        angle = np.random.uniform(0, np.pi/6) * (1 - experience_level)
        random_direction = np.random.randn(self.dim)
        random_direction /= (np.linalg.norm(random_direction) + 1e-8)
    
        # Combine all components
        new_step = dir_norm * np.linalg.norm(direction_to_target) * np.cos(angle)
        new_step += random_direction * np.linalg.norm(direction_to_target) * np.sin(angle)
        new_step += random_factor + strong_noise
    
        # Update posisi
        particle.position += new_step
        particle.position = np.clip(particle.position, self.minx, self.maxx)
        
    def update_geese(self, current_iter):
        # Dynamic neighborhood radius: besar di awal, kecil di akhir
        neighborhood_radius = 1.5 * (1 - current_iter / self.max_iter)
   
        for particle in self.geese:
            # Hitung tingkat pengalaman berdasarkan umur atau fitness
            experience_level = particle.experience / self.max_iter if hasattr(particle, 'experience') else 0.5
   
            # === EKSPLORASI AKTIF UNTUK ANGSA MUDA ===
            if experience_level < 0.2:
                # Gunakan random walk atau variasi besar
                exploration_step = np.random.uniform(-0.1, 0.1, size=self.dim) * (1 - experience_level)
                particle.position += exploration_step
   
            # === WHIFFLING EKSPLOITASI UNTUK ANGSA DEWASA ===
            elif np.random.rand() < 0.2 + 0.3 * experience_level:
                self.whiffling_exploitation(particle,experience_level)
   
            # === INTERAKSI SOSIAL (Cohesion, Alignment, Separation) ===
            neighbors = [p for p in self.geese if np.linalg.norm(p.position - particle.position) < neighborhood_radius]
    
           
            r1 = np.random.uniform(0.0, 1, size=self.dim)
            r2 = np.random.uniform(0.0, 1, size=self.dim)
            #r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia = self.adaptive_inertia(current_iter)
            #inertia = 0.4 - 0.2 * (current_iter / self.max_iter)  # Inersia menurun sedikit
            cognitive = 0.6
            social = 0.6
            particle.velocity = inertia * particle.velocity \
                                + cognitive * r1 * (particle.best_position - particle.position) \
                                + social * r2 * (self.gbest - particle.position)

            noise_factor = 0.1 * (1 - current_iter / self.max_iter)
            #particle.position = particle.velocity * ( noise_factor * np.random.rand(self.dim))
            #particle.position += particle.velocity * noise_factor * np.random.randn(self.dim)
            particle.position += particle.velocity * np.random.normal(loc=1, scale=noise_factor, size=self.dim)

   
            # Batasi posisi
            particle.position = np.clip(particle.position, self.minx, self.maxx)
            
   
            # Update pengalaman (jika ada atribut ini)
            if hasattr(particle, 'experience'):
                particle.experience += 1
       
            # Use Updraft
            if np.random.rand() < 0.2:
                particle.use_updraft_front_neighbor(particle, self.gbest, self.geese)
            else:
                particle.use_updraft_best_neighbor(particle,self.geese)


    def optimize(self, pareto_front):
        self.set_leader
        for i in range(self.max_iter):
            self.evaluate()
            self.set_leader()
            self.incentive()
            self.update_geese(i)
            #self.change_leader()
            if i % 10 == 0:
                self.assist_lagging_geese()
                
               
                print(f"Iteration {i} | Archive size: {len(self.archive.items)}")
            
        return self.archive.items
