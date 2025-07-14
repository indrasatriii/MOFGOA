import numpy as np
from random import choice


class Goose:
    def __init__(self, dim, minx, maxx):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=2, size=dim)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf
        self.score = np.inf
        
    def use_updraft_front_neighbor(self, particle, gbest,geese):
        formation_direction = gbest - particle.best_position
        formation_direction /= np.linalg.norm(formation_direction) + 1e-8
    
        best_upwash = None
        max_dot = -np.inf
    
        for neighbor in geese:
            if neighbor is particle:
                continue
    
            relative_dir = neighbor.position - particle.position
            relative_dir /= np.linalg.norm(relative_dir) + 1e-8
    
            # Cari tetangga yang berada "di depan" berdasarkan sudut
            dot_product = np.dot(relative_dir, formation_direction)
    
            if dot_product > max_dot and dot_product > 0.5:
                max_dot = dot_product
                best_upwash = neighbor
    
        if best_upwash is not None:
            direction = best_upwash.position - particle.best_position
            distance = np.linalg.norm(direction)
            
            if distance > 0 and distance < 1:
                normalized_dir = direction / distance
                influence_strength = 0.5 / (distance + 1e-8)
                particle.velocity += influence_strength * normalized_dir

    def use_updraft_best_neighbor(self, particle,geese):
        
        direction = particle.best_position - particle.position
        distance = np.linalg.norm(direction)
        
        if distance > 0 and distance < 1.5:
            normalized_dir = direction / distance
            influence_strength = 0.5 / (distance + 1e-8)
            particle.velocity += influence_strength * normalized_dir
    
    '''        
    def use_updraft(self, updraft_center_list, strength_list, inertia, cognitive, social, gbest):
        for updraft, strength in zip(updraft_center_list, strength_list):
            distance = np.linalg.norm(self.position - updraft)
            r1 = np.random.uniform(0.0,0.3, size=self.dim)
            r2 = np.random.uniform(0.0, 0.3, size=self.dim)
            az = np.random.uniform(0, 2)
            A = 0.2 * (r1 * az) - az
            C = 0.2 * r2
            
            # Jika dalam jangkauan, tambahkan pengaruh updraft
            if distance < 1.5:
                influence = strength * np.exp(-distance**2 / (2 * 1.0**2))
                direction = (updraft - self.position) / (distance + 1e-10)
                self.velocity += influence * direction

            else:
                
                self.velocity = self.velocity + A * abs(C * (self.best_position - self.position))
                  
                #print(cognitive, social)
                index = np.random.randint(0, len(cognitive))  # Generate a random integer index
                cognitive_value = cognitive[index]
                social = 4-social

                #print(cognitive, social)
                # 若不在氣流範圍內，按標準的 PSO 更新規則更新速度
                #r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity = inertia * self.velocity + cognitive_value * r1 * (self.best_position - self.position) + social * r2 * (gbest - self.position)
               #self.velocity += np.clip(velocity_change, -max_velocity_change, max_velocity_change)
    def use_updraft(self, updraft_center_list, strength_list, inertia, cognitive, social, gbest,gbest2,gbest3,z,k):
        for updraft, strength in zip(updraft_center_list, strength_list):
            distance = np.linalg.norm(self.position - updraft)
            r1 = np.random.uniform(0.0,2, size=self.dim)
            r2 = np.random.uniform(0.0, 2, size=self.dim)
            az = np.random.uniform(0, 1)
            A = 0.2 * (gbest2 * az) -z
            C = 0.2 * gbest3
            
            # Jika dalam jangkauan, tambahkan pengaruh updraft
            if distance < 1.5:
                influence = strength * np.exp(-distance**2 / (2 * 1.0**2))
                direction = (updraft - self.position) / (distance + 1e-10)
                self.velocity += influence * direction

            else:
                if az <1:
                    self.velocity = (self.velocity + A * abs(C * (self.best_position - self.position)))
                else:
                #print(cognitive, social)
                    index = np.random.randint(0, len(cognitive))  # Generate a random integer index
                    cognitive_value = cognitive[index]
                    social = 4-social
    
                    #print(cognitive, social)
                    # 若不在氣流範圍內，按標準的 PSO 更新規則更新速度
                    #r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    self.velocity =  (
    inertia * self.velocity +
    cognitive_value * r1 * (self.best_position - self.position) +
    social * r2 * (gbest - self.position)
)
                    #self.velocity = inertia * self.velocity + cognitive_value * r1 * (self.best_position - self.position) + social * r2 * (gbest - self.position)
                    #self.velocity += np.clip(velocity_change, -max_velocity_change, max_velocity_change)
'''            