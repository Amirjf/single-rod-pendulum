import time
import pygame
from Digital_twin import DigitalTwin
import math
import numpy as np
import random

class SimpleGA:
    def __init__(self, population_size=100, num_generations=50):
        self.population_size = population_size
        self.num_generations = num_generations
        self.digital_twin = DigitalTwin()
        
        # Possible actions: (direction, duration)
        self.possible_actions = [
            ('right', 500), ('right', 450), ('right', 400), ('right', 350), ('right', 300),
            ('left', 500), ('left', 450), ('left', 400), ('left', 350), ('left', 300)
        ]
        
        # Maximum sequence length
        self.max_sequence_length = 5
        
        # Target angle (π radians = inverted position)
        self.target_angle = np.pi
    
    def create_individual(self):
        """Create a random sequence of actions"""
        sequence_length = random.randint(2, self.max_sequence_length)
        return [random.choice(self.possible_actions) for _ in range(sequence_length)]
    
    def create_population(self):
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_sequence(self, sequence):
        """Evaluate a sequence and return fitness score"""
        # Reset pendulum state
        self.digital_twin.theta = 0  # Start at 0 radians (pointing down)
        self.digital_twin.theta_dot = 0
        self.digital_twin.x_pivot = 0
        self.digital_twin.steps = 0
        
        # Execute sequence
        for direction, duration in sequence:
            self.digital_twin.perform_action(direction, duration)
        
        # Run simulation
        max_angle = 0
        time_near_top = 0
        simulation_time = 4.0  # Increased simulation time
        steps = int(simulation_time / self.digital_twin.delta_t)
        
        for _ in range(steps):
            theta, theta_dot, _, _ = self.digital_twin.step()
            # Track the maximum angle reached (in radians)
            max_angle = max(max_angle, abs(theta))
            
            # Count time spent near inverted position (within π/6 radians of π)
            if abs(abs(theta) - self.target_angle) < np.pi/6:
                time_near_top += self.digital_twin.delta_t
        
        # Calculate fitness score
        # Angle score: heavily reward getting close to π radians
        angle_diff = abs(abs(max_angle) - self.target_angle)
        angle_score = np.exp(-10 * angle_diff)  # Steeper exponential decay
        
        # Time score: reward for time spent near inverted position
        time_score = time_near_top / simulation_time
        
        # Velocity score: reward for high angular velocity (helps with swinging up)
        velocity_score = min(1.0, abs(self.digital_twin.theta_dot) / 15.0)  # Increased velocity threshold
        
        # Combine scores with weights - heavily prioritize angle
        fitness = 0.7 * angle_score + 0.2 * time_score + 0.1 * velocity_score
        
        # Penalize longer sequences less
        length_penalty = len(sequence) / self.max_sequence_length
        fitness *= (1 - 0.2 * length_penalty)  # Reduced penalty
        
        # Bonus for actually reaching π radians
        if abs(abs(max_angle) - self.target_angle) < 0.1:  # Within 0.1 radians of π
            fitness *= 2.0
        
        return fitness
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if not parent1 or not parent2:
            return self.create_individual()
        
        # Choose crossover point
        min_len = min(len(parent1), len(parent2))
        if min_len <= 1:
            return self.create_individual()
        
        crossover_point = random.randint(1, min_len - 1)
        
        # Create child by combining parts of parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        # Ensure child doesn't exceed max length
        if len(child) > self.max_sequence_length:
            child = child[:self.max_sequence_length]
        
        return child
    
    def mutate(self, individual, mutation_rate=0.5):  # Increased mutation rate
        """Mutate an individual with given probability"""
        if random.random() < mutation_rate:
            # Choose mutation type
            mutation_type = random.choice(['add', 'remove', 'change', 'swap'])
            
            if mutation_type == 'add' and len(individual) < self.max_sequence_length:
                # Add a new action
                individual.append(random.choice(self.possible_actions))
            elif mutation_type == 'remove' and len(individual) > 2:  # Keep at least 2 actions
                # Remove a random action
                individual.pop(random.randint(0, len(individual) - 1))
            elif mutation_type == 'change':
                # Change a random action
                idx = random.randint(0, len(individual) - 1)
                individual[idx] = random.choice(self.possible_actions)
            elif mutation_type == 'swap' and len(individual) > 1:
                # Swap two random actions
                idx1, idx2 = random.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual
    
    def select_parent(self, population, fitnesses):
        """Select a parent using tournament selection"""
        tournament_size = 7  # Increased tournament size
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx]
    
    def evolve(self):
        """Run the genetic algorithm"""
        # Create initial population
        population = self.create_population()
        best_sequence = None
        best_fitness = 0
        generations_without_improvement = 0
        
        for generation in range(self.num_generations):
            # Evaluate all individuals
            fitnesses = [self.evaluate_sequence(ind) for ind in population]
            
            # Update best solution
            max_idx = fitnesses.index(max(fitnesses))
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_sequence = population[max_idx]
                generations_without_improvement = 0
                print(f"\nGeneration {generation + 1}: Found better sequence!")
                print(f"Sequence: {best_sequence}")
                print(f"Fitness: {best_fitness:.4f}")
            else:
                generations_without_improvement += 1
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best individual
            new_population.append(population[max_idx])
            
            # Fill the rest of the new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                
                # Create child through crossover
                child = self.crossover(parent1, parent2)
                
                # Mutate child
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Print generation summary
            avg_fitness = sum(fitnesses) / len(fitnesses)
            print(f"Generation {generation + 1}: Avg Fitness = {avg_fitness:.4f}, Best = {max(fitnesses):.4f}")
            
            # Early stopping if no improvement for 10 generations
            if generations_without_improvement >= 10:
                print("No improvement for 10 generations, stopping early.")
                break
        
        return best_sequence, best_fitness
    
    def test_sequence(self, sequence):
        """Test a sequence and print detailed results"""
        print("\nTesting best sequence:")
        print(f"Sequence: {sequence}")
        
        # Reset pendulum state
        self.digital_twin.theta = 0  # Start at 0 radians (pointing down)
        self.digital_twin.theta_dot = 0
        self.digital_twin.x_pivot = 0
        self.digital_twin.steps = 0
        
        # Execute sequence
        for direction, duration in sequence:
            self.digital_twin.perform_action(direction, duration)
            print(f"Performed {direction} push for {duration}ms")
        
        # Run simulation
        max_angle = 0
        time_near_top = 0
        simulation_time = 4.0
        steps = int(simulation_time / self.digital_twin.delta_t)
        
        for i in range(steps):
            theta, theta_dot, _, _ = self.digital_twin.step()
            max_angle = max(max_angle, abs(theta))
            
            # Count time spent near inverted position
            if abs(abs(theta) - self.target_angle) < np.pi/6:
                time_near_top += self.digital_twin.delta_t
            
            # Print state every 0.5 seconds
            if i % 20 == 0:  # 20 steps = 0.5 seconds
                print(f"Time: {i * self.digital_twin.delta_t:.2f}s, Angle: {theta:.3f} rad ({np.degrees(theta):.1f}°), Velocity: {theta_dot:.3f} rad/s")
        
        print(f"\nResults:")
        print(f"Maximum angle: {max_angle:.3f} rad ({np.degrees(max_angle):.1f}°)")
        print(f"Time near inverted position: {time_near_top:.2f}s")
        print(f"Distance from target (π): {abs(abs(max_angle) - self.target_angle):.3f} rad")
        
        return max_angle, time_near_top

def main():
    # Create and run the genetic algorithm
    ga = SimpleGA(population_size=100, num_generations=50)
    best_sequence, best_fitness = ga.evolve()
    
    # Test the best sequence
    ga.test_sequence(best_sequence)
    
    print("\nBest sequence found:")
    print(best_sequence)
    print(f"Fitness score: {best_fitness:.4f}")

if __name__ == "__main__":
    main() 