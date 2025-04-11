import time
import pygame
from Digital_twin import DigitalTwin
import math
import numpy as np
import random

class SimpleGA:
    def __init__(self, population_size=200, num_generations=50):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = 0.3
        self.elite_size = 5
        self.tournament_size = 5
        self.max_sequence_length = 8
        self.digital_twin = DigitalTwin()
        
        # Possible actions: (direction, duration)
        self.possible_actions = [
            ('right', 200), ('right', 150), ('right', 100), ('right', 50),
            ('left', 200), ('left', 150), ('left', 100), ('left', 50)
        ]
        
        # Target angle (π radians = inverted position)
        self.target_angle = np.pi
    
    def create_initial_population(self):
        population = []
        # Create some predefined patterns that might work well
        basic_patterns = [
            [('left', 200), ('left', 200), ('right', 200)],
            [('left', 250), ('left', 150), ('right', 250)],
            [('left', 300), ('left', 100), ('right', 300)],
            [('left', 200), ('left', 150), ('left', 100), ('right', 250)]
        ]
        
        # Add basic patterns to population
        population.extend(basic_patterns)
        
        # Fill rest with random sequences
        while len(population) < self.population_size:
            sequence_length = random.randint(3, self.max_sequence_length)
            sequence = []
            total_time = 0
            for _ in range(sequence_length):
                direction = random.choice(['left', 'right'])
                # Bias towards longer pushes at start, shorter at end
                if len(sequence) < 2:
                    duration = random.randint(150, 300)
                else:
                    duration = random.randint(50, 200)
                sequence.append((direction, duration))
                total_time += duration + 50  # Add 50ms gap
                if total_time > 1000:  # Limit total sequence time
                    break
            population.append(sequence)
        return population

    def tournament_selection(self, population, fitness_scores):
        tournament = random.sample(list(enumerate(population)), self.tournament_size)
        return max(tournament, key=lambda x: fitness_scores[x[0]])[1]

    def crossover(self, parent1, parent2):
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1[:]
        
        # Two-point crossover
        point1 = random.randint(1, len(parent1)-1)
        point2 = random.randint(1, len(parent2)-1)
        child = parent1[:point1] + parent2[point2:]
        
        # Ensure sequence isn't too long
        if len(child) > self.max_sequence_length:
            child = child[:self.max_sequence_length]
        return child

    def mutate(self, sequence):
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['change_duration', 'change_direction', 'add_action', 'remove_action'])
            
            if mutation_type == 'change_duration':
                if sequence:
                    idx = random.randint(0, len(sequence)-1)
                    direction, _ = sequence[idx]
                    duration = random.randint(50, 300)
                    sequence[idx] = (direction, duration)
            
            elif mutation_type == 'change_direction':
                if sequence:
                    idx = random.randint(0, len(sequence)-1)
                    _, duration = sequence[idx]
                    direction = 'right' if sequence[idx][0] == 'left' else 'left'
                    sequence[idx] = (direction, duration)
            
            elif mutation_type == 'add_action' and len(sequence) < self.max_sequence_length:
                direction = random.choice(['left', 'right'])
                duration = random.randint(50, 300)
                insert_pos = random.randint(0, len(sequence))
                sequence.insert(insert_pos, (direction, duration))
            
            elif mutation_type == 'remove_action' and len(sequence) > 2:
                idx = random.randint(0, len(sequence)-1)
                sequence.pop(idx)
        
        return sequence

    def evolve(self):
        population = self.create_initial_population()
        best_fitness = float('-inf')
        best_sequence = None
        generations_without_improvement = 0
        
        for generation in range(self.num_generations):
            fitness_scores = [self.evaluate_sequence(seq) for seq in population]
            
            # Check for new best solution
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_sequence = population[fitness_scores.index(max_fitness)]
                generations_without_improvement = 0
                print(f"\nGeneration {generation + 1}: Found better sequence!")
                print(f"Sequence: {best_sequence}")
                print(f"Fitness: {best_fitness:.4f}")
            else:
                generations_without_improvement += 1
            
            print(f"Generation {generation + 1}: Avg Fitness = {sum(fitness_scores)/len(fitness_scores):.4f}, Best = {max_fitness:.4f}")
            
            # Early stopping
            if generations_without_improvement >= 10:
                print("No improvement for 10 generations, stopping early.")
                break
            
            # Elitism - keep best solutions
            elite = sorted(zip(fitness_scores, population), reverse=True)[:self.elite_size]
            new_population = [seq for _, seq in elite]
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_sequence, best_fitness

    def evaluate_sequence(self, sequence):
        dt = DigitalTwin()
        dt.theta = 0  # Reset angle
        dt.theta_dot = 0  # Reset angular velocity
        dt.x_pivot = 0  # Reset pivot position
        
        # Track history
        theta_history = []
        theta_dot_history = []
        
        # Execute sequence with 50ms gaps
        for direction, duration in sequence:
            dt.perform_action(direction, duration)
            theta_history.append(dt.theta)
            theta_dot_history.append(dt.theta_dot)
            # Add 50ms gap between moves
            for _ in range(5):  # 5 steps of 10ms each = 50ms
                dt.step()
                theta_history.append(dt.theta)
                theta_dot_history.append(dt.theta_dot)
        
        # Let the pendulum swing for a bit
        for _ in range(100):
            dt.step()
            theta_history.append(dt.theta)
            theta_dot_history.append(dt.theta_dot)
        
        # Calculate fitness based on multiple factors
        max_angle = max(abs(angle) for angle in theta_history)
        target_angle = math.pi
        
        # Calculate time spent near inverted position (within 0.05 radians)
        time_near_target = sum(1 for angle in theta_history[-50:] 
                              if abs(abs(angle) - target_angle) < 0.05)
        
        # Calculate energy (combination of potential and kinetic)
        final_velocity = theta_dot_history[-1]
        energy_score = 1.0 / (1.0 + abs(final_velocity))  # Reward low energy
        
        # Base fitness on how close we get to target
        angle_diff = abs(max_angle - target_angle)
        angle_score = 1.0 / (1.0 + angle_diff)
        
        # Combine scores with weights
        fitness = (angle_score * 0.4 +  # Primary goal: reach target angle
                  (time_near_target / 50.0) * 0.4 +  # Secondary: stay there
                  energy_score * 0.2)  # Tertiary: maintain low energy
        
        return fitness
    
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
    ga = SimpleGA(population_size=200, num_generations=50)
    best_sequence, best_fitness = ga.evolve()
    
    # Convert sequence to timed format with 50ms gaps
    current_time = 0.0
    timed_sequence = []
    for i, (direction, duration) in enumerate(best_sequence):
        timed_sequence.append((current_time, (direction, duration)))
        current_time += (duration / 1000.0) + 0.050  # Add 50ms gap
    
    print("\nBest sequence in timed format:")
    print("sequence = [")
    for i, (time, (direction, duration)) in enumerate(timed_sequence):
        comment = ""
        if i == 0:
            comment = "# Initial strong push"
        elif i == len(timed_sequence) - 1:
            comment = "# Final push"
        else:
            ordinal = ["First", "Second", "Third", "Fourth", "Fifth"][min(i-1, 4)]
            comment = f"# {ordinal} follow-up push"
        
        print(f"    ({time:.3f}, ('{direction}', {duration})),      {comment}")
    print("]")
    
    # Save the sequence to a file
    with open("sequence_output.py", "w") as f:
        f.write("sequence = [\n")
        for i, (time, (direction, duration)) in enumerate(timed_sequence):
            comment = ""
            if i == 0:
                comment = "# Initial strong push"
            elif i == len(timed_sequence) - 1:
                comment = "# Final push"
            else:
                ordinal = ["First", "Second", "Third", "Fourth", "Fifth"][min(i-1, 4)]
                comment = f"# {ordinal} follow-up push"
            
            f.write(f"    ({time:.3f}, ('{direction}', {duration})),      {comment}\n")
        f.write("]\n")
    
    # Test the best sequence
    ga.test_sequence(best_sequence)
    
    print("\nBest sequence found:")
    print(best_sequence)
    print(f"Fitness score: {best_fitness:.4f}")

if __name__ == "__main__":
    main() 