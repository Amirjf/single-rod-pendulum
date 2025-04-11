import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Digital_twin import DigitalTwin
import time
from scipy.fft import fft
from scipy import signal
import multiprocessing
from functools import partial
from scipy import stats
import random
from deap import base, creator, tools, algorithms
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
import os
from utility import (
    load_real_data, parallel_cost_function, add_parameter_box, analyze_parameter_sensitivity,
    plot_comprehensive_analysis, simulate_and_plot, generate_optimization_report
)

# Define the CSV filename for data loading
csv_filename = "half_theta_2"

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3
SEED = 42

# Parameter bounds
BOUNDS = [
    (0.06, 0.9),     # I_scale
    (0.001, 0.04),  # damping_coefficient
    (0.5, 1.5)      # mass
]


# Create fitness and individual classes for DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Global variables for evaluation function
global_time_array = None
global_theta_real = None
global_theta_dot_real = None
global_theta0 = None
global_theta_dot0 = None

def evaluate(individual):
    """Evaluation function for GA"""
    cost = parallel_cost_function(individual, global_time_array, global_theta_real, 
                                 global_theta_dot_real, global_theta0, global_theta_dot0)
    return (cost,)  # Return as a tuple for DEAP compatibility

def create_individual():
    """Create a random individual within bounds"""
    return [
        random.uniform(BOUNDS[0][0], BOUNDS[0][1]),  # I_scale
        random.uniform(BOUNDS[1][0], BOUNDS[1][1]),  # damping
        random.uniform(BOUNDS[2][0], BOUNDS[2][1])   # mass
    ]

def custom_mutation(individual, indpb=0.2):
    """Custom mutation that respects bounds"""
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:  # I_scale
                individual[i] = random.uniform(BOUNDS[0][0], BOUNDS[0][1])
            elif i == 1:  # damping
                individual[i] = random.uniform(BOUNDS[1][0], BOUNDS[1][1])
            else:  # mass
                individual[i] = random.uniform(BOUNDS[2][0], BOUNDS[2][1])
    return individual,

def optimize_pendulum_params():
    """Run the optimization process using Genetic Algorithm"""
    # Load the real data
    global global_time_array, global_theta_real, global_theta_dot_real, global_theta0, global_theta_dot0
    global_time_array, global_theta_real, global_theta_dot_real, global_theta0, global_theta_dot0 = load_real_data()
    
    # Initialize DEAP toolbox
    toolbox = base.Toolbox()
    
    # Register genetic operators
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Create initial population
    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(POPULATION_SIZE)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("\nStarting Genetic Algorithm optimization...")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Number of generations: {NUM_GENERATIONS}")
    print(f"Crossover probability: {CROSSOVER_PROB}")
    print(f"Mutation probability: {MUTATION_PROB}")
    
    # Run the GA
    population, logbook = algorithms.eaSimple(population, toolbox,
                                            cxpb=CROSSOVER_PROB,
                                            mutpb=MUTATION_PROB,
                                            ngen=NUM_GENERATIONS,
                                            stats=stats,
                                            halloffame=hof,
                                            verbose=True)
    
    # Get best solution
    best_solution = hof[0]
    best_fitness = best_solution.fitness.values[0]
    
    # Create a result object for consistency with other optimization methods
    class OptimizeResult:
        pass
    
    result = OptimizeResult()
    result.x = np.array(best_solution)
    result.fun = best_fitness
    result.success = True
    result.nfev = POPULATION_SIZE * NUM_GENERATIONS
    result.nit = NUM_GENERATIONS
    result.message = "Genetic Algorithm optimization terminated successfully."
    
    # Print GA-specific results
    print("\nGenetic Algorithm Results:")
    print("-" * 50)
    print(f"Best individual: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Number of evaluations: {POPULATION_SIZE * NUM_GENERATIONS}")
    
    # Plot evolution statistics
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness')
    plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/GA_evolution.png')
    plt.close()
    
    return result

# Main execution
if __name__ == "__main__":
    # Load real data
    time_array, theta_real, theta_dot_real, theta0, theta_dot0 = load_real_data()
    
    # Run optimization
    result = optimize_pendulum_params()
    
    # Print optimization results
    print("\nGenetic Algorithm Results:")
    print("-" * 50)
    print("Best parameters found:")
    print(f"I_scale: {result.x[0]:.6f}")
    print(f"Damping: {result.x[1]:.6f}")
    print(f"Mass: {result.x[2]:.6f}")
    print(f"Best cost: {result.fun:.6f}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    # Simulate and plot with best parameters
    theta_sim, error, params = simulate_and_plot(result.x, time_array, theta_real, theta_dot_real, theta0, theta_dot0, "half_theta_2", model_name='GA')
    
    # Generate optimization report
    sensitivities = analyze_parameter_sensitivity(result, time_array, theta_real, theta_dot_real, theta0, theta_dot0)
    generate_optimization_report(
        best_params=result.x,
        time_array=time_array,
        theta_real=theta_real,
        theta_dot_real=theta_dot_real,
        theta0=theta0,
        theta_dot0=theta_dot0,
        title="half_theta_2",
        model_name="GA",
        cost_value=result.fun,
        n_evaluations=result.nfev,
        optimization_time=None,  # GA doesn't track time directly
        sensitivities=sensitivities
    )
    