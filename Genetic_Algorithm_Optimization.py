import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Markdown: Step 1 - Load Data from CSV File
# Load coolant data from a CSV file and store each row as a separate coolant with specified features.
data = pd.read_csv('coolants.csv', header=None)
data.columns = ['Flow Rate', 'Initial Temperature', 'Initial Pressure', 'Transfer Rate', 'Special Heat']
coolants = {f"Coolent_R{i+1}": data.iloc[i] for i in range(10)}

# Markdown: Step 2 - Define Evaluation Function
# Define a function to calculate performance and exergy for the optimization process.
def evaluate(individual):
    flow_rate, init_temp, init_pressure, transfer_rate, special_heat = individual
    # Simple hypothetical formulae for demonstration purposes:
    performance = flow_rate * transfer_rate / (init_temp + 1)
    exergy = special_heat / (init_pressure + 1)
    return performance, exergy

# Markdown: Step 3 - Setup Genetic Algorithm
# Configure the DEAP framework for multi-objective optimization.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Define individuals and population
toolbox.register("attr_float", np.random.uniform, 0.1, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# Markdown: Step 4 - Run Optimization
# Run the genetic algorithm to optimize coolant parameters.
population = toolbox.population(n=100)
num_generations = 50

# Statistics tracking
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

# Perform optimization
for gen in range(num_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    fitnesses = map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(offspring, len(population))
    record = stats.compile(population)
    logbook.record(gen=gen, evals=len(offspring), **record)

# Markdown: Step 5 - Plot Results
# Extract and plot results including the Pareto frontier.
front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# Plot Pareto Frontier
plt.figure()
pareto_performance = [ind.fitness.values[0] for ind in front]
pareto_exergy = [ind.fitness.values[1] for ind in front]
plt.scatter(pareto_performance, pareto_exergy, color='red', label='Pareto Frontier')
plt.xlabel('Performance')
plt.ylabel('Exergy')
plt.title('Pareto Frontier')
plt.legend()
plt.grid()
plt.show()

# Markdown: Save Logbook and Plot Metrics
# Save the logbook for performance analysis and plot generation trends.
averages = [gen["avg"] for gen in logbook]
avg_performance, avg_exergy = zip(*averages)

plt.figure()
plt.plot(range(num_generations), avg_performance, label='Average Performance')
plt.plot(range(num_generations), avg_exergy, label='Average Exergy')
plt.xlabel('Generation')
plt.ylabel('Value')
plt.title('Average Performance and Exergy over Generations')
plt.legend()
plt.grid()
plt.show()

# Markdown: Save Outputs
# Save the optimized parameters and Pareto front data.
optimized_parameters = pd.DataFrame([list(ind) for ind in front], columns=data.columns)
optimized_parameters.to_csv('optimized_parameters.csv', index=False)
