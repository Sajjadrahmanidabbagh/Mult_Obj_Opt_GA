# Mult_Obj_Opt_GA

This Python script implements a multi-objective optimization using a genetic algorithm (GA). This code was used in a research paper entitled "Multi-objective design optimization of a multi-generation energy system based on geothermal and solar energy."

It starts by loading coolant data from a CSV file, where each row corresponds to a coolant characterized by five features: flow rate, initial temperature, initial pressure, transfer rate, and special heat (the original dataset size for the research paper was larger. To have a readable code, some of the input data is neglected in this code). Using this data, the script evaluates two objectives — performance and exergy — for each coolant configuration. The genetic algorithm evolves a population of candidate solutions to find Pareto-optimal solutions that balance these objectives.

The outputs include the optimized parameters for the coolants, visualizations of the Pareto frontier (highlighting the trade-off between performance and exergy), and trends of average fitness values across generations. This approach has applications in engineering design (e.g., optimizing cooling systems in aerospace and automotive industries), sustainable energy systems, and any field requiring multi-criteria decision-making.
