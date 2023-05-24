import pandas as pd
import numpy as np
import scipy.optimize as opt
import random
import math

# 100m ...

# Load the CO2 data for each day
data_100x100 = pd.read_csv("QGIS_upsampling_100m_day2.csv")

# Find the maximum CO2 concentration in each dataset
max_conc_100x100 = data_100x100["CO2 (ppm)"].max()

# Select the dataset with the highest concentration
data_100 = data_100x100

# Calculate the time duration in seconds
time_start1 = pd.to_datetime(data_100['TimeStamp (UTC)'].min())
time_end1 = pd.to_datetime(data_100['TimeStamp (UTC)'].max())
time_duration1 = (time_end1 - time_start1).total_seconds()

# Calculate the sum of CO2 concentration
CO2_sum1 = data_100["CO2 (ppm)"].sum()

# Calculate the emission rate in ppm per second
emission_rate1 = CO2_sum1 / time_duration1



# Split the data into 5 chunks
chunk_size_100 = int(len(data_100) / 20)
chunks_100 = [data_100.iloc[i:i+chunk_size_100] for i in range(0, len(data_100), chunk_size_100)]

# Define the Gaussian plume model
H100= 10 # stack height according to the Gaussian plume

def gaussian_plume_model100(y100, z100, y0100, z0100, sigma_y100, sigma_z100, q100, U100):
    #original equation of gaussian plume
    C100 = (q100 / (2 * np.pi * U100 * sigma_y100 * sigma_z100)) * np.exp(-0.5 * (y100 - y0100)**2 / (sigma_y100**2)) * (np.exp(-0.5 * (z100 - z0100 -H100)**2 / (sigma_z100**2)) + np.exp(-0.5 * (z100 - z0100 + H100)**2 / (sigma_z100**2)))
    return C100


# Define the cost function to be minimized
def cost_function100(params100, x100, y100, z100, co2_100, U100):
    x0100, y0100, z0100, sigma_y100, sigma_z100, q100 = params100
    plume_model100 = gaussian_plume_model100( y100, z100,y0100, z0100, sigma_y100, sigma_z100, q100, U100)
    return np.sum((co2_100 - plume_model100)**2)

#Averaging wind speed
wind_speeds100 = list(data_100['wind_speed'])
wind_speeds100 = [float(speed.split(" ")[0]) for speed in wind_speeds100]
average_wind_speed100 = sum(wind_speeds100) / len(wind_speeds100)
average_wind_speed_str100 = "{:.2f} m/s".format(average_wind_speed100)
print("The average wind speed for 100m is:", average_wind_speed_str100)

# Averaging wind direction
wind_direction100 = list(data_100['wind_direction'])
wind_direction100 = [float(speed.split("°")[0]) for speed in wind_direction100]
average_wind_direction100 = sum(wind_direction100) / len(wind_direction100)
print("The average wind direction for 100m is:", round(average_wind_direction100, 2))

# Fit the Gaussian plume model to the data
def fit_gaussian_plume_model100(data_100):
    x100 = data_100['X']
    y100 = data_100[ 'Y']
    z100 = data_100['"Altitude (Meter AGL)"']
    co2_100 = data_100["CO2 (ppm)"]
    # Set initial guess for the parameters
    x0_0100 = np.mean(x100)
    y0_0100 = np.mean(y100)
    z0_0100 = np.mean(z100)
    sigma_y_0100 = np.std(y0_0100)
    sigma_z_0100 = np.std(z0_0100)
    q_0100 = emission_rate1
    U_0100 = average_wind_speed100


    #Set the bounds for the parameters
    bounds100 = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize the cost function
    params_opt100 = opt.minimize(cost_function100, (x0_0100, y0_0100, z0_0100, sigma_y_0100, sigma_z_0100, q_0100), args=(x100, y100, z100, co2_100, average_wind_speed100), bounds=bounds100, method='L-BFGS-B')

    # Get the optimized parameters
    x0_opt100, y0_opt100, z0_opt100, sigma_y_opt100, sigma_z_opt100, q_opt100 = params_opt100.x

    # Calculate the plume model using the optimized parameters
    plume_model100 = gaussian_plume_model100(y100, z100, y0_opt100, z0_opt100, sigma_y_opt100, sigma_z_opt100, q_opt100, average_wind_speed100)

    return x0_opt100, y0_opt100, z0_opt100, sigma_y_opt100, sigma_z_opt100, q_opt100, average_wind_speed100, plume_model100


# Define the comparison function using Euclidean distance
def compare_results100(chunks_100, ref_location100):
    x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100, plume_model100 = fit_gaussian_plume_model100(chunks_100)
    return math.sqrt((x0100 - ref_location100[0])**2 + (y0100 - ref_location100[1])**2 + (z0100 - ref_location100[2])**2 + (sigma_y100 - ref_location100[3])**2 + (sigma_z100 - ref_location100[4])**2)


pop_size100 = 3000
num_generations100 = 1
mutation_rate100 = 0.025
crossover_rate100 = 0.8

def generate_initial_population100(pop_size100):
    population100 = []
    for i in range(pop_size100):
        x0100 = random.uniform(data_100['X'].min(), data_100['X'].max())
        y0100 = random.uniform(data_100[ 'Y'].min(), data_100[ 'Y'].max())
        z0100 = random.uniform(data_100['"Altitude (Meter AGL)"'].min(),data_100['"Altitude (Meter AGL)"'].max())
        sigma_y100 = random.uniform(0, np.std(y0100))
        sigma_z100 = random.uniform(0, np.std(z0100))
        q100 = random.uniform(0, 100)
        U100 = random.uniform(0, 10)
        params100 = (x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100)
        population100.append(params100)
    return population100

def calculate_fitness100(params100, x100, y100, z100, co2_100):
    x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100 = params100
    plume_model100 = gaussian_plume_model100( y100, z100, y0100, z0100, sigma_y100, sigma_z100, q100, U100)
    return -np.sum((co2_100 - plume_model100)**2)

def get_best_individual100(population100, x100, y100, z100, co2_100):
    fitness_values100 = [calculate_fitness100(ind, x100, y100, z100, co2_100) for ind in population100]
    fitness_values100 = fitness_values100[~np.isnan(fitness_values100)]
    return population100[np.argmin(fitness_values100)]

def mutate100(params100, mutation_rate100):
    x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100 = params100
    if random.random() < mutation_rate100:
        x0100 = x0100 + random.uniform(-100, 100)
        y0100 = y0100 + random.uniform(-100, 100)
        z0100 = z0100 + random.uniform(-100, 100)
    sigma_y100 = sigma_y100 + random.uniform(-0.1, 0.1)
    sigma_z100 = sigma_z100 + random.uniform(-0.1, 0.1)
    q100 = q100 + random.uniform(-100, 100)
    U100 = U100 + random.uniform(-100, 100)
    return (x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100)

def crossover100(params1_100, params2_100, crossover_rate100):
    x0_1_100, y0_1_100, z0_1_100, sigma_y_1_100, sigma_z_1_100, q_1_100, U_1_100 = params1_100
    x0_2_100, y0_2_100, z0_2_100, sigma_y_2_100, sigma_z_2_100, q_2_100, U_2_100 = params2_100

    if random.random() < crossover_rate100:
            child_params100 = (x0_1_100, y0_2_100, z0_1_100, sigma_y_2_100, sigma_z_1_100, q_1_100, U_1_100)
    else:
        child_params100 = params1_100
    return child_params100


def selection100(population100, fitness_values100):

    fitness_values100 = np.array(fitness_values100)
    fitness_values100 += 1e-6

    # fitness_values_normalized = fitness_values / sum(fitness_values)


    if np.isinf(fitness_values100).any():
        indices_100 = np.where(np.isinf(fitness_values100))[0]
        for index in indices_100:
            if index > 0:
                fitness_values100[index] = fitness_values100[index - 1]

    for i in range(len(fitness_values100)):
        if fitness_values100[i] >= 0:
            fitness_values100[i] = fitness_values100[i-1]


    if np.isinf(fitness_values100).any():
        indices_100 = np.where(np.isinf(fitness_values100))[0]
        for index in indices_100:
            fitness_values100[index] = 0

    for i in range(len(fitness_values100)):
        if fitness_values100[i] >= 0:
            fitness_values100[i] = fitness_values100[i-1]



    if np.sum(fitness_values100) == 0:
        raise ValueError("Sum of fitness values is zero.")
    fitness_values_normalized100 = fitness_values100 / np.sum(fitness_values100)

    selected_index100 = np.random.choice(np.arange(len(population100)), p=fitness_values_normalized100)


    return population100[selected_index100]


def genetic_algorithm100(population100, fitness_function100, num_generations100, mutation_rate100, crossover_rate100):
    for i in range(num_generations100):

        x100 = data_100['X']
        y100 = data_100[ 'Y']
        z100 = data_100['"Altitude (Meter AGL)"']
        co2_100 = data_100["CO2 (ppm)"]
        print("Generation:", i+1)
        fitness_values100 = [fitness_function100(params100, x100, y100, z100, co2_100) for params100 in population100]
        population100 = [selection100(population100, fitness_values100) for i in range(pop_size100)]
        population100 = [mutate100(params100, mutation_rate100) for params100 in population100]
        population100 = [crossover100(population100[i], population100[i+1], crossover_rate100) for i in range(0, pop_size100-1, 2)]

    return population100

population100 = generate_initial_population100(pop_size100)
best_params100 = genetic_algorithm100(population100, calculate_fitness100, num_generations100, mutation_rate100, crossover_rate100)

print("Best parameters 100m:", best_params100)

print('working on finding the best chunk for 100m please wait ..............')
def check_concentration_value100(ppm_value100, current_ppm_value100, deviation_threshold100):
    # Check if the current ppm value is within the deviation threshold
    if abs(ppm_value100 - current_ppm_value100) <= deviation_threshold100:
     return True
    else:
     return False

def calculate_avg_ppm100(ppm_list100):
# Calculate the average ppm value from the list
    return sum(ppm_list100)/len(ppm_list100)

def get_final_location100(coord_list100, ppm_list100, deviation_threshold100):
# Get the final location of the car by checking the deviation threshold of the average ppm value
    avg_ppm100 = calculate_avg_ppm100(ppm_list100)
    final_location100 = None
    for i in range(len(coord_list100)):
        if check_concentration_value100(avg_ppm100, ppm_list100[i], deviation_threshold100):
            final_location100 = coord_list100[i]
            break
    return final_location100

# Find the chunk with the most accurate result
ref_location100 = fit_gaussian_plume_model100(data_100)
distances100 = [compare_results100(chunk100, ref_location100) for chunk100 in chunks_100]
accurate_chunk_index100 = distances100.index(min(distances100))

# Retrieve the most accurate chunk
most_accurate_chunk_100m = chunks_100[accurate_chunk_index100]
print("Most Accurate Chunk 100m:", most_accurate_chunk_100m)

# Heat map code
x100 = data_100['X']
y100 = data_100[ 'Y']
co2_100 = data_100["CO2 (ppm)"]


# Final location parameters (assumed to be in the form of latitude and longitude)
result100 = fit_gaussian_plume_model100(data_100)
x0100, y0100, z0100, sigma_y100, sigma_z100, q100, U100, plume_model100 = result100

latitude1= x0100
longitude1 = y0100

# 200m ...

# Load the CO2 data for each day
data_200x200 = pd.read_csv("QGIS_upsampling_200m_day2.csv")

# Find the maximum CO2 concentration in each dataset
max_conc_200x200 = data_200x200["CO2 (ppm)"].max()

# Select the dataset with the highest concentration
data_200 = data_200x200

# Calculate the time duration in seconds
time_start2 = pd.to_datetime(data_200['TimeStamp (UTC)'].min())
time_end2 = pd.to_datetime(data_200['TimeStamp (UTC)'].max())
time_duration2 = (time_end2 - time_start2).total_seconds()

# Calculate the sum of CO2 concentration
CO2_sum2 = data_200["CO2 (ppm)"].sum()

# Calculate the emission rate in ppm per second
emission_rate2 = CO2_sum2 / time_duration2


# Split the data into 5 chunks
chunk_size_200 = int(len(data_200) / 20)
chunks_200 = [data_200.iloc[i:i + chunk_size_200] for i in range(0, len(data_200), chunk_size_200)]

# Define the Gaussian plume model
H200 = 20  # stack height according to the Gaussian plume


def gaussian_plume_model200(y200, z200, y0200, z0200, sigma_y200, sigma_z200, q200, U200):
    # original equation of gaussian plume
    C200 = (q200 / (2 * np.pi * U200 * sigma_y200 * sigma_z200)) * np.exp(
        -0.5 * (y200 - y0200) ** 2 / (sigma_y200 ** 2)) * (
                       np.exp(-0.5 * (z200 - z0200 - H200) ** 2 / (sigma_z200 ** 2)) + np.exp(
                   -0.5 * (z200 - z0200 + H200) ** 2 / (sigma_z200 ** 2)))
    return C200


# Define the cost function to be minimized
def cost_function200(params200, x200, y200, z200, co2_200, U200):
    x0200, y0200, z0200, sigma_y200, sigma_z200, q200 = params200
    plume_model200 = gaussian_plume_model200(y200, z200, y0200, z0200, sigma_y200, sigma_z200, q200, U200)
    return np.sum((co2_200 - plume_model200) ** 2)


# Averaging wind speed
wind_speeds200 = list(data_200['wind_speed'])
wind_speeds200 = [float(speed.split(" ")[0]) for speed in wind_speeds200]
average_wind_speed200 = sum(wind_speeds200) / len(wind_speeds200)
average_wind_speed_str200 = "{:.2f} m/s".format(average_wind_speed200)
print("The average wind speed for 200m is:", average_wind_speed_str200)

# Averaging wind direction
wind_direction200 = list(data_200['wind_direction'])
wind_direction200 = [float(speed.split("°")[0]) for speed in wind_direction200]
average_wind_direction200 = sum(wind_direction200) / len(wind_direction200)
print("The average wind direction for 200m is:", round(average_wind_direction200, 2))

# Fit the Gaussian plume model to the data
def fit_gaussian_plume_model200(data_200):
    x200 = data_200['X']
    y200 = data_200['Y']
    z200 = data_200['"Altitude (Meter AGL)"']
    co2_200 = data_200["CO2 (ppm)"]
    # Set initial guess for the parameters
    x0_0200 = np.mean(x200)
    y0_0200 = np.mean(y200)
    z0_0200 = np.mean(z200)
    sigma_y_0200 = np.std(y0_0200)
    sigma_z_0200 = np.std(z0_0200)
    q_0200 = emission_rate2
    U_0200 = average_wind_speed200

    # Set the bounds for the parameters
    bounds200 = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize the cost function
    params_opt200 = opt.minimize(cost_function200, (x0_0200, y0_0200, z0_0200, sigma_y_0200, sigma_z_0200, q_0200),
                                 args=(x200, y200, z200, co2_200, average_wind_speed200), bounds=bounds200,
                                 method='L-BFGS-B')

    # Get the optimized parameters
    x0_opt200, y0_opt200, z0_opt200, sigma_y_opt200, sigma_z_opt200, q_opt200 = params_opt200.x

    # Calculate the plume model using the optimized parameters
    plume_model200 = gaussian_plume_model200(y200, z200, y0_opt200, z0_opt200, sigma_y_opt200, sigma_z_opt200, q_opt200,
                                             average_wind_speed200)

    return x0_opt200, y0_opt200, z0_opt200, sigma_y_opt200, sigma_z_opt200, q_opt200, average_wind_speed200, plume_model200


# Define the comparison function using Euclidean distance
def compare_results200(chunks_200, ref_location200):
    x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200, plume_model200 = fit_gaussian_plume_model200(chunks_200)
    return math.sqrt(
        (x0200 - ref_location200[0]) ** 2 + (y0200 - ref_location200[1]) ** 2 + (z0200 - ref_location200[2]) ** 2 + (
                    sigma_y200 - ref_location200[3]) ** 2 + (sigma_z200 - ref_location200[4]) ** 2)


pop_size200 = 3000
num_generations200 = 1
mutation_rate200 = 0.025
crossover_rate200 = 0.8


def generate_initial_population200(pop_size200):
    population200 = []
    for i in range(pop_size200):
        x0200 = random.uniform(data_200['X'].min(), data_200['X'].max())
        y0200 = random.uniform(data_200['Y'].min(), data_200['Y'].max())
        z0200 = random.uniform(data_200['"Altitude (Meter AGL)"'].min(), data_200['"Altitude (Meter AGL)"'].max())
        sigma_y200 = random.uniform(0, np.std(y0200))
        sigma_z200 = random.uniform(0, np.std(z0200))
        q200 = random.uniform(0, 100)
        U200 = random.uniform(0, 10)
        params200 = (x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200)
        population200.append(params200)
    return population200


def calculate_fitness200(params200, x200, y200, z200, co2_200):
    x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200 = params200
    plume_model200 = gaussian_plume_model200(y200, z200, y0200, z0200, sigma_y200, sigma_z200, q200, U200)
    return -np.sum((co2_200 - plume_model200) ** 2)


def get_best_individual200(population200, x200, y200, z200, co2_200):
    fitness_values200 = [calculate_fitness200(ind, x200, y200, z200, co2_200) for ind in population200]
    fitness_values200 = fitness_values200[~np.isnan(fitness_values200)]
    return population200[np.argmin(fitness_values200)]


def mutate200(params200, mutation_rate200):
    x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200 = params200
    if random.random() < mutation_rate200:
        x0200 = x0200 + random.uniform(-100, 100)
        y0200 = y0200 + random.uniform(-100, 100)
        z0100 = z0200 + random.uniform(-100, 100)
    sigma_y200 = sigma_y200 + random.uniform(-0.1, 0.1)
    sigma_z200 = sigma_z200 + random.uniform(-0.1, 0.1)
    q200 = q200 + random.uniform(-100, 100)
    U200 = U200 + random.uniform(-100, 100)
    return (x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200)


def crossover200(params1_200, params2_200, crossover_rate200):
    x0_1_200, y0_1_200, z0_1_200, sigma_y_1_200, sigma_z_1_200, q_1_200, U_1_200 = params1_200
    x0_2_200, y0_2_200, z0_2_200, sigma_y_2_200, sigma_z_2_200, q_2_200, U_2_200 = params2_200

    if random.random() < crossover_rate200:
        child_params200 = (x0_1_200, y0_2_200, z0_1_200, sigma_y_2_200, sigma_z_1_200, q_1_200, U_1_200)
    else:
        child_params200 = params1_200
    return child_params200


def selection200(population200, fitness_values200):
    fitness_values200 = np.array(fitness_values200)
    fitness_values200 += 1e-6

    # fitness_values_normalized = fitness_values / sum(fitness_values)

    if np.isinf(fitness_values200).any():
        indices_100 = np.where(np.isinf(fitness_values200))[0]
        for index in indices_100:
            if index > 0:
                fitness_values200[index] = fitness_values200[index - 1]

    for i in range(len(fitness_values200)):
        if fitness_values200[i] >= 0:
            fitness_values200[i] = fitness_values200[i - 1]

    if np.isinf(fitness_values200).any():
        indices_200 = np.where(np.isinf(fitness_values200))[0]
        for index in indices_200:
            fitness_values200[index] = 0

    for i in range(len(fitness_values200)):
        if fitness_values200[i] >= 0:
            fitness_values200[i] = fitness_values200[i - 1]

    if np.sum(fitness_values200) == 0:
        raise ValueError("Sum of fitness values is zero.")
    fitness_values_normalized200 = fitness_values200 / np.sum(fitness_values200)

    selected_index200 = np.random.choice(np.arange(len(population200)), p=fitness_values_normalized200)

    return population200[selected_index200]


def genetic_algorithm200(population200, fitness_function200, num_generations200, mutation_rate200, crossover_rate200):
    for i in range(num_generations200):
        x200 = data_200['X']
        y200 = data_200['Y']
        z200 = data_200['"Altitude (Meter AGL)"']
        co2_200 = data_200["CO2 (ppm)"]
        print("Generation:", i + 1)
        fitness_values200 = [fitness_function200(params200, x200, y200, z200, co2_200) for params200 in population200]
        population200 = [selection200(population200, fitness_values200) for i in range(pop_size200)]
        population200 = [mutate200(params200, mutation_rate200) for params200 in population200]
        population200 = [crossover200(population200[i], population200[i + 1], crossover_rate200) for i in
                         range(0, pop_size200 - 1, 2)]

    return population200


population200 = generate_initial_population200(pop_size200)
best_params200 = genetic_algorithm200(population200, calculate_fitness200, num_generations200, mutation_rate200,
                                      crossover_rate200)

print("Best parameters 200m:", best_params200)

print('working on finding the best chunk for 200m please wait ..............')


def check_concentration_value200(ppm_value200, current_ppm_value200, deviation_threshold200):
    # Check if the current ppm value is within the deviation threshold
    if abs(ppm_value200 - current_ppm_value200) <= deviation_threshold200:
        return True
    else:
        return False


def calculate_avg_ppm200(ppm_list200):
    # Calculate the average ppm value from the list
    return sum(ppm_list200) / len(ppm_list200)


def get_final_location200(coord_list200, ppm_list200, deviation_threshold200):
    # Get the final location of the car by checking the deviation threshold of the average ppm value
    avg_ppm200 = calculate_avg_ppm200(ppm_list200)
    final_location200 = None
    for i in range(len(coord_list200)):
        if check_concentration_value200(avg_ppm200, ppm_list200[i], deviation_threshold200):
            final_location200 = coord_list200[i]
            break
    return final_location200


# Find the chunk with the most accurate result
ref_location200 = fit_gaussian_plume_model200(data_200)
distances200 = [compare_results200(chunk200, ref_location200) for chunk200 in chunks_200]
accurate_chunk_index200 = distances200.index(min(distances200))

# Retrieve the most accurate chunk
most_accurate_chunk_200m = chunks_200[accurate_chunk_index200]
print("Most Accurate Chunk 200m:", most_accurate_chunk_200m)

# Heat map code
x200 = data_200['X']
y200 = data_200['Y']
co2_200 = data_200["CO2 (ppm)"]


# Final location parameters (assumed to be in the form of latitude and longitude)
result200 = fit_gaussian_plume_model200(data_200)
x0200, y0200, z0200, sigma_y200, sigma_z200, q200, U200, plume_model200 = result200

latitude2 = x0200
longitude2 = y0200

# 300m ...

# Load the CO2 data for each day
data_300x300 = pd.read_csv("QGIS_upsampling_300m_day2.csv")

# Find the maximum CO2 concentration in each dataset
max_conc_300x300 = data_300x300["CO2 (ppm)"].max()

# Select the dataset with the highest concentration
data_300 = data_300x300

# Calculate the time duration in seconds
time_start3 = pd.to_datetime(data_300['TimeStamp (UTC)'].min())
time_end3 = pd.to_datetime(data_300['TimeStamp (UTC)'].max())
time_duration3 = (time_end3 - time_start3).total_seconds()

# Calculate the sum of CO2 concentration
CO2_sum3 = data_300["CO2 (ppm)"].sum()

# Calculate the emission rate in ppm per second
emission_rate3 = CO2_sum3 / time_duration3


# Split the data into 5 chunks
chunk_size_300 = int(len(data_300) / 21)
chunks_300 = [data_300.iloc[i:i + chunk_size_300] for i in range(0, len(data_300), chunk_size_300)]

# Define the Gaussian plume model
H300 = 30  # stack height according to the Gaussian plume


def gaussian_plume_model300(y300, z300, y0300, z0300, sigma_y300, sigma_z300, q300, U300):
    # original equation of gaussian plume
    C300 = (q300 / (2 * np.pi * U300 * sigma_y300 * sigma_z300)) * np.exp(
        -0.5 * (y300 - y0300) ** 2 / (sigma_y300 ** 2)) * (
                       np.exp(-0.5 * (z300 - z0300 - H300) ** 2 / (sigma_z300 ** 2)) + np.exp(
                   -0.5 * (z300 - z0300 + H300) ** 2 / (sigma_z300 ** 2)))
    return C300


# Define the cost function to be minimized
def cost_function300(params300, x300, y300, z300, co2_300, U300):
    x0300, y0300, z0300, sigma_y300, sigma_z300, q300 = params300
    plume_model300 = gaussian_plume_model300(y300, z300, y0300, z0300, sigma_y300, sigma_z300, q300, U300)
    return np.sum((co2_300 - plume_model300) ** 2)


# Averaging wind speed
wind_speeds300 = list(data_300['wind_speed'])
wind_speeds300 = [float(speed.split(" ")[0]) for speed in wind_speeds300]
average_wind_speed300 = sum(wind_speeds300) / len(wind_speeds300)
average_wind_speed_str300 = "{:.2f} m/s".format(average_wind_speed300)
print("The average wind speed for 300m is:", average_wind_speed_str300)

# Averaging wind direction
wind_direction300 = list(data_300['wind_direction'])
wind_direction300 = [float(speed.split("°")[0]) for speed in wind_direction300]
average_wind_direction300 = sum(wind_direction300) / len(wind_direction300)
print("The average wind direction for 300m is:", round(average_wind_direction300, 2))


# Fit the Gaussian plume model to the data
def fit_gaussian_plume_model300(data_300):
    x300 = data_300['X']
    y300 = data_300['Y']
    z300 = data_300['"Altitude (Meter AGL)"']
    co2_300 = data_300["CO2 (ppm)"]
    # Set initial guess for the parameters
    x0_0300 = np.mean(x300)
    y0_0300 = np.mean(y300)
    z0_0300 = np.mean(z300)
    sigma_y_0300 = np.std(y0_0300)
    sigma_z_0300 = np.std(z0_0300)
    q_0300 = emission_rate3
    U_0300 = average_wind_speed300

    # Set the bounds for the parameters
    bounds300 = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize the cost function
    params_opt300 = opt.minimize(cost_function300, (x0_0300, y0_0300, z0_0300, sigma_y_0300, sigma_z_0300, q_0300),
                                 args=(x300, y300, z300, co2_300, average_wind_speed300), bounds=bounds300,
                                 method='L-BFGS-B')

    # Get the optimized parameters
    x0_opt300, y0_opt300, z0_opt300, sigma_y_opt300, sigma_z_opt300, q_opt300 = params_opt300.x

    # Calculate the plume model using the optimized parameters
    plume_model300 = gaussian_plume_model300(y300, z300, y0_opt300, z0_opt300, sigma_y_opt300, sigma_z_opt300, q_opt300,
                                             average_wind_speed300)

    return x0_opt300, y0_opt300, z0_opt300, sigma_y_opt300, sigma_z_opt300, q_opt300, average_wind_speed300, plume_model300


# Define the comparison function using Euclidean distance
def compare_results300(chunks_300, ref_location300):
    x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300, plume_model300 = fit_gaussian_plume_model300(chunks_300)
    return math.sqrt(
        (x0300 - ref_location300[0]) ** 2 + (y0300 - ref_location300[1]) ** 2 + (z0300 - ref_location300[2]) ** 2 + (
                    sigma_y300 - ref_location300[3]) ** 2 + (sigma_z300 - ref_location300[4]) ** 2)


pop_size300 = 3000
num_generations300 = 2
mutation_rate300 = 0.025
crossover_rate300 = 0.8


def generate_initial_population300(pop_size300):
    population300 = []
    for i in range(pop_size300):
        x0300 = random.uniform(data_300['X'].min(), data_300['X'].max())
        y0300 = random.uniform(data_300['Y'].min(), data_300['Y'].max())
        z0300 = random.uniform(data_300['"Altitude (Meter AGL)"'].min(), data_300['"Altitude (Meter AGL)"'].max())
        sigma_y300 = random.uniform(0, np.std(y0300))
        sigma_z300 = random.uniform(0, np.std(z0300))
        q300 = random.uniform(0, 100)
        U300 = random.uniform(0, 10)
        params300 = (x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300)
        population300.append(params300)
    return population300


def calculate_fitness300(params300, x300, y300, z300, co2_300):
    x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300 = params300
    plume_model300 = gaussian_plume_model300(y300, z300, y0300, z0300, sigma_y300, sigma_z300, q300, U300)
    return -np.sum((co2_300 - plume_model300) ** 2)


def get_best_individual300(population300, x300, y300, z300, co2_300):
    fitness_values300 = [calculate_fitness300(ind, x300, y300, z300, co2_300) for ind in population300]
    fitness_values300 = fitness_values300[~np.isnan(fitness_values300)]
    return population300[np.argmin(fitness_values300)]


def mutate300(params300, mutation_rate300):
    x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300 = params300
    if random.random() < mutation_rate300:
        x0300 = x0300 + random.uniform(-100, 100)
        y0300 = y0300 + random.uniform(-100, 100)
        z0300 = z0300 + random.uniform(-100, 100)
    sigma_y300 = sigma_y300 + random.uniform(-0.1, 0.1)
    sigma_z300 = sigma_z300 + random.uniform(-0.1, 0.1)
    q300 = q300 + random.uniform(-100, 100)
    U300 = U300 + random.uniform(-100, 100)
    return (x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300)


def crossover300(params1_300, params2_300, crossover_rate300):
    x0_1_300, y0_1_300, z0_1_300, sigma_y_1_300, sigma_z_1_300, q_1_300, U_1_300 = params1_300
    x0_2_300, y0_2_300, z0_2_300, sigma_y_2_300, sigma_z_2_300, q_2_300, U_2_300 = params2_300

    if random.random() < crossover_rate300:
        child_params300 = (x0_1_300, y0_2_300, z0_1_300, sigma_y_2_300, sigma_z_1_300, q_1_300, U_1_300)
    else:
        child_params300 = params1_300
    return child_params300


def selection300(population300, fitness_values300):
    fitness_values300 = np.array(fitness_values300)
    fitness_values300 += 1e-6

    # fitness_values_normalized = fitness_values / sum(fitness_values)

    if np.isinf(fitness_values300).any():
        indices_300 = np.where(np.isinf(fitness_values300))[0]
        for index in indices_300:
            if index > 0:
                fitness_values300[index] = fitness_values300[index - 1]

    for i in range(len(fitness_values300)):
        if fitness_values300[i] >= 0:
            fitness_values300[i] = fitness_values300[i - 1]

    if np.isinf(fitness_values300).any():
        indices_300 = np.where(np.isinf(fitness_values300))[0]
        for index in indices_300:
            fitness_values300[index] = 0

    for i in range(len(fitness_values300)):
        if fitness_values300[i] >= 0:
            fitness_values300[i] = fitness_values300[i - 1]

    if np.sum(fitness_values300) == 0:
        raise ValueError("Sum of fitness values is zero.")
    fitness_values_normalized300 = fitness_values300 / np.sum(fitness_values300)

    selected_index300 = np.random.choice(np.arange(len(population300)), p=fitness_values_normalized300)

    return population300[selected_index300]


def genetic_algorithm300(population300, fitness_function300, num_generations300, mutation_rate300, crossover_rate300):
    for i in range(num_generations300):
        x300 = data_300['X']
        y300 = data_300['Y']
        z300 = data_300['"Altitude (Meter AGL)"']
        co2_300 = data_300["CO2 (ppm)"]
        print("Generation:", i + 1)
        fitness_values300 = [fitness_function300(params300, x300, y300, z300, co2_300) for params300 in population300]
        population300 = [selection300(population300, fitness_values300) for i in range(pop_size300)]
        population300 = [mutate300(params300, mutation_rate300) for params300 in population300]
        population300 = [crossover300(population300[i], population300[i + 1], crossover_rate300) for i in
                         range(0, pop_size300 - 1, 2)]

    return population300


population300 = generate_initial_population300(pop_size300)
best_params300 = genetic_algorithm300(population300, calculate_fitness300, num_generations300, mutation_rate300,
                                      crossover_rate300)

print("Best parameters 300m:", best_params300)

print('working on finding the best chunk for 300m please wait ..............')


def check_concentration_value300(ppm_value300, current_ppm_value300, deviation_threshold300):
    # Check if the current ppm value is within the deviation threshold
    if abs(ppm_value300 - current_ppm_value300) <= deviation_threshold300:
        return True
    else:
        return False


def calculate_avg_ppm300(ppm_list300):
    # Calculate the average ppm value from the list
    return sum(ppm_list300) / len(ppm_list300)


def get_final_location300(coord_list300, ppm_list300, deviation_threshold300):
    # Get the final location of the car by checking the deviation threshold of the average ppm value
    avg_ppm300 = calculate_avg_ppm300(ppm_list300)
    final_location300 = None
    for i in range(len(coord_list300)):
        if check_concentration_value300(avg_ppm300, ppm_list300[i], deviation_threshold300):
            final_location300 = coord_list300[i]
            break
    return final_location300


# Find the chunk with the most accurate result
ref_location300 = fit_gaussian_plume_model300(data_300)
distances300 = [compare_results300(chunk300, ref_location300) for chunk300 in chunks_300]
accurate_chunk_index300 = distances300.index(min(distances300))

# Retrieve the most accurate chunk
most_accurate_chunk_300m = chunks_300[accurate_chunk_index300]
print("Most Accurate Chunk 300m:", most_accurate_chunk_300m)

# Heat map code ...............................................................................................................

import matplotlib.pyplot as plt
import seaborn as sns

# Heat map code
x100 = data_100['X']
y100 = data_100['Y']
co2_100 = data_100["CO2 (ppm)"]
plt.figure(figsize=(15,10))

x300 = data_200['X']
y300 = data_200['Y']
co2_200 = data_200["CO2 (ppm)"]
plt.figure(figsize=(15,10))

x300 = data_300['X']
y300 = data_300['Y']
co2_300 = data_300["CO2 (ppm)"]
plt.figure(figsize=(15,10))


# Create a figure and axis
fig, ax = plt.subplots(figsize=(15,10))
# Plot your map or heatmap data

# Add directional labels
ax.text(0.5, 0, 'South', ha='center', va='bottom', transform=ax.transAxes,fontsize =20)
ax.text(0.5, 1, 'North', ha='center', va='top', transform=ax.transAxes, fontsize =20)
ax.text(1, 0.5, 'East', ha='right', va='center', rotation='vertical', transform=ax.transAxes, fontsize =20)
ax.text(0, 0.5, 'West', ha='left', va='center', rotation='vertical', transform=ax.transAxes, fontsize =20)

# Plot the first heatmap with transparency and filled area using RdYlGn colormap
sns.kdeplot(x=x100, y=y100, cmap='RdYlGn', fill=True, alpha=0.5, ax=ax)

# Overlay a contour plot on the first heatmap with borders
sns.kdeplot(x=x100, y=y100, cmap='RdYlGn', fill=False, levels=1, linewidths=1, ax=ax)

# Plot the second heatmap on top of the first one with transparency and filled area using RdYlGn colormap
sns.kdeplot(x=x200, y=y200, cmap='RdYlGn', fill=True, alpha=0.5, ax=ax)

# Overlay a contour plot on the second heatmap with borders
sns.kdeplot(x=x200, y=y200, cmap='RdYlGn', fill=False, levels=1, linewidths=1, ax=ax)

# Plot the third heatmap on top of the first one with transparency and filled area using RdYlGn colormap
sns.kdeplot(x=x300, y=y300, cmap='RdYlGn', fill=True, alpha=0.5, ax=ax)

# Overlay a contour plot on the third heatmap with borders
sns.kdeplot(x=x300, y=y300, cmap='RdYlGn', fill=False, levels=1, linewidths=1, ax=ax)


# Final location parameters (assumed to be in the form of latitude and longitude)
result300 = fit_gaussian_plume_model300(data_300)
x0300, y0300, z0300, sigma_y300, sigma_z300, q300, U300, plume_model300 = result300

latitude3 = x0300
longitude3 = y0300

# Lat = [53.32860045464625]
# Lon = [5.933187224731924]
Lat = [695325.0811199092]
Lon = [5912837.280386977]

# max_x = data_300[' "Latitude"'].max()
# min_x= data_300[' "Latitude"'].min()
# max_y = data_300[' "Longitude"'].max()
# min_y= data_300[' "Longitude"'].min()

# print(min_x,max_x)
# print(min_y,max_y)

plt.scatter(x0100, y0100, c="blue", s=50, marker="o", label="100x100 m")
plt.scatter(x0200, y0200, c="yellow", s=50, marker="o", label="200x200 m")
plt.scatter(x0300, y0300, c="green", s=50, marker="o", label="300x300 m")
plt.scatter(Lat, Lon, c="black", s=50, marker="x",label="CAR (Ground truth)")
plt.xlabel("Latitude (meter) ",fontsize=15)
plt.ylabel("Longitude (meter) ",fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.xlim([,])
# plt.ylim([,])
plt.title("Coordinate Locations of different testing heights & areas with the Car location (Day 2)",fontsize=20,pad =20)
plt.show()


import pyproj

# Define the UTM projection for your zone (replace with your actual zone)
utm_zone = 31
utm_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')

# Convert Cartesian coordinates to latitude and longitude
p1 = x0100
q1 = y0100
p2 = x0200
q2 = y0200
p3 = x0300
q3 = y0300

longitude1, latitude1 = utm_proj(p1, q1, inverse=True)
longitude2, latitude2 = utm_proj(p2, q2, inverse=True)
longitude3, latitude3 = utm_proj(p3, q3, inverse=True)

print("Latitude:", latitude1)
print("Longitude:", longitude1)
print("Latitude:", latitude2)
print("Longitude:", longitude2)
print("Latitude:", latitude3)
print("Longitude:", longitude3)


# Final output
print("Location in cartesian from 100m^2 (day 2):", [p1,q1])
print("Location in cartesian from 200m^2 (day 2):", [p2,q2])
print("Location in cartesian from 300m^2 (day 2):", [p3,q3])

print("Coordinate in polar from 100m^2 (day 2):", [latitude1,longitude1])
print("Coordinate in polar from 200m^2 (day 2):", [latitude2,longitude2])
print("Coordinate in polar from 300m^2 (day 2):", [latitude3,longitude3])

from geopy.distance import geodesic


Ground_truth = (53.32860045464625, 5.933187224731924)  # Car Known Location
coordinate_100 = (latitude1, longitude1)    # Location from the GPM model 100 square meters
coordinate_200 = (latitude2, longitude2)    # Location from the GPM model 200 square meters
coordinate_300 = (latitude3, longitude3)    # Location from the GPM model 300 square meters

print("Distance from the ground truth 100m^2 (day 2):", (geodesic(Ground_truth, coordinate_100).meters))  # result in distance 100 square meters)
print("Distance from the ground truth 200m^2 (day 2):", (geodesic(Ground_truth, coordinate_200).meters))  # result in distance 200 square meters)
print("Distance from the ground truth 300m^2 (day 2):", (geodesic(Ground_truth, coordinate_300).meters))  # result in distance 300 square meters)