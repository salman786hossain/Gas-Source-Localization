# Gas-Source-Localization

The folder "GPM_GA Localization Code", which contains python code, implements a Gaussian plume model to estimate the dispersion of CO2 concentrations in the atmosphere based on wind speed and XY direction data. The code reads data from a CSV file and calculates various statistics, such as maximum concentration, time duration, emission rate, and wind parameters, from the data. It defines the Gaussian plume model function and then uses optimization techniques to fit the model to the data and estimate the parameters of the plume model. The code also includes functions for generating an initial population for a genetic algorithm and a comparison function based on Euclidean distance for evaluating the performance of different solutions in the genetic algorithm. The genetic algorithm is used to optimize the parameters of the Gaussian plume model. The code provides a framework for analyzing and modeling CO2 concentration data to understand the dispersion of CO2 in the atmosphere at a specific location.
 
# Repository Contents 

###Additional Downsampling Data/: The data downsampling could also pose a significant limitation for this project. Downsampling involves reducing the number of data points in the datasets, resulting in the least amount of data available for evaluating localization. Therefore, the combined data was down-sampled, and the minimum amount of data required for source localization was determined.


