# Gas-Source-Localization

The folder "GPM_GA Localization Code", which contains python code, implements a Gaussian plume model to estimate the dispersion of CO2 concentrations in the atmosphere based on wind speed and XY direction data. The code reads data from a CSV file and calculates various statistics, such as maximum concentration, time duration, emission rate, and wind parameters, from the data. It defines the Gaussian plume model function and then uses optimization techniques to fit the model to the data and estimate the parameters of the plume model. The code also includes functions for generating an initial population for a genetic algorithm and a comparison function based on Euclidean distance for evaluating the performance of different solutions in the genetic algorithm. The genetic algorithm is used to optimize the parameters of the Gaussian plume model. The code provides a framework for analyzing and modeling CO2 concentration data to understand the dispersion of CO2 in the atmosphere at a specific location.
 
## Repository Contents 

- **Additional Downsampling Data/:** The data downsampling could also pose a significant limitation for this project. Downsampling involves reducing the number of data points in the datasets, resulting in the least amount of data available for evaluating localization. Therefore, the combined data was down-sampled, and the minimum amount of data required for source localization was determined.

- **Chunk 100 Datapoints/:** The folder contains 100 points as limited data assumed for localization. The testing chunk data is fetched from the datasets and there are two types of chunk criteria has been determind. One is just default 100 random data points from the datasets. Another one fetched from the selected model that assumes mosly accurate 100 data points with the highest concentration value of CO2. 

- **KML Files/:** The 'Keyhole Markup Language' folder contains the geo location from the test field.

- **QGIS UpSampling Data/:** This folder contains data upsampled from two different size datasets. The original data from the sensor and the weather log were merged and converted to geo-coordinate locations to XY cartesian in 2 dimension plane. The Upsampling code included inside the folder.

- **Raw Data/:** This folder contains the whole original data directly fetched from the drone sensor. There are two folders inside in different size of dataframe. The sensor data is the large dataset which contains CO2 gas concentration data and the wind data folder contains wind data of the weather information from the drone weather website.
