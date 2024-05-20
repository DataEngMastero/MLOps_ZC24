# MLOps_ZC24

### Setting Up Environment in Local

1. Create Conda environment - 
conda create --prefix ./env pandas numpy scikit-learn jupyter seaborn matplotlib
conda env list
conda activate /prefix-path/MLOps_ZC24/env

2. Downloading Files
curl -o ./data/yellow_tripdata_2023-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
/Users/poojasingh/Documents/Git_Reposit/MLOps_ZC24/data/green_tripdata_2021-01.parquet


### Trip Duration Prediction Model
Calculating Duration column using the formula = dropoff_time - pickup_time
Converting Duration into minutes
Filtering the majority of data: Trips with duration 1-60 mins

Feature Selection -
 1. Categorical - PULocationID, DOLocationID
 2. Numerical - trip_distance

Converting categorical features as string to use dictionary vectorisation
Training Model using Linear Regression, Ridge, and Lasso
Finding Mean Squared Error


