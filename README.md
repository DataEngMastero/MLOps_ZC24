# MLOps_ZC24

### Setting Up Environment in Local

1. Create Conda environment - <br>
conda create --prefix ./env pandas numpy scikit-learn jupyter seaborn matplotlib <br>
conda env list <br>
conda activate /prefix-path/MLOps_ZC24/env <br>

2. Downloading Files <br>
curl -o ./data/yellow_tripdata_2023-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet <br>


### Trip Duration Prediction Model
Calculating Duration column using the formula = dropoff_time - pickup_time <br>
Converting Duration into minutes <br>
Filtering the majority of data: Trips with duration 1-60 mins <br>

Feature Selection - <br>
 1. Categorical - PULocationID, DOLocationID 
 2. Numerical - trip_distance

Converting categoical features as string to use dictionary vectorisation <br>
Training Model using Linear Regression, Ridge, and Lasso <br>
Finding Mean Squared Error <br>


### Experiment Tracking with MLflow

Experiment Tracking is the process of keeping records of relevent information when doing ML experiment runs. <br>

MLflow is an open source tool for ML lifecycle. <br>
1. Allows us to organise experiements
2. Log Extra Informations about runs

Command to execute MLflow: mlflow ui --port 5000
MLflow URL: http://127.0.0.1:5000 

MLflow allows us to comapre multiple runs and understand the relationship between hyperparameters using Parllel, Scatter, Box and Contour Charts
MLflow supports autologging which logs more informations and create artifact files
