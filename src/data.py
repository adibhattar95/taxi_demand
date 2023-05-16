import requests
from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List


def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    downloads data as parquet files from the NYC taxi website
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception (f'{URL} is not available')
    

def validate_raw_data(rides: pd.DataFrame,
                    year: int,
                    month: int) -> pd.DataFrame:
    """
    removes rows with pickup_datetimes outside of their range
    """

    rides['month'] = rides['pickup_datetime'].dt.month
    rides['year'] = rides['pickup_datetime'].dt.year

    rides = rides[(rides['year'] == year) & (rides['month'] == month)]
    return rides


def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:

    """
    Loads raw data from local storage and downloads it from MYC website, and then loads it into a Pandas dataframe
    
    Args:
        year: year of the data to download
        months: months of the data to download, if None, download all months available
    
    Returns:
        pd.DataFrame: DataFrame with the following columns
            - pickup_datetime - datetime of the pickup
            - pickup_location_id - ID of the pickup location
    """
    rides = pd.DataFrame()

    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        months = [months]

    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'

        if not local_file.exists():
            try:
                #Download the file from NYC website
                print(f'Downloading the file - {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} file not available')
                continue
        else:
            print(f'File {year}-{month:02d} was already in local storage')


        #Load the file into pandas
        rides_one_month = pd.read_parquet(local_file)

        #Rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month = rides_one_month.rename(columns = {'tpep_pickup_datetime':'pickup_datetime', 'PULocationID':'pickup_location_id'})

        #Validate the raw file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        #Append to existing data
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        #No data, so we return an empty dataframe
        print('No Data')
        return pd.DataFrame()
    else:
        #Keep only pickup time and location
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides
    
def add_missing_slots(agg_rides:pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the ts_data so that data for every hour and day is avialable for each location
    """

    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq="H")
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        #Keep only rides for this location_id
        agg_rides_i = agg_rides[agg_rides['pickup_location_id'] == location_id][['pickup_hour', 'rides']]

        agg_rides_i = agg_rides_i.set_index('pickup_hour')
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value = 0)
        agg_rides_i['pickup_location_id'] = location_id
        output = pd.concat([output, agg_rides_i])

    output = output.reset_index().rename(columns = {'index':'pickup_hour'})

    return output

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:

    #Sum rides per location and hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides = agg_rides.rename(columns = {0:'rides'})

    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots

def transform_ts_data_into_features_and_target(ts_data: pd.DataFrame, input_seq_len: int, step_size: int) -> pd.DataFrame:
    
    """
    Slices and transforms data from time_series format into a (features, target) format that we can use to train Supervised ML Models
    """

    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):

        #Loop through each location_id
        ts_data_one_location = ts_data[ts_data['pickup_location_id'] == location_id]

        indices = get_cutoff_indices(ts_data_one_location, input_seq_len, step_size)

        #Slice and transform data into numpy arrays for features and targets 
        n_examples = len(indices)
        x = np.ndarray(shape = (n_examples, input_seq_len), dtype = np.float32)
        y = np.ndarray(shape = (n_examples), dtype = np.float32)
        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        #Convert numpy array to dataframe
        features_one_location = pd.DataFrame(x, columns = [f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))])

        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        targets_one_location = pd.DataFrame(y, columns = ['target_rides_next_hour'])

        #Concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])
    
    features = features.reset_index(drop = True)
    targets = targets.reset_index(drop = True)

    return features, targets['target_rides_next_hour']

def get_cutoff_indices(data: pd.DataFrame, n_features: int, step_size: int) -> list :
    stop_position = len(data) - 1
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append([subseq_first_idx, subseq_mid_idx, subseq_last_idx])

        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices


