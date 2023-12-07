import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def data_cleaning(original_data_path='final_data_combined.csv', save_path='preprocessed_data.csv'):
    # Load the original dataset
    original_data = pd.read_csv(original_data_path)

    # Apply filters as described in the pre-processing steps

    # Remove wet races
    filtered_data = original_data[original_data['Race track category'] != 'wet']

    # Remove data of drivers making more than three pit stops
    filtered_data = filtered_data[filtered_data['Remaining pit stops'] <= 3]

    # Remove data of drivers making their final pit stop after 90% race progress
    filtered_data = filtered_data[filtered_data['Race progress'] <= 0.9]

    # Remove data of drivers with a lap time above 200 s or pit stop duration above 50 s
    filtered_data = filtered_data[(filtered_data['Lap time'] <= 200) & (filtered_data['Pit stop duration'] <= 50)]

    # Remove data of drivers with a result position greater than ten
    filtered_data = filtered_data[filtered_data['Result position'] <= 10]

    # Save the pre-processed data to a new CSV file
    filtered_data.to_csv(save_path, index=False)

def data_preprocessing(preprocessed_data_path='preprocessed_data.csv', test_size=0.2):
    # Load the pre-processed dataset
    preprocessed_data = pd.read_csv(preprocessed_data_path)

    # Separate specified test races
    train_data, test_data = train_test_split(preprocessed_data, test_size=test_size, random_state=42)

    # Remove the mean and normalize numerical features
    numerical_features = ['Race progress', 'Tire age progress', 'Lap time', 'Pit stop duration']
    scaler = StandardScaler()
    train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    test_data[numerical_features] = scaler.transform(test_data[numerical_features])

    # One-hot encode categorical features
    categorical_features = ['Position', 'Relative compound', 'Race track category', 'FCY status',
                             'Remaining pit stops', 'Tire change of pursuer', 'Close ahead']
    encoder = OneHotEncoder(drop='first', sparse=False)
    train_data_encoded = pd.DataFrame(encoder.fit_transform(train_data[categorical_features]))
    test_data_encoded = pd.DataFrame(encoder.transform(test_data[categorical_features]))

    # Concatenate encoded features with the original dataframe
    train_data = pd.concat([train_data, train_data_encoded], axis=1)
    test_data = pd.concat([test_data, test_data_encoded], axis=1)

    # Drop the original categorical columns
    train_data = train_data.drop(categorical_features, axis=1)
    test_data = test_data.drop(categorical_features, axis=1)

    # Save the pre-processed and encoded data to new CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

if __name__ == "__main__":
    # data_cleaning()
    pass
