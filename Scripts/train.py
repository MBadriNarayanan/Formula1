import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import tensorflow as tf
from preprocessing import data_cleaning, data_preprocessing
from keras.activations import ReLU, sigmoid

# Step 1: Data Cleaning
data_cleaning()

# Step 2: Data Preprocessing
data_preprocessing()

# Step 3: Load the pre-processed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Step 4: Split the data into features and labels
X_train = train_data.drop('Pit stop decision', axis=1)
y_train = train_data['Pit stop decision']

X_test = test_data.drop('Pit stop decision', axis=1)
y_test = test_data['Pit stop decision']

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='ReLU', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='ReLU'),
    tf.keras.layers.Dense(64, activation='ReLU'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 7: Compile the model
class_weights = {0: 1, 1: 5}
model.compile(optimizer=tf.keras.optimizers.Nadam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy'],
              class_weight=class_weights)

# Step 8: Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# Step 9: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Step 10: Make predictions
predictions = model.predict(X_test_scaled)

# Convert probabilities to binary predictions
binary_predictions = (predictions > 0.5).astype(int)

# Calculate and print F1 score
f1 = f1_score(y_test, binary_predictions)
print(f'F1 Score: {f1}')

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_test, binary_predictions)
print('Confusion Matrix:')
print(conf_matrix)