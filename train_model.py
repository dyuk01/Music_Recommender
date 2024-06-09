import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import extract_mfcc
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

metadata_df = extract_mfcc.metadata_df

# Check the dimensions of the features
print(metadata_df['features'].apply(lambda x: x.shape))

# Prepare features and labels
X = np.array(metadata_df['features'].tolist())
y = metadata_df['genre']

# Check if X or y is empty
if X.size == 0 or len(y) == 0:
    print("X or y is empty")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check if y_encoded is empty
if y_encoded.size == 0:
    print("y_encoded is empty")

# Convert labels to categorical
y_categorical = to_categorical(y_encoded)

# Check dimensions of y_categorical
if y_categorical.size == 0:
    print("y_categorical is empty")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Check dimensions of X_train and y_train
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Build a simple neural network model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model
model.save('singModel.keras')

# Save the label encoder for later use
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Make predictions on the test set
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_)
print(report)
