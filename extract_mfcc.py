import librosa
import numpy as np
import pandas as pd
import os

# Function to extract MFCC features
def extract_features(file_path):
    '''
    librosa will load the file in file_path without resampling it
    y : audio signal data
    sr : sample rate
    '''
    y, sr = librosa.load(file_path, sr=None)
    # mfcc coefficient is 25 since music needs 20~40, and speech recognition needs 12~13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    # Calculates the mean value of each MFCC coefficient across all time dimension
    # If we are working with 3D data, axis becomes axis=2
    mfcc_mean = np.mean(mfcc, axis=1)
    # Standard deviation
    mfcc_std = np.std(mfcc, axis=1)
    # Concatenates the mean and standard deviation arrays into a single feature vector
    return np.concatenate([mfcc_mean, mfcc_std])

# Directory containing all segmented audio files
segmented_audio_dir = 'segmented_audio'

# Extract features from all audio files and organize into a DataFrame
metadata = {'filename': [], 'features': [], 'genre': []}

for root, _, files in os.walk(segmented_audio_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            genre = os.path.basename(root)  # Use directory name as genre label
            try:
                features = extract_features(file_path)
                if features.size == 0:
                    print(f"Empty features for {file_path}")
                metadata['filename'].append(file)
                metadata['features'].append(features)
                metadata['genre'].append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert metadata to DataFrame
metadata_df = pd.DataFrame(metadata)

# Check if DataFrame is empty
if metadata_df.empty:
    print("Metadata DataFrame is empty")