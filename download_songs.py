from googleapiclient.discovery import build
import yt_dlp
import os
import librosa
import numpy as np
import soundfile as sf
import config

# Set up YouTube Data API
api_key = config.API_KEY  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to search for Korean songs by genre
def search_korean_songs_by_genre(genres, max_results=10):
    all_results = []
    for genre in genres:
        request = youtube.search().list(
            q=f'Korean {genre} playlist',
            part='snippet',
            type='video',
            videoCategoryId='10',  # Numbers of music to download
            maxResults=max_results
        )
        response = request.execute()
        for item in response['items']:
            # Check if the video is a live broadcast
            video_id = item['id']['videoId']
            video_details = youtube.videos().list(part='snippet,liveStreamingDetails', id=video_id).execute()
            if 'liveStreamingDetails' in video_details['items'][0]:
                print(f"Skipping live stream: {item['snippet']['title']}")
                continue
            item['genre'] = genre  # Add genre information to each item
            all_results.append(item)
    return all_results

# Function to sanitize a string to be used as a filename
def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

# Function to download video from YouTube
def download_video_from_youtube(video_id, title, genre, output_dir='audio_files'):
    # Ensure genre-specific output directory exists
    genre_dir = os.path.join(output_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)

    # Sanitize title to create a valid filename
    sanitized_title = sanitize_filename(title)
    
    # Define the output path based on the sanitized title
    output_path = os.path.join(genre_dir, f"{sanitized_title}")
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Skipping download, already exists: {output_path}")
        return
    
    # yt-dlp options to extract audio and save it
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
    except yt_dlp.utils.ExtractorError as e:
        print(f"Skipping {title} due to an error: {e}")

# Function to segment audio into individual songs based on silence detection
def segment_audio(file_path, output_dir='segmented_audio'):
    y, sr = librosa.load(file_path, sr=None)
    intervals = librosa.effects.split(y, top_db=30)  # Adjust top_db as needed

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(intervals):
        segment = y[start:end]
        segment_file_path = os.path.join(output_dir, f'{os.path.basename(file_path).replace(".mp3", "")}_segment_{i}.wav')
        sf.write(segment_file_path, segment, sr)

# Example usage
genres = ["ballad", "rnb", "hiphop", "pop"]
results = search_korean_songs_by_genre(genres)

# Ensure the output directory exists
output_dir = 'audio_files'
os.makedirs(output_dir, exist_ok=True)

# Download each video
for item in results:
    video_id = item['id']['videoId']
    genre = item['genre']
    title = item['snippet']['title']

    # Sanitize the title
    title = sanitize_filename(title)

    print(f'Downloading {title}...')
    download_video_from_youtube(video_id, title, genre, output_dir)

# Segment each downloaded video into individual songs
for genre in genres:
    genre_dir = os.path.join(output_dir, genre)
    if os.path.exists(genre_dir):
        for file in os.listdir(genre_dir):
            if file.endswith('.mp3'):
                file_path = os.path.join(genre_dir, file)
                segment_audio(file_path, output_dir=os.path.join('segmented_audio', genre))
