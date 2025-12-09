# Majority of data cleaning and renaming was coded with the help of Github Copilot
import os
from pydub import AudioSegment
import csv

# Standardize across different audio samples
input_base_dir = '../data'
output_base_dir = '../standardized_data'
os.makedirs(output_base_dir, exist_ok=True)

for subfolder in os.listdir(input_base_dir):
    print(f"Looking at subfolder {subfolder}")
    input_dir = os.path.join(input_base_dir, subfolder)
    output_dir = os.path.join(output_base_dir, subfolder)
    
    if os.path.isdir(input_dir):  
        os.makedirs(output_dir, exist_ok=True)
        
        # Standardize audio file frequency, channels, and sample width
        for filename in os.listdir(input_dir):
            print(f"Processing: {filename}")
            audio = AudioSegment.from_wav(os.path.join(input_dir, filename))
            # Monochannel, 16 kHz, 16-bit wav
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2) 
            audio.export(os.path.join(output_dir, filename), format='wav')


# Create CSV file with .wav path and labels
csv_file_path = os.path.join(output_base_dir, "labels.csv")

with open(csv_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label"])  

    for subfolder in os.listdir(output_base_dir):
        subfolder_path = os.path.join(output_base_dir, subfolder)
        if os.path.isdir(subfolder_path):  
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder, filename)
                writer.writerow([file_path, subfolder])  


# Scripts to help clean general audio files (Not preprocessing, just generalizing samples)
# Script to extract relevant knock and alarm audio files using csv
# folder = "../data/alarm"
# csv_path = "../data/esc50.csv"
# target_label = "siren"
# allowed_files = set()
# with open(csv_path, "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         filename = row[0]  # first column is the filename
#         label = row[3]     # <<< adjust this if label column is different
#         if label == target_label:
#             allowed_files.add(filename)
# for file in os.listdir(folder):
#     if file.endswith(".wav") and file not in allowed_files:
#         path = os.path.join(folder, file)
#         print(f"Deleting {file}")
#         os.remove(path)
# print("Cleanup complete!")


# # Script to convert .wav to .mp3 file and delete original .mp3 files
# import os
# from pydub import AudioSegment
# input_folder = "../data/talking"
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(".mp3"):
#         mp3_path = os.path.join(input_folder, filename)
        
#         # new filename: replace .mp3 with .wav
#         wav_filename = filename[:-4] + ".wav"
#         wav_path = os.path.join(input_folder, wav_filename)
        
#         print(f"Converting {filename} → {wav_filename}")
#         audio = AudioSegment.from_mp3(mp3_path)
#         audio.export(wav_path, format="wav")
        
#         # Delete the original .mp3 file
#         os.remove(mp3_path)
#         print(f"Deleted original file: {filename}")
# print("Done converting all mp3 files!")


# # Script to help trim audio files to 5 seconds
# def trim_audio_files(directory, duration=5000):
#     """
#     Trims all .wav files in the specified directory to the given duration.

#     Args:
#         directory (str): Path to the directory containing .wav files.
#         duration (int): Duration to trim the audio files to, in milliseconds (default is 5000ms).
#     """
#     # Ensure the directory exists
#     if not os.path.exists(directory):
#         print(f"Directory {directory} does not exist.")
#         return

#     # List all .wav files in the directory
#     files = [f for f in os.listdir(directory) if f.endswith(".wav")]

#     for file in files:
#         file_path = os.path.join(directory, file)
#         try:
#             # Load the audio file
#             audio = AudioSegment.from_wav(file_path)
            
#             # Trim the audio to the specified duration
#             trimmed_audio = audio[:duration]
            
#             # Export the trimmed audio back to the same file
#             trimmed_audio.export(file_path, format="wav")
#             print(f"Trimmed: {file}")
#         except Exception as e:
#             print(f"Error processing {file}: {e}")
# if __name__ == "__main__":
#     # Replace 'data/' with the path to your directory
#     trim_audio_files("../data/talking")


# # Script to help rename files in a standardized manner
# # import os
# def rename_files_in_directory(directory):
#     # Ensure the directory exists
#     if not os.path.exists(directory):
#         print(f"Directory {directory} does not exist.")
#         return

#     # Get the parent folder name
#     parent_folder = os.path.basename(os.path.abspath(directory))
    
#     # List all files in the directory
#     files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
#     # Rename files
#     for index, file in enumerate(files, start=1):
#         # Construct the new file name
#         new_name = f"talking{index}.wav"
#         old_path = os.path.join(directory, file)
#         new_path = os.path.join(directory, new_name)

#         # Rename the file
#         os.rename(old_path, new_path)
#         print(f"Renamed: {file} -> {new_name}")
# if __name__ == "__main__":
#     # Replace 'data/' with the path to your directory
#     rename_files_in_directory("../data/talking")