# START of code personally written without assistance

# Import the necessary modules
import deepspeech
import wave
import numpy as np
from scipy.io import wavfile
import scipy
import librosa

# Print the starting menu for language selection
print("Select your language-------------Seleccione su idioma-------------Seleziona la tua lingua")

# Define a dictionary to map language input to model files and scorer files
language_options = {    
    "english": ("en", 'models/EN/deepspeech-0.9.3-models.pbmm', 'models/EN/deepspeech-0.9.3-models.scorer'),
    "en": ("en", 'models/EN/deepspeech-0.9.3-models.pbmm', 'models/EN/deepspeech-0.9.3-models.scorer'),
    "espanol": ("es", 'models/ES/output_graph_es.pbmm', 'models/ES/kenlm_es.scorer'),
    "es": ("es", 'models/ES/output_graph_es.pbmm', 'models/ES/kenlm_es.scorer'),
    "italiana": ("it", 'models/IT/output_graph_it.pbmm', 'models/IT/kenlm_it.scorer'),
    "it": ("it", 'models/IT/output_graph_it.pbmm', 'models/IT/kenlm_it.scorer')
}

# Initialize validation flag
valid = False

# Continue to prompt user until a valid language is selected
while not valid:
    # Prompt user for language choice
    username = input("English(EN), Espanol(ES), Italiana(IT): ").lower()
    
    # Check if the user's input matches any language option
    if username in language_options:
        # Retrieve the language code, model file path, and scorer file path
        language, model_file_path, scorer_file_path = language_options[username]
        
        # Load DeepSpeech model and enable external scorer
        model = deepspeech.Model(model_file_path)
        model.enableExternalScorer(scorer_file_path)
        
        # Set validation flag to true to exit loop
        valid = True
    else:
        # Notify user of invalid selection
        print("Invalid language selection. Please try again.")

# Enable external scorer for improved accuracy in speech recognition
model.enableExternalScorer(scorer_file_path)

# Set language model parameters for scoring to balance recognition
lm_alpha = 0.75  # Language model weight affects how much the model relies on the language probability
lm_beta = 1.85   # Word insertion bonus favors shorter outputs by penalizing insertions

# Apply the scorer parameters for enhanced model accuracy
model.setScorerAlphaBeta(lm_alpha, lm_beta)

# Set the beam width for the search network to manage the trade-off between speed and accuracy
beam_width = 500
model.setBeamWidth(beam_width)

# Retrieve the desired sample rate for resampling audio inputs to match the model's requirements
desired_sample_rate = model.sampleRate()

# Define filenames for each language using a dictionary
language_files = {
    "en": [
        'Ex4_audio_files/EN/checkin.wav', 'Ex4_audio_files/EN/parents.wav',
        'Ex4_audio_files/EN/suitcase.wav', 'Ex4_audio_files/EN/what_time.wav',
        'Ex4_audio_files/EN/where.wav', 'Ex4_audio_files/EN/your_sentence1.wav', 
        'Ex4_audio_files/EN/your_sentence2.wav'
    ],
    "es": [
        'Ex4_audio_files/ES/checkin_es.wav', 'Ex4_audio_files/ES/parents_es.wav',
        'Ex4_audio_files/ES/suitcase_es.wav', 'Ex4_audio_files/ES/what_time_es.wav',
        'Ex4_audio_files/ES/where_es.wav'
    ],
    "it": [
        'Ex4_audio_files/IT/checkin_it.wav', 'Ex4_audio_files/IT/parents_it.wav',
        'Ex4_audio_files/IT/suitcase_it.wav', 'Ex4_audio_files/IT/what_time_it.wav',
        'Ex4_audio_files/IT/where_it.wav'
    ]
}

# Select filenames based on the chosen language
filenames = language_files.get(language, [])
converted_texts = []

# Initialize index for processed audio files
m = 1

# Path to the crowd noise audio file
noise_file = 'Ex4_audio_files/crowd_noise.wav'

# Load noise audio file with the desired sample rate
ats, y_sampling_rate = librosa.load(noise_file, sr=desired_sample_rate)

# Perform Short-Time Fourier Transform (STFT) on the noise file
transformed_ats = librosa.stft(ats)

# Calculate the magnitude spectrum of the transformed audio
nss = np.abs(transformed_ats)

# Compute the mean spectrum across time for noise reduction
mns = np.mean(nss, axis=1)

def apply_stft_and_noise_subtraction(filename, mns):
    """
    Perform Short-Time Fourier Transform (STFT) and apply noise subtraction.
    """
    # Load the audio file
    file_ats, y_sr = librosa.load(filename, sr=None, mono=True)

    # Perform STFT to decompose the audio into frequency components
    transformed_ats = librosa.stft(file_ats)

    # Compute the magnitude of the frequency spectrum
    magnitude = np.abs(transformed_ats)

    # Compute the phase angle of the frequency spectrum
    phase_angle = np.angle(transformed_ats)

    # Reconstruct the complex spectrum using the phase angle
    complex_spectrum = np.exp(1.0j * phase_angle)
    
    # Subtract the mean noise spectrum from the magnitude spectrum
    subtracted_audio = (magnitude - mns.reshape((mns.shape[0], 1))) * complex_spectrum

    # Perform inverse STFT to convert the frequency-domain audio back to time-domain
    return librosa.istft(subtracted_audio), y_sr

def write_wav_file(file_path, sample_rate, audio_data):
    """
    Write audio data to a WAV file.
    """
    # Converts normalized audio data into 16-bit PCM format before writing
    scipy.io.wavfile.write(file_path, sample_rate, (audio_data * 32768).astype(np.int16))


def perform_low_pass_filter(input_file, cut_off_frequency, m):
    """
    Apply a low-pass filter to the audio.
    """
    # Read audio data and sample rate from the input WAV file
    freq_sampling_rate, data = wavfile.read(input_file)

    # Calculate the frequency ratio for the filter cutoff
    freq_ratio = cut_off_frequency / freq_sampling_rate

    # Determine the filter length N based on a given formula, ensuring a smooth cutoff
    N = int(np.sqrt(0.196201 + freq_ratio**2) / freq_ratio)

    # Create a simple moving average filter window
    window = np.ones(N) / N

    # Apply the low-pass filter and ensure the output is in 16-bit integer format
    return scipy.signal.lfilter(window, [1], data).astype(np.int16), freq_sampling_rate

def process_audio_files(filenames, mns, language, m):
    """
    Processes a list of audio files using noise reduction and filtering techniques, then converts speech to text.
    """
    converted_texts = []
    
    # Set cutoff frequency depending on the language
    cut_off_frequency = 2000.0 if language in {"en", "it"} else 3000.0
    
    for filename in filenames:
        # Apply STFT (Short-Time Fourier Transform) and noise subtraction
        y, y_sr = apply_stft_and_noise_subtraction(filename, mns)

        # Write the noise-reduced audio to a file
        reduced_noise_file = f"Ex4_audio_files/mywav_reduced_noise{m}.wav"
        write_wav_file(reduced_noise_file, y_sr, y)

        # Apply a low-pass filter to the noise-reduced audio
        filtered, desired_sample_rate = perform_low_pass_filter(reduced_noise_file, cut_off_frequency, m)

        # Open the noise-reduced audio file to retrieve audio parameters
        with wave.open(reduced_noise_file, 'r') as wf:
            amp_width = wf.getsampwidth()  # Get byte width of audio samples
            n_frames = wf.getnframes()     # Get total number of frames

        # Write the filtered audio to a new file
        filtered_file = f"Ex4_audio_files/mywav_reduced_noise{m}filtered.wav"
        with wave.open(filtered_file, 'w') as wf:
            wf.setnchannels(1)                     # Set to mono channel
            wf.setsampwidth(amp_width)             # Set sample width from original file
            wf.setframerate(desired_sample_rate)   # Set sample rate from filtered data
            wf.writeframes(filtered.tobytes('C'))  # Write audio data

        # Convert the filtered audio to text using Speech-to-Text model
        with wave.open(filtered_file, 'r') as wf:
            frames = wf.readframes(wf.getnframes())  # Read all frames
            data16 = np.frombuffer(frames, dtype=np.int16)  # Convert frames to 16-bit integer numpy array
            text = model.stt(data16)  # Perform speech-to-text conversion
            converted_texts.append(text)  # Add the transcribed text to results

        m += 1  # Increment the file index

    return converted_texts  # Return list of converted text from each audio file

# Process files based on the language  
converted_texts = process_audio_files(filenames, mns, language, m)  

# Define a function to normalize text by removing punctuation and converting to lowercase  
def normalize_text(text):  
    punctuation_chars = {"?", ".", ";", ":", "!", ",", "¿", "'"}  
    text = "".join(character for character in text if character not in punctuation_chars)  
    return text.replace("-", " ").lower()  

# Set the appropriate transcript texts based on the selected language  
if language == "en":  
    transcript_texts = [  
        "Where is the check-in desk?", "I have lost my parents.",
        "Please, I have lost my suitcase.", "What time is my plane?", 
        "Where are the restaurants and shops?", "Where is the lounge?", 
        "Is the flight on time?"  
    ]
elif language == "es":  
    transcript_texts = [  
        "¿Dónde están los mostradores?", "He perdido a mis padres.",  
        "Por favor, he perdido mi maleta.", "¿A qué hora es mi avión?",  
        "¿Dónde están los restaurantes y las tiendas?"  
    ]
elif language == "it":  
    transcript_texts = [  
        "Dove e' il bancone?", "Ho perso i miei genitori.",  
        "Per favore, ho perso la mia valigia.", "A che ora e' il mio aereo?",  
        "Dove sono i ristoranti e i negozi?"  
    ]

# Normalize the transcript texts 
# This applies the 'normalize_text' function to each text in 'transcript_texts',
# which helps in standardizing the texts for accurate comparison
transcript_texts = [normalize_text(text) for text in transcript_texts]  

def calculate_wer(converted_texts, transcript_texts):
    # Initialize counters for total errors and total words
    total_errors = 0  
    total_words = 0  
    
    # Loop over each pair of converted and transcript text
    # 'enumerate' provides both the index and the converted text for easy pairing
    for i, converted_text in enumerate(converted_texts):  
        print(f"Recognized text: {converted_text}")  
        print(f"Transcript text: {transcript_texts[i]}")  
        
        # Split texts into lists of words for detailed comparison
        converted_words = converted_text.split(" ")  
        transcript_words = transcript_texts[i].split(" ")  
        
        # Calculate the sum of substitution and deletion errors
        # Errors are determined by missing matches between converted and transcript words
        subs_and_dels = len(transcript_words) - len(set(converted_words).intersection(transcript_words))  
        
        # Calculate insertion errors, which are extra words in the converted text
        insertion_errors = max(0, len(converted_words) - len(transcript_words))  
        
        # Aggregate the total errors (substitutions, deletions, insertions)
        total_sdi = subs_and_dels + insertion_errors  
        
        # Get the total number of words in the transcript for calculating WER
        num_words = len(transcript_words)  
        
        print(f"Number of errors: {total_sdi}")  
        print(f"Total number of words: {num_words}")  
        
        # Calculate Word Error Rate (WER) for the current text pair
        current_wer = (total_sdi / num_words) * 100  
        print(f"WER for current transcript: {current_wer:.2f}%")  
        
        # Accumulate the errors and words for overall WER
        total_errors += total_sdi  
        total_words += num_words  
    
    # Calculate and print overall WER across all text samples
    overall_wer = (total_errors / total_words) * 100  
    print(f"Overall {language} WER: {overall_wer:.2f}%")  

# Call the function with your text data to evaluate WER
calculate_wer(converted_texts, transcript_texts)

# END of code personally written without assistance