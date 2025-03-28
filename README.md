# Speech Recognition System using DeepSpeech

This project implements a speech recognition system leveraging Mozilla's DeepSpeech model, enabling users to transcribe audio into text accurately and efficiently. The system supports multiple languages and employs advanced audio processing techniques to enhance the quality of input audio before transcription.

## Features

- Language Support: English, Spanish, and Italian.
- Audio Processing: Includes Short-Time Fourier Transform (STFT) for noise reduction and low-pass filtering.
- Performance Evaluation: Utilizes Word Error Rate (WER) to evaluate transcription accuracy.
- Custom Language Models: Supports external scoring to enhance model accuracy.

## Setup and Usage

### Prerequisites

Ensure you have the following installed:

- Python 3.8
- DeepSpeech
- NumPy
- SciPy
- Librosa
- Wave

Download the English deepspeech models named 'deepspeech-0.9.3-models.pbmm' and deepspeech-0.9.3-models.scorer [here](https://github.com/mozilla/DeepSpeech/releases)

### Installation

1. Clone this repository:

```
   git clone <repository-url>
   cd <repository-directory>
```

2. Install the required Python packages:

```
   pip install -r requirements.txt
```

3. Download the DeepSpeech model files for the desired languages.

### Running the System

1. Select Language: The system begins with a user-friendly menu for language selection.

2. Process Audio Files: The audio processing functions handle noise reduction and apply low-pass filters to enhance clarity.

3. Speech-to-Text Conversion: The filtered audio is converted to text using the DeepSpeech model with enhanced scoring for accuracy.

4. Evaluate Accuracy: The system compares the output against the original transcript to calculate the Word Error Rate (WER).

### Example Execution

To run the system, use the following command and follow the prompts for language selection and audio processing:

```
python speech_recognition.py
```

## Contributing

If you'd like to contribute, please follow the standard guidelines for pull requests and report any issues in the issue tracker.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Thanks to Mozilla for the DeepSpeech engine.
- References to the libraries used in audio processing and signal analysis.

This README provides an overview and guidance on setting up and running the speech recognition system. For any further questions or support, please refer to the documentation and community resources provided by the libraries utilized in this project.

 
