import whisper
import sys
#from gtts import gTTS


import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


def main(audio_file):
    # Load the model
    model = whisper.load_model("large")

    # Transcribe the audio file
    #result = model.transcribe(audio_file, task="translate", language=lang)
    result = model.transcribe(audio_file)
    # Print the translated text
    print(result["text"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python translate_audio.py <audio_file>")
    else:
        audio_file = sys.argv[1]
        main(audio_file)
