import whisper
import sys
import time
import numpy as np
from functools import wraps

import torch
import coremltools as ct
import soundfile as sf
from coremltools.models import MLModel
import mlx_whisper


import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


model = whisper.load_model("large")
model.eval()


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper


# Conver the Audio to Mel-spectrogram
def load_audio(filepath):
    audio, _ = sf.read(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel = mel.float()  
    return mel


@measure_time
def audio_to_text(audio_file):
    result = model.transcribe(audio_file)
    print("result in whisper:"+result["text"])
    

@measure_time
def audio_to_text_mlx(audio_file):
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
    #text = mlx_whisper.transcribe(audio_file)["text"]
    print("result in mlx:"+result["text"])



@measure_time
def audio_to_text_coreml(audio_file):

    # load audio file
    mel = load_audio(audio_file).unsqueeze(0)  # 添加批次维度
    # Convert Whisper model to TorchScript
    scripted_model = torch.jit.trace(model, mel)
    # Convert TorchScript Model to Core ML Model
    #mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape)])
    mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape)])
    #mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape, dtype=mel.dtype)])
    mlmodel.save("whisper.mlmodel")
    # Load Core ML Model
    coreml_model = MLModel("whisper.mlmodel")
    # Convert Mel-spectrogram to NumPy Array
    input_data = mel.cpu().numpy()
    result = coreml_model.predict({"input": input_data})
    print(result)


def main(audio_file):
    audio_to_text_mlx(audio_file)
    audio_to_text(audio_file)
    audio_to_text_coreml(audio_file)
   

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python whisper_demo.py <audio_file>")
    else:
        audio_file = sys.argv[1]
        main(audio_file)
