import whisper
import sys
import time
import numpy as np
from functools import wraps

import torch
import coremltools as ct
import soundfile as sf
from coremltools.models import MLModel


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
def audio_to_text_coreml(audio_file):

    # load audio file
    mel = load_audio(audio_file).unsqueeze(0)  # 添加批次维度
    # Convert Whisper model to TorchScript
    scripted_model = torch.jit.trace(model, mel)
    # 将 TorchScript 模型转换为 Core ML 模型
    #mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape)])
    mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape)])
    #mlmodel = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=mel.shape, dtype=mel.dtype)])
    # 保存 Core ML 模型
    mlmodel.save("whisper.mlmodel")
    # 加载 Core ML 模型
    coreml_model = MLModel("whisper.mlmodel")
    # 将 Mel-spectrogram 转换为 NumPy 数组
    input_data = mel.cpu().numpy()
    result = coreml_model.predict({"input": input_data})
    print(result)





@measure_time
def audio_to_text(audio_file):
    result = model.transcribe(audio_file)
    print(result["text"])
    


def main(audio_file):
    audio_to_text(audio_file)
    audio_to_text_coreml(audio_file)
   

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python whisper_demo.py <audio_file>")
    else:
        audio_file = sys.argv[1]
        main(audio_file)
