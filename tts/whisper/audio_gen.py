from gtts import gTTS


import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


# Define the text for the speech
text_en = """
Ladies and gentlemen,

Today, I'd like to talk about Large Language Models, or LLMs. These models represent a significant advancement in artificial intelligence. They are capable of understanding and generating human-like text based on vast amounts of data. LLMs can assist in various fields such as customer service, content creation, and even complex problem-solving. By leveraging LLMs, we can enhance efficiency, creativity, and innovation in countless industries. The future of AI, powered by LLMs, holds immense potential to transform our world in unimaginable ways.

Thank you.
"""

text_zh ="""
女士们，先生们，

今天，我想谈谈大型语言模型（LLM）。这些模型代表了人工智能的重大进步。它们能够基于大量数据理解和生成类似人类的文本。LLM 可以在各个领域提供帮助，如客户服务、内容创作，甚至复杂问题的解决。通过利用 LLM，我们可以在无数行业中提升效率、创造力和创新能力。由 LLM 驱动的 AI 未来，具有改变我们世界的巨大潜力，以难以想象的方式实现这一目标。

谢谢。

"""


# Create the speech
tts = gTTS(text_en, lang='en')
file_path = "llm_speech.mp3"
tts.save(file_path)

tts = gTTS(text_zh, lang='zh-cn')
file_path = "llm_speech_zh.mp3"
tts.save(file_path)

