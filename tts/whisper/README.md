## Installration

Wisper
'''
pip install -U openai-whisper
'''

ffmpeg
'''
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
'''


## How to run
'''
python simple-demo.py llm_speech.mp3
python simple-demo.py llm_speech_zh.mp3
'''


## how to generate your audio
'''
# need to call google service for gtts.
python audio_gen.py
'''