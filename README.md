## A running implementation is found here: [Genre Change](https://huggingface.co/spaces/larimei/genre-change)

## How to run project locally:
# must have miniconda installed
conda create -n style_generation python=3.9
conda activate style_generation
brew install ffmpeg (for mac)
winget install ffmpeg (for windows)
clone repo https://github.com/larimei/Music-KI.git
cd Music-KI
conda install -c pytorch pytorch
python -m pip install -r requirements.txt
python -m demos.style_generation
