
## A running implementation is found here: 
[Genre Change](https://huggingface.co/spaces/larimei/genre-change)

## How to run project locally:
- **Prerequisites**: Must have miniconda installed.
- **Environment Setup**:
  ```
  conda create -n style_generation python=3.9
  conda activate style_generation
  ```
- **FFmpeg Installation**:
  - For Mac: `brew install ffmpeg`
  - For Windows: `winget install ffmpeg`
- **Repository Setup**:
  ```
  clone repo https://github.com/larimei/Music-KI.git
  cd Music-KI
  ```
- **Dependencies Installation**:
  ```
  conda install -c pytorch pytorch
  python -m pip install -r requirements.txt
  ```
- **Run the Demo**:
  ```
  python -m demos.style_generation
  ```
