# Hug Face Collection

This repository contains a collection of modules for speech-to-text, text-to-speech, and model integration using Hugging Face and related tools.

## Modules

- **model_hug_face.py**
  - Integrates and manages Hugging Face models for various NLP tasks.

- **speech-to-text-bm25.py**
  - Provides speech-to-text functionality and leverages BM25 for text retrieval or ranking.

- **test.py**
  - Contains test scripts and examples to validate the functionality of the modules.

- **model_hug_face-env-requirements.txt**
  - Environment requirements for running Hugging Face model integrations.

- **tts-env-requirements.txt**
  - Environment requirements for text-to-speech modules.

- **whisper-env-requirements.txt**
  - Environment requirements for Whisper-based speech-to-text modules.

- **replace.txt**
  - (Purpose unspecified; likely used for text replacement or configuration.)



## Important Note for PyTorch Installation

If you require CUDA 12.8 support, install torch and torchvision using the following command **after creating and activating your virtual environment**:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

This ensures you get the correct CUDA-enabled versions. Do this instead of installing torch/torchvision from the requirements file if you encounter installation errors.

## Environment Setup


To use the modules, you should create a separate Python virtual environment for each requirements file as needed. 

**Important for Windows users:**
Before activating a virtual environment, you may need to run the following command in PowerShell to allow script execution for the current process:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Below are the instructions for each:

### 1. model_hug_face-env-requirements.txt

```
python -m venv venv_hugface
venv_hugface\Scripts\activate  # On Windows
pip install -r model_hug_face-env-requirements.txt
```

### 2. tts-env-requirements.txt

```
python -m venv venv_tts
venv_tts\Scripts\activate  # On Windows
pip install -r tts-env-requirements.txt
```

### 3. whisper-env-requirements.txt

```
python -m venv venv_whisper
venv_whisper\Scripts\activate  # On Windows
pip install -r whisper-env-requirements.txt
```

Deactivate the environment with:
```
deactivate
```

## Usage

1. Activate the appropriate environment for the module you want to use.
2. Run the desired module as needed.
3. Refer to `test.py` for usage examples.

---

Feel free to contribute or open issues for improvements!
