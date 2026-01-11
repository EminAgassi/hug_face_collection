# README

## Hug Face Collection

This project provides tools for working with medical transcripts and large language models, including integration with Hugging Face models and Ollama for local inference.

---

## Modules Overview

- **model_hug_face.py**: Loads and interacts with Hugging Face models (e.g., google/gemma-3-4b-it), handles authentication, and provides a chatbot interface for answering medical questions based on transcripts.
- **model_hug_face-env-requirements.txt**: Lists Python dependencies for the Hugging Face model environment.
- **questions.json**: Contains the list of questions to be answered by the model.
- **Other scripts**: May include utilities for transcript processing, Whisper ASR, and Ollama integration.

---

## Environment Setup

### 1. Create a Virtual Environment

```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

### 3. Install Requirements

```powershell
pip install -r model_hug_face-env-requirements.txt
```

**Important Note:**  
For CUDA 12.8 support, install PyTorch and torchvision with:
```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Setting Up Hugging Face Authentication

You must set the `HF_TOKEN` environment variable with your Hugging Face access token for authentication.

**Temporary (for current session):**
```powershell
$env:HF_TOKEN="your_huggingface_token_here"
```

**Permanent (all sessions):**
1. Open "Edit the system environment variables" from the Start menu.
2. Click "Environment Variables..."
3. Under "User variables", click "New..."
4. Set `HF_TOKEN` as the name and paste your token as the value.
5. Click OK.

---

## Usage

1. Activate your virtual environment as shown above.
2. Ensure `HF_TOKEN` is set in your environment.
3. Run the chatbot:
    ```powershell
    py .\model_hug_face.py
    ```

---

## Notes

- The script will attempt to log in to Hugging Face using the `HF_TOKEN` environment variable.
- The model `google/gemma-3-4b-it` is loaded from Hugging Face and run with optimal precision for your hardware.
- For best accuracy, use the Hugging Face version; for speed and convenience on local CPUs, consider Ollama.
- Model files are cached by default in `C:\Users\<your-username>\.cache\huggingface\hub`.

---
