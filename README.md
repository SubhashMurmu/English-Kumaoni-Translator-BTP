# English → Kumaoni Neural Machine Translation

A Neural Machine Translation (NMT) system for **English-to-Kumaoni translation**, developed as a **Bachelor's Thesis Project (BTP)**.

The system fine-tunes Meta's **mBART-50** multilingual sequence-to-sequence model using **LoRA (Low-Rank Adaptation)** and combines it with a **FAISS-based Translation Memory** to improve translation retrieval and inference.

---

## Overview

Kumaoni is an Indo-Aryan language primarily spoken in the **Kumaon region of Uttarakhand, India**. Due to the limited availability of parallel training data, Kumaoni is considered a **low-resource language** for machine translation.

This project explores the use of parameter-efficient fine-tuning and retrieval-based techniques for building an English-to-Kumaoni translation system.

### Key Components

* **mBART-50** for multilingual neural machine translation
* **LoRA / PEFT** for parameter-efficient fine-tuning
* **FAISS** for efficient similarity search
* **Translation Memory** for retrieving previously seen or similar translations
* **Gradio** for the web-based translation interface

---

## System Architecture

```text
                    English Input
                         │
                         ▼
                ┌─────────────────┐
                │  Preprocessing  │
                └────────┬────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Translation Memory │
              │       (FAISS)       │
              └──────────┬──────────┘
                         │
                 Similarity Search
                         │
                         ▼
              ┌─────────────────────┐
              │   LoRA Fine-tuned   │
              │      mBART-50       │
              └──────────┬──────────┘
                         │
                         ▼
                Kumaoni Translation
                         │
                         ▼
                 Gradio Web Interface
```

---

## Model

The project uses:

**Base Model**

```text
facebook/mbart-large-50-many-to-many-mmt
```

The base mBART-50 model is fine-tuned for the English-to-Kumaoni translation task using **LoRA** through the **PEFT (Parameter-Efficient Fine-Tuning)** framework.

### Why LoRA?

Instead of updating all parameters of the large mBART model, LoRA introduces a relatively small number of trainable parameters.

This provides:

* Lower GPU memory requirements during training
* Faster fine-tuning
* Smaller adapter weights
* Easier model distribution
* Preservation of the original pretrained model

---

## Translation Memory

The inference system is augmented with a **FAISS-based Translation Memory**.

The parallel corpus is converted into vector representations and stored in a FAISS index. During inference, the system searches for translations corresponding to inputs that are semantically similar to the current query.

```text
Parallel Corpus
      │
      ▼
Vector Embeddings
      │
      ▼
FAISS Index ─────────────┐
                         │
English Input            │
      │                  │
      ▼                  │
Vector Representation    │
      │                  │
      └──── Similarity Search
                         │
                         ▼
                Relevant Translations
```

This allows the system to make use of previously observed translation pairs in addition to the neural translation model.

---

## Requirements

* Python **3.10+**
* NVIDIA GPU recommended
* Minimum **8 GB VRAM** for GPU inference
* CUDA-compatible PyTorch installation for GPU acceleration

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SubhashMurmu/English-Kumaoni-Translator-BTP.git
cd English-Kumaoni-Translator-BTP
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch transformers peft gradio faiss-cpu sentencepiece tiktoken
```

> **Note:** `faiss-cpu` is used by default. If you have a compatible CUDA-enabled FAISS installation, you can replace it with the appropriate GPU version.

---

## Acquiring Model Weights

The local inference server requires the **LoRA adapter weights** generated during the training process.

The weights are generated in Google Colab and are not included in the repository. They must be downloaded and placed in the local repository before starting the inference server.

### 1. Run the Training Pipeline

Open the following notebook in **Google Colab**:

```text
english_kumaoni_mbart_finetuning.ipynb
```

Execute the training pipeline.

After training completes, the LoRA adapter weights will be saved in:

```text
lora_weights/
```

### 2. Compress the Weights

In a new Google Colab cell, run:

```bash
!zip -r lora_weights.zip lora_weights/
```

This will create:

```text
lora_weights.zip
```

### 3. Download the Weights

In Google Colab:

1. Open the **Files** panel using the folder icon on the left sidebar.
2. Locate `lora_weights.zip`.
3. Right-click the file.
4. Select **Download**.

### 4. Move the Zip File to the Repository

Move the downloaded `lora_weights.zip` file into the **root directory of this repository**, alongside `app.py`.

The directory should look like:

```text
English-Kumaoni-Translator-BTP/
│
├── app.py
├── build_cache.py
├── lora_weights.zip
├── english_kumaoni_mbart_finetuning.ipynb
└── README.md
```

### 5. Extract the Weights

From the root directory of the repository, run:

```bash
unzip lora_weights.zip -d .
```

After extraction, verify that the `lora_weights/` directory exists:

```text
English-Kumaoni-Translator-BTP/
│
├── app.py
├── build_cache.py
├── lora_weights/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── english_kumaoni_mbart_finetuning.ipynb
└── README.md
```

> **Important:** The `lora_weights/` directory must be located in the **same directory as `app.py`**. The application expects the LoRA adapter weights at this location.

---

## Running the System

The inference pipeline consists of two steps.

### Step 1 — Build the Translation Memory

Before starting the application, generate the FAISS cache:

```bash
python build_cache.py
```

This creates the required translation-memory files:

```text
faiss_cache.index
translation_memory.pkl
```

### Step 2 — Start the Application

Run the Gradio application:

```bash
python app.py
```

The application will start a local web server. The terminal will display the URL, typically:

```text
http://127.0.0.1:7860
```

Open the URL in your web browser to access the translation interface.

---

## Repository Structure

```text
English-Kumaoni-Translator-BTP/
│
├── english_kumaoni_mbart_finetuning.ipynb
│
├── build_cache.py
│
├── app.py
│
├── lora_weights/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
│
├── faiss_cache.index
│
├── translation_memory.pkl
│
└── README.md
```

### File Descriptions

| File / Directory                         | Description                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| `english_kumaoni_mbart_finetuning.ipynb` | Model training, LoRA configuration, and dataset preprocessing |
| `build_cache.py`                         | Builds the FAISS index and translation memory                 |
| `app.py`                                 | Main inference application and Gradio interface               |
| `lora_weights/`                          | LoRA adapter weights generated during model fine-tuning       |
| `faiss_cache.index`                      | FAISS vector index used for translation-memory retrieval      |
| `translation_memory.pkl`                 | Stored translation-memory data                                |
| `README.md`                              | Project documentation                                         |

---

## Training Pipeline

The model training process is implemented in:

```text
english_kumaoni_mbart_finetuning.ipynb
```

The general training workflow is:

```text
Parallel English–Kumaoni Dataset
              │
              ▼
       Data Preprocessing
              │
              ▼
      Tokenization (mBART)
              │
              ▼
       LoRA Configuration
              │
              ▼
      mBART-50 Fine-tuning
              │
              ▼
       LoRA Adapter Weights
              │
              ▼
        Inference System
```

---

## Inference Pipeline

At inference time:

```text
             English Sentence
                    │
                    ▼
             Input Processing
                    │
                    ▼
          Translation Memory
             FAISS Search
                    │
                    ▼
        Relevant Translation Pairs
                    │
                    ▼
       LoRA Fine-tuned mBART
                    │
                    ▼
          Generated Translation
                    │
                    ▼
             Kumaoni Output
```

---

## Technologies Used

| Technology        | Purpose                          |
| ----------------- | -------------------------------- |
| **Python**        | Development language             |
| **PyTorch**       | Deep learning framework          |
| **Transformers**  | mBART model implementation       |
| **PEFT**          | LoRA-based fine-tuning           |
| **FAISS**         | Vector similarity search         |
| **Gradio**        | Web interface                    |
| **SentencePiece** | Tokenization                     |
| **mBART-50**      | Neural machine translation model |

---

## Acknowledgements

This project builds upon the following open-source technologies and research:

* Meta AI's **mBART-50**
* Hugging Face **Transformers**
* Hugging Face **PEFT**
* **FAISS**
* **Gradio**

---

## Authors

**Subhash Murmu** & **Akshat Jain**

Bachelor's Thesis Project
Indian Institute of Technology Kharagpur

---

## License

This project is intended for **academic and research purposes**. Please check the licenses of the underlying models, datasets, and dependencies before redistributing or deploying the system.
