# Legal Text Classification

This repository contains code for training and predicting legal text classification using BERT.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Configuration](#configuration)

## Installation

1. Clone the repository:
    ```bash
    git clone 
    cd legal_text_classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Linux/Mac
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the [train.py]:

```bash
python train.py
```

### Prediction

```bash
python predict.py
```

### Training Configuration

The training parameters are specified in the config.json file. Here is an example configuration:

```bash
{
  "bert_model_name": "bert-base-uncased",
  "num_classes": 10,
  "max_length": 128,
  "batch_size": 8,
  "num_epochs": 100,
  "learning_rate": 2e-5,
  "data_file": "legal_text_classification.csv"
}
```

### License

This project is licensed under the MIT License.
