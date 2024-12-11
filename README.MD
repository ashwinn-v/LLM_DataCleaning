
---

# README

## Overview

1. **Initial Noise Introduction Script (modify_dataset.py)**  
   - Reads a cleaned dataset (`online_retail_cleaned.csv`).
   - Samples rows to create a "Gold Truth" dataset (`Gold_truth_dataset.csv`).
   - Introduces format inconsistencies and special characters into the `Description` and `InvoiceDate` fields.
   - Outputs a noisy dataset (`dataset_with_error.csv`).

2. **LLM-based Cleaning and Evaluation Script**
   - Loads the noisy dataset (`modified_dataset.csv`).
   - Uses a Hugging Face pipeline (with the `mistralai/Mistral-7B-v0.1` model) to remove special characters from the `Description` field.
   - Saves the cleaned results (`cleaned_dataset.csv`).
   - Compares the cleaned dataset against the gold truth dataset (`Gold_truth_dataset.csv`) using standard classification metrics (Accuracy, Precision, Recall, F1).

## Prerequisites

- **Python 3.7+**
- **Pandas** (for data manipulation)
- **Transformers** (for the Hugging Face LLM pipeline)
- **Torch** (needed as a backend for Transformers)
- **scikit-learn** (for computing evaluation metrics)
- **tqdm** (for progress bars)

Example installation steps:

```bash
pip install pandas transformers torch scikit-learn tqdm
```

*Note:*  
- `torch` installation may vary depending on your system and whether you have GPU support. For GPU usage, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- Ensure you have `cuda` drivers if you intend to use GPU acceleration (`device=0` parameter in the code).

## Files Overview

- **`modify_dataset.py`**:  
  Introduces noise into the dataset.
  - Input: `online_retail_cleaned.csv`  
  - Output: `Gold_truth_dataset.csv` and `dataset_with_error.csv`

- **`clean_and_evaluate.py`** (example name for the second code snippet):  
  Uses the mistralai/Mistral-7B model to clean the noisy dataset and then evaluates results.
  - Input: `modified_dataset.csv` (assumed to be the noisy dataset you want to clean), `Gold_truth_dataset.csv` (for evaluation)
  - Output: `cleaned_dataset.csv`  
  - Prints evaluation metrics (Accuracy, Precision, Recall, F1)

## Detailed Steps

### 1. Run the Initial Noise Introduction Script

1. **Ensure you have `online_retail_cleaned.csv`**  
   This should be a cleaned version of the Online Retail dataset or a dataset with a similar schema (including `InvoiceNo`, `Description`, and `InvoiceDate` columns).

2. **Run the script:**
   ```bash
   python modify_dataset.py
   ```

3. **Result:**
   - `Gold_truth_dataset.csv`: A sample of the original dataset without noise.
   - `dataset_with_error.csv`: The dataset with intentionally introduced noise (irregular date formats, special characters in `Description`).

### 2. Prepare the Noisy Dataset for Cleaning

Before running the cleaning code, you may need to rename or ensure the noisy dataset is named `modified_dataset.csv` as indicated by the script. You can simply copy or rename `dataset_with_error.csv` to `modified_dataset.csv` if needed:

```bash
cp dataset_with_error.csv modified_dataset.csv
```


- **Load Noisy Dataset**: Reads `modified_dataset.csv`.
- **LLM-based Cleaning**:  
  Uses `mistralai/Mistral-7B-v0.1` model via the Hugging Face `transformers` pipeline to remove special characters from `Description`. The prompt is given to the model, and the model’s response is parsed to extract the cleaned text.
  
- **Saving Cleaned Results**:  
  The cleaned DataFrame is saved as `cleaned_dataset.csv`.

- **Evaluation**:  
  Loads `Gold_truth_dataset.csv` for comparison. Uses `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` (with `average='micro'`) from `scikit-learn` to measure how closely the cleaned `Description` matches the gold standard.

### 4. Check the Results

- **Cleaned Dataset**:  
  Inspect `cleaned_dataset.csv` to confirm special characters are removed.
  
- **Metrics**:  
  The script prints evaluation metrics to the console. Compare these metrics to understand how effective the cleaning step was.

### 5. Adjusting Parameters

- **LLM Prompt**:  
  Edit the prompt in the `clean_and_evaluate.py` script if you need a different cleaning approach or if you want to try another model.
  
- **Scoring Method**:  
  The evaluation currently uses micro-average metrics. You can change the averaging method (e.g., `average='macro'`) or use different metrics as required.

## Troubleshooting

- **Model Loading Errors**:  
  Ensure you have a stable internet connection since Hugging Face model weights must be downloaded the first time you run the script.
  
- **GPU Usage**:  
  If you do not have a GPU, remove `device=0` from the pipeline call or set `device=-1` to use CPU. Using CPU may be slower.
  
- **File Not Found**:  
  Ensure all CSV files (`online_retail_cleaned.csv`, `modified_dataset.csv`, `Gold_truth_dataset.csv`) are in the working directory or adjust file paths accordingly.

