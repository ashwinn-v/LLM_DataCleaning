
# README

## Overview

This repository demonstrates a pipeline for data cleaning and evaluation using a Large Language Model (LLM) based approach. 

1. **Initial Noise Introduction Script (`modify_dataset.py`)**  
   - Reads a cleaned dataset (`online_retail_cleaned.csv`).
   - Samples rows to create a gold truth dataset (`Gold_truth_dataset.csv`).
   - Introduces format inconsistencies and special characters into the `Description` and `InvoiceDate` fields.
   - Outputs a noisy dataset (`dataset_with_error.csv`).

2. **LLM-based Cleaning and Evaluation Script (`clean_and_evaluate.py`)**  
   - Loads the noisy dataset (`modified_dataset.csv`).
   - Uses the Groq API and a model (e.g., `mixtral-8x7b-32768`) to remove special characters and fix formatting in the `Description` and `InvoiceDate` fields.
   - Saves the cleaned results (`cleaned_dataset_new.csv`).
   - Compares the cleaned dataset against the gold truth dataset (`Gold_truth_dataset.csv`) using standard classification metrics (Accuracy, Precision, Recall, F1).


## Prerequisites

- **Python 3.7+**
- **Pandas** (for data manipulation)
- **Transformers** (for the Hugging Face pipeline, if you choose to run a local model)
- **Torch** (needed as a backend for Transformers, if local)
- **scikit-learn** (for computing evaluation metrics)
- **tqdm** (for progress bars)

Example installation steps:

```bash
pip install pandas transformers torch scikit-learn tqdm
```

**Note:**  
- `torch` installation may vary depending on your system and whether you have GPU support. For GPU usage, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).  
- Ensure you have CUDA drivers if you intend to use GPU acceleration.

## Files Overview

- **`modify_dataset.py`**:  
  Introduces noise into the dataset.
  - **Input:** `online_retail_cleaned.csv`  
  - **Outputs:** `Gold_truth_dataset.csv` and `dataset_with_error.csv`

- **`clean_and_evaluate.py`** (the code snippet provided below):  
  Uses a specified model (via the Groq API) to clean the noisy dataset and then evaluates results.
  - **Input:** `modified_dataset.csv` (the noisy dataset you want to clean), `Gold_truth_dataset.csv` (for evaluation)
  - **Output:** `cleaned_dataset_new.csv`
  - **Prints evaluation metrics (Accuracy, Precision, Recall, F1)**

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

Before running the cleaning code, rename or ensure the noisy dataset is named `modified_dataset.csv` as indicated by the script. For example:

```bash
cp dataset_with_error.csv modified_dataset.csv
```


### 3. Run the Cleaning and Evaluation Script

**`clean_and_evaluate.py` Code Snippet:**

```python
import pandas as pd
import re
from groq import Groq
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


PROMPT_STYLE = 3  

# -------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------
# Style 1: Minimal column information
STYLE_1_PROMPT = """Clean the rows here.
Return your answer as a JSON object with the keys "Description" and "InvoiceDate".

Input:
Description: "{description}"
InvoiceDate: "{invoice_date}"

Output (as JSON):
{{"Description": "<cleaned_description>", "InvoiceDate": "<cleaned_invoice_date>"}}"""

# Style 2: Explicit column names
STYLE_2_PROMPT = """ For the 'Description' column, the description has onformation about the products description and the InvoiceDate column consists of the date and time.
Return your answer as a JSON object with keys "Description" and "InvoiceDate".

Input:
Description: "{description}"
InvoiceDate: "{invoice_date}"

Output (as JSON):
{{"Description": "<cleaned_description>", "InvoiceDate": "<cleaned_invoice_date>"}}"""

# Style 3: Column Names + Gold Truth Format
STYLE_3_PROMPT = """The data you are given includes the columns 'Description' and 'InvoiceDate'.

The gold truth dataset is formatted as follows:
- 'Description': no special characters, only letters, digits, and whitespace, with original spacing and letter casing preserved.
- 'InvoiceDate': strictly in the format 'YYYY-MM-DD HH:MM:SS'.

Please clean the given row accordingly.

Return your answer as a JSON object with the keys "Description" and "InvoiceDate".

Input:
Description: "{description}"
InvoiceDate: "{invoice_date}"

Output (as JSON):
{{"Description": "<cleaned_description>", "InvoiceDate": "<cleaned_invoice_date>"}}"""

# -------------------------------------------------------------------------
# Function to select the prompt style
# -------------------------------------------------------------------------
def get_prompt(description: str, invoice_date: str) -> str:
    if PROMPT_STYLE == 1:
        return STYLE_1_PROMPT.format(description=description, invoice_date=invoice_date)
    elif PROMPT_STYLE == 2:
        return STYLE_2_PROMPT.format(description=description, invoice_date=invoice_date)
    elif PROMPT_STYLE == 3:
        return STYLE_3_PROMPT.format(description=description, invoice_date=invoice_date)
    else:
        raise ValueError("Invalid PROMPT_STYLE selected.")

# -------------------------------------------------------------------------
# Initialize Groq client
# -------------------------------------------------------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load the noisy dataset
df = pd.read_csv("modified_dataset.csv")

def clean_row(description: str, invoice_date: str):
    prompt = get_prompt(description, invoice_date)

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=100,
        )
        
        cleaned_text = response.choices[0].message.content.strip()
        
        # Debug prints
        print("Original Description:", description)
        print("Original InvoiceDate:", invoice_date)
        print("Model Response:", cleaned_text)
        
        # Attempt to parse the JSON from the model response
        try:
            # Using a regex to find a JSON object in the response if any extraneous text is present
            json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                cleaned_data = json.loads(json_str)
            else:
                # If no JSON found, fallback to simple cleaning
                cleaned_data = {
                    "Description": re.sub(r'[!@#$%^&*]', '', description),
                    "InvoiceDate": invoice_date  # no formatting fallback
                }
        except Exception:
            # If JSON parsing fails, fallback again
            cleaned_data = {
                "Description": re.sub(r'[!@#$%^&*]', '', description),
                "InvoiceDate": invoice_date
            }
        
        cleaned_description = cleaned_data.get("Description", description)
        cleaned_invoice_date = cleaned_data.get("InvoiceDate", invoice_date)
        
        # Validate cleaned_invoice_date format
        date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        if not re.match(date_pattern, cleaned_invoice_date):
            cleaned_invoice_date = invoice_date
        
        return cleaned_description, cleaned_invoice_date
        
    except Exception as e:
        print(f"Error processing row: {str(e)}")
        # Fallback to simple regex cleaning if API call fails
        fallback_description = re.sub(r'[!@#$%^&*]', '', description)
        return fallback_description, invoice_date

# Apply cleaning with a progress bar
print("Cleaning the dataset...")
for i in tqdm(range(len(df)), desc="Processing rows", unit="row"):
    desc, inv_date = clean_row(df.at[i, "Description"], df.at[i, "InvoiceDate"])
    df.at[i, "Description"] = desc
    df.at[i, "InvoiceDate"] = inv_date

# Save the cleaned DataFrame
df.to_csv("cleaned_dataset_new.csv", index=False)

# Display the cleaned DataFrame
print("Cleaned Dataset:")
print(df.head())

# -------------------------------------------------------------------------
# Evaluation metrics against gold truth dataset
# -------------------------------------------------------------------------

# Load the gold truth dataset
gold_truth_df = pd.read_csv("Gold_truth_dataset.csv")

# Align both datasets by sorting on a key column
df = df.sort_values(by="InvoiceNo").reset_index(drop=True)
gold_truth_df = gold_truth_df.sort_values(by="InvoiceNo").reset_index(drop=True)

def compute_metrics(cleaned_series, gold_series):
    accuracy = accuracy_score(gold_series, cleaned_series)
    precision = precision_score(gold_series, cleaned_series, average='micro')
    recall = recall_score(gold_series, cleaned_series, average='micro')
    f1 = f1_score(gold_series, cleaned_series, average='micro')

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Evaluate 'Description' column
description_metrics = compute_metrics(df["Description"], gold_truth_df["Description"])
print("\nMetrics for 'Description' column:")
print(description_metrics)

# Evaluate 'InvoiceDate' column
invoice_date_metrics = compute_metrics(df["InvoiceDate"], gold_truth_df["InvoiceDate"])
print("\nMetrics for 'InvoiceDate' column:")
print(invoice_date_metrics)
```


### 4. Check the Results

- **Cleaned Dataset**:  
  Inspect `cleaned_dataset_new.csv` to confirm that special characters are removed and date formats are corrected.

- **Metrics**:  
  The script prints evaluation metrics (Accuracy, Precision, Recall, F1) to the console. Compare these metrics to understand how effective the cleaning step was.

### 5. Adjusting Parameters

- **LLM Prompt**:  
  Edit the prompt in the `clean_and_evaluate.py` script to try different prompt styles (controlled by `PROMPT_STYLE`) or modify the formatting rules.

- **Scoring Method**:  
  Currently uses micro-average metrics. You can change `average='micro'` to `average='macro'` or other methods as needed.

## Troubleshooting

- **Model Loading Errors**:  
  Ensure your `GROQ_API_KEY` is set and that the Groq API credentials are valid. Check your environment variables.

- **GPU Usage**:  
  If you do not have GPU acceleration configured, you may rely on CPU. Adjust model and API calls accordingly.

- **File Not Found**:  
  Ensure that all CSV files (`online_retail_cleaned.csv`, `modified_dataset.csv`, `Gold_truth_dataset.csv`) are in the working directory or adjust file paths accordingly.
```