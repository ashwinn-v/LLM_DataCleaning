import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Load the noisy dataset
df = pd.read_csv("modified_dataset.csv")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    device=0  # Use GPU 0 if available
)

# Functions for cleaning
def clean_description(text):
    prompt = f"""Please remove all special characters (including !, @, #, $, %, ^, &, and *) 
    from the following text, leaving only letters, numbers, and standard whitespace. 
    For example, from "SMA*LL WHIT^E HEART OF WIC$KER", produce "SMALL WHITE HEART OF WICKER". 
    Keep the spacing between words intact, and do not alter the case of any letters.

    Input text: "{text}"
    Output text: the cleaned text output from LLM
    """

    response = pipe(prompt)[0]['generated_text']
    
    # Add debugging to see the actual response
    # print('Text input:', text)
    # print('Model response:', response)
    
    cleaned_match = re.search(r'Output text:\s*"([^"]+)"', response)
    if cleaned_match:
        cleaned_text = cleaned_match.group(1)
    else:
        cleaned_text = re.sub(r'[!@#$%^&*]', '', text)
    print('cleaned_text is',cleaned_text)  
    
    return cleaned_text if cleaned_text else text

# Apply cleaning functions with a progress bar
print("Cleaning the dataset...")
for i in tqdm(range(len(df)), desc="Processing rows", unit="row"):
    df.at[i, "Description"] = clean_description(df.at[i, "Description"])


# Save the cleaned DataFrame
df.to_csv("cleaned_dataset.csv", index=False)

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

