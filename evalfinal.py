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
        # If the model followed instructions, it should return valid JSON
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
                    "InvoiceDate": invoice_date  # no formatting fallback here
                }
        except Exception:
            # If JSON parsing fails, fallback again
            cleaned_data = {
                "Description": re.sub(r'[!@#$%^&*]', '', description),
                "InvoiceDate": invoice_date
            }
        
        cleaned_description = cleaned_data.get("Description", description)
        cleaned_invoice_date = cleaned_data.get("InvoiceDate", invoice_date)
        
        # Validate cleaned_invoice_date format, if not correct fallback to original
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
