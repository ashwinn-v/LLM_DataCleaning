import pandas as pd
import re
from groq import Groq
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load the noisy dataset
df = pd.read_csv("modified_dataset.csv")

def clean_description(text):
    prompt = f"""Please remove all special characters (including !, @, #, $, %, ^, &, and *) 
    from the following text, leaving only letters, numbers, and standard whitespace. 
    For example, from "SMA*LL WHIT^E HEART OF WIC$KER", produce "SMALL WHITE HEART OF WICKER". 
    Keep the spacing between words intact, and do not alter the case of any letters.

    Input text: "{text}"
    Output text: the cleaned text output
    just return
    """

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
            max_tokens=50,
        )
        
        cleaned_text = response.choices[0].message.content
        
        # Add debugging to see the actual response
        print('Text input:', text)
        print('Model response:', cleaned_text)
        
        # Try to extract the cleaned text between quotes if present
        cleaned_match = re.search(r'Output text:\s*"?([^"]+)"?', cleaned_text)
        if cleaned_match:
            cleaned_text = cleaned_match.group(1)
        else:
            # Fallback to simple regex cleaning if model output isn't in expected format
            cleaned_text = re.sub(r'[!@#$%^&*]', '', text)
            
        print('cleaned_text is', cleaned_text)
        
        return cleaned_text if cleaned_text else text
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        # Fallback to simple regex cleaning if API call fails
        return re.sub(r'[!@#$%^&*]', '', text)

def clean_invoice_date(date):
    """
    Clean and format a date string using Groq's Mixtral model.
    
    Args:
        date: Input date string in any format
    
    Returns:
        str: Formatted date string in 'YYYY-MM-DD HH:MM:SS' format or original date if parsing fails
    """
    prompt = f"""Convert the following date to the format 'YYYY-MM-DD HH:MM:SS': '{date}'
    Only provide the formatted date as the response, nothing else."""
    
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
            max_tokens=50,
        )
        
        cleaned_text = response.choices[0].message.content
        print('date input is', date)
        print('response is:', cleaned_text)
        
        # Extract the formatted date using regex
        date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        cleaned_date = re.search(date_pattern, cleaned_text)
        print('cleaned date is:', cleaned_date.group(0))
        return cleaned_date.group(0) if cleaned_date else date
        
    except Exception as e:
        print(f"Error processing date: {str(e)}")
        return date

# Apply cleaning functions with a progress bar
print("Cleaning the dataset...")
for i in tqdm(range(len(df)), desc="Processing rows", unit="row"):
    df.at[i, "Description"] = clean_description(df.at[i, "Description"])
    df.at[i, "InvoiceDate"] = clean_invoice_date(df.at[i, "InvoiceDate"])

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

# Evaluate 'InvoiceDate' column
invoice_date_metrics = compute_metrics(df["InvoiceDate"], gold_truth_df["InvoiceDate"])
print("\nMetrics for 'InvoiceDate' column:")
print(invoice_date_metrics)