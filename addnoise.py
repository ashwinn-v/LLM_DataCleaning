import pandas as pd
import random
import re

# Load dataset from CSV file
df = pd.read_csv("online_retail_cleaned.csv")

random.seed(42)

# Sample 2500 rows from the dataset
df = df.sample(n=2000, random_state=42)
df.to_csv("Gold_truth_dataset.csv", index=False)

# Function to introduce Format Inconsistency in date
def add_date_format_inconsistency(date):
    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%m-%d-%Y %I:%M %p",
        "%Y.%m.%d %H:%M:%S",
    ]
    random_format = random.choice(formats)
    return pd.to_datetime(date).strftime(random_format)

# Function to add Special Characters to 'Description'
def add_special_characters(description):
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*']
    return ''.join([char + random.choice(special_chars) if random.random() < 0.2 else char for char in description])

# Apply noise to the dataset
df["InvoiceDate"] = df["InvoiceDate"].apply(add_date_format_inconsistency)
df["Description"] = df["Description"].apply(add_special_characters)

# Save the modified DataFrame to a new CSV file
df.to_csv("dataset_with_error.csv", index=False)

# Display the modified DataFrame
print(df)
