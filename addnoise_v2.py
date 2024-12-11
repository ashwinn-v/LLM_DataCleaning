import pandas as pd
import random
import re
import numpy as np

# Load dataset from CSV file
df = pd.read_csv("online_retail_cleaned.csv")

random.seed(42)

# Sample 2000 rows from the dataset
df = df.sample(n=2000, random_state=42)

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

# Function to introduce Typographical Errors
def add_typo(description):
    typo_chance = 0.15
    if random.random() < typo_chance:
        typo_types = ['insert', 'delete', 'substitute']
        typo_type = random.choice(typo_types)
        pos = random.randint(0, len(description) - 1) if description else 0
        if typo_type == 'insert':
            return description[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + description[pos:]
        elif typo_type == 'delete' and len(description) > 1:
            return description[:pos] + description[pos + 1:]
        elif typo_type == 'substitute':
            return description[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + description[pos + 1:]
    return description

# Function to introduce digit transposition errors in numeric fields
def add_numeric_error(value):
    value_str = str(value)
    if len(value_str) > 1 and random.random() < 0.1:
        pos = random.randint(0, len(value_str) - 2)
        value_list = list(value_str)
        value_list[pos], value_list[pos + 1] = value_list[pos + 1], value_list[pos]
        try:
            return float(''.join(value_list))
        except ValueError:
            return value
    return value

# Function to introduce Whitespace Errors in 'Description'
def add_whitespace_error(description):
    whitespace_chance = 0.1
    if random.random() < whitespace_chance:
        if random.choice([True, False]):
            return description + ' ' * random.randint(1, 3)
        else:
            return ' ' * random.randint(1, 3) + description
    return description

# Function to introduce Missing Values
def add_missing_values(value):
    if random.random() < 0.05:
        return np.nan
    return value

# Function to introduce Country Name Inconsistencies
def add_country_inconsistency(country):
    if random.random() < 0.1:
        return country.lower().capitalize()
    return country

# Apply noise to the dataset
df["InvoiceDate"] = df["InvoiceDate"].apply(add_date_format_inconsistency)
df["Description"] = df["Description"].apply(lambda x: add_special_characters(add_typo(add_whitespace_error(x))))
df["Quantity"] = df["Quantity"].apply(add_numeric_error)
df["UnitPrice"] = df["UnitPrice"].apply(add_numeric_error)
df["CustomerID"] = df["CustomerID"].apply(add_missing_values)
df["Country"] = df["Country"].apply(add_country_inconsistency)

# Save the modified DataFrame to a new CSV file
df.to_csv("dataset_with_errors.csv", index=False)

# Display the modified DataFrame
print(df.head())