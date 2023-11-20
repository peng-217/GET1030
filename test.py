import pandas as pd
from transformers import pipeline
import sympy
import os
# Read the CSV file
path_to_directory = 'D:/study/AY2023-2024 SEM1/GET1030/project'
os.chdir(path_to_directory)

df = pd.read_csv('customer reviews.csv')

# Load the model using the pipeline from transformers
classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# Define a function to get the predicted emotion with the highest score
def get_emotion(text):
    results = classifier(text, truncation=True)
    return max(results, key=lambda x: x['score'])['label']

# Apply the function to the 'review description' column
# This will create a new 'emotion' column with the predicted emotion
df['emotion'] = df['review description'].apply(get_emotion)

# Save the new dataframe to a new CSV file
df.to_csv('path_to_your_updated_file.csv', index=False)