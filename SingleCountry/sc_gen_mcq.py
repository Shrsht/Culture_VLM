
import os
import random
import pandas as pd

def generate_mcq(image_name, metadata_df):
    """
    Generates a multiple-choice question with 4 options.
    The correct option is the country from the metadata.
    The other three options are randomly selected countries.
    """
    # Get the correct country from the metadata
    correct_country = metadata_df.loc[metadata_df['image'] == image_name, 'country'].iloc[0]

    # Get a list of all unique countries
    all_countries = metadata_df['country'].unique().tolist()

    # Remove the correct country from the list of all countries
    all_countries.remove(correct_country)

    # Randomly select 3 other countries
    distractors = random.sample(all_countries, 3)

    # Create the options and shuffle them
    options = [correct_country] + distractors
    random.shuffle(options)

    # Create the MCQ dictionary
    mcq = {opt: chr(65 + i) for i, opt in enumerate(options)}

    # Create the answer key
    answer_key = {image_name: mcq[correct_country]}

    return mcq, answer_key

if __name__ == '__main__':
    # Load the metadata
    metadata_df = pd.read_csv('../mosaic_metadata.csv')

    # Example usage:
    image_name = '1.jpg'
    mcq, answer_key = generate_mcq(image_name, metadata_df)
    print("MCQ:", mcq)
    print("Answer Key:", answer_key)
