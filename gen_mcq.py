import os
import io
import random
import PIL
from ast import literal_eval
from dotenv import load_dotenv
from PIL import Image
#import google.generativeai as genai
from datasets import load_dataset
import country_converter as coco




# Get the default cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print(f"Hugging Face dataset cache directory: {cache_dir}")
# After trying the above, attempt to load the dataset again
try:
    image_dataset = load_dataset("Shresht-Venkat/Adverserial_Cultural-Images")
    flag_dataset = load_dataset("Shresht-Venkat/Country-Flags")
    print("Datasets loaded successfully after cache clearing!")
except NotImplementedError as e:
    print(f"Still encountering an error: {e}")


# Initialize converter
cc = coco.CountryConverter()

# Get all countries from coco
all_countries = cc.data['name_short']

# Create continent and subregion dictionaries
continent_dict = {}
subregion_dict = {}

for country in all_countries:
    # Replace name if needed
    display_name = "United States of America" if country == "United States" else country
    display_name = 'Russia' if country == 'Russian Federation' else country
    display_name = 'European Union' if country == 'Europe' else country

    continent = cc.convert(names=country, to='continent')
    subregion = cc.convert(names=country, to='UNregion')

    # Build continent dict
    if continent not in continent_dict:
        continent_dict[continent] = []
    continent_dict[continent].append(display_name)

    # Build subregion dict
    if subregion not in subregion_dict:
        subregion_dict[subregion] = []
    subregion_dict[subregion].append(display_name)

# Sort entries (optional)
for d in [continent_dict, subregion_dict]:
    for region in d:
        d[region].sort()







def generate_q1_options_from_metadata(
    image: 'PIL.Image.Image',
    countries: list,
    cultures: list,
    subregions: list,
    continents: list,
    ans_geography: str,
    continent_dict: dict,
    subregion_dict: dict
):
    answer_key = {}
    options = []
    mcq_dict = {}

    if ans_geography == 'continent':
        region = continents[0]
        source_pool = continent_dict.get(region, [])
    elif ans_geography == 'subregion':
        region = subregions[0]
        source_pool = subregion_dict.get(region, [])
    else:
        raise ValueError("ans_geography must be 'continent' or 'subregion'")

    # Filter out countries already in the image's country list
    valid_options = list(set(source_pool) - set(countries))
    if not valid_options:
        return None, None  # No valid distractor, skip

    correct_option = random.choice(valid_options)

    # Compose distractors based on how many countries are already known
    if len(countries) == 1:
        options = [countries[0], correct_option]
    elif len(countries) == 2:
        options = countries + [correct_option]
    else:
        sampled = random.sample(countries, min(3, len(countries)))
        options = sampled + [correct_option]

    random.shuffle(options)
    for i, option in enumerate(options):
        mcq_dict[option] = chr(65 + i)

    # Use file_name or hash of image for tracking
    identifier = getattr(image, 'filename', 'unknown_image')
    answer_key[identifier] = mcq_dict[correct_option]

    return mcq_dict, answer_key


def generate_q2_options_from_metadata(
    image: PIL.Image.Image,
    countries: list,
    cultures: list,
    subregions: list,
    continents: list,
    ans_geography: str,
    continent_dict: dict,
    subregion_dict: dict
):
    answer_key = {}
    mcq_dict_global = {}

    # Select valid region list
    if ans_geography == 'continent':
        region = continents[0]
        region_pool = set(continent_dict.get(region, [])) - set(countries)
    elif ans_geography == 'subregion':
        region = subregions[0]
        region_pool = set(subregion_dict.get(region, [])) - set(countries)
    else:
        raise ValueError("ans_geography must be 'continent' or 'subregion'")

    # If not enough distractors, skip
    if len(region_pool) < 2:
        return None, None

    correct_option1 = random.choice(list(region_pool))
    correct_option2 = random.choice(list(region_pool - {correct_option1}))
    correct_answers = [correct_option1, correct_option2]

    # Build option list
    if len(countries) == 1:
        options = [countries[0], correct_option1, correct_option2]
    elif len(countries) == 2:
        options = countries + [correct_option1, correct_option2]
    else:
        sampled = random.sample(countries, min(3, len(countries)))
        options = sampled + [correct_option1, correct_option2]

    random.shuffle(options)

    # Map options to choices A, B, C...
    mcq_dict = {option: chr(65 + i) for i, option in enumerate(options)}

    # Use file_name or some fallback identifier
    identifier = getattr(image, 'filename', 'unknown_image')
    mcq_dict_global[identifier] = mcq_dict
    answer_key[identifier] = [mcq_dict[opt] for opt in correct_answers]

    return mcq_dict_global[identifier], answer_key


