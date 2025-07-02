
import os
import random
from ast import literal_eval
from datasets import load_dataset
import PIL
from PIL import Image
import country_converter as coco

def load_datasets():
    try:
        image_dataset = load_dataset("Shresht-Venkat/Adverserial_Cultural-Images")
        flag_dataset = load_dataset("Shresht-Venkat/Country-Flags")
        print("Datasets loaded successfully after cache clearing!")
        return image_dataset, flag_dataset
    except NotImplementedError as e:
        print(f"Still encountering an error: {e}")

def create_geo_dictionaries():

    cc = coco.CountryConverter()
    all_countries = cc.data['name_short']

    continent_dict = {}
    subregion_dict = {}

    for country in all_countries:
        try:
            continent = cc.convert(names=country, to='continent')
            subregion = cc.convert(names=country, to='UNregion')
            if continent not in continent_dict:
                continent_dict[continent] = []
            if subregion not in subregion_dict:
                subregion_dict[subregion] = []
            continent_dict[continent].append(country)
            subregion_dict[subregion].append(country)
        except Exception:
            continue
    return continent_dict, subregion_dict



def generate_q1(item, ans_geography: str, continent_dict: dict, subregion_dict: dict):

    image = item.get('image')
    # countries =  item.get('countries')
    # cultures =  item.get('cultures')
    # subregions =  item.get('un_subregion')
    # continents = item.get('continents')
    countries =  literal_eval(item['countries'])
    cultures =  literal_eval(item['cultures'])
    subregions =  literal_eval(item['un_subregion'])
    continents = literal_eval(item['continents'])
    
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


def generate_q2(item, ans_geography: str, continent_dict: dict, subregion_dict: dict):


    image = item['image']
    countries =  literal_eval(item['countries'])
    cultures =  literal_eval(item['cultures'])
    subregions =  literal_eval(item['un_subregion'])
    continents = literal_eval(item['continents'])
    
    answer_key = {}
    options = []
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






# def generate_mcq(example, ans_geography='continent'):
#     countries = literal_eval(example['countries'])
#     subregions = literal_eval(example['un_subregion'])
#     continents = literal_eval(example['continents'])

#     answer_key = {}
#     options = []
#     mcq_dict = {}

#     if ans_geography == 'continent':
#         continent_countries = set(continent_dict[continents[0]]) - set(countries)
#         sampled = random.sample(continent_countries, 2)
#         correct = sampled
#     elif ans_geography == 'subregion':
#         subregion_countries = set(subregion_dict[subregions[0]]) - set(countries)
#         sampled = random.sample(subregion_countries, 2)
#         correct = sampled
#     else:
#         raise ValueError("Unsupported geography level")

#     distractors = random.sample(set(all_countries) - set(correct) - set(countries), 2)
#     all_options = correct + distractors
#     random.shuffle(all_options)

#     option_labels = ['A', 'B', 'C', 'D']
#     mcq_dict = dict(zip(all_options, option_labels))
#     answer_key = {label: country for country, label in mcq_dict.items() if country in correct}
#     return mcq_dict, answer_key
