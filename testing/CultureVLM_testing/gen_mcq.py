
import os
import random
from ast import literal_eval
from datasets import load_dataset
import country_converter as coco
# import country_converter as coco

def load_datasets():
    try:
        image_dataset = load_dataset("Shresht-Venkat/Adverserial_Cultural-Images")
        flag_dataset = load_dataset("Shresht-Venkat/Country-Flags")
        print("Datasets loaded successfully after cache clearing!")
        return image_dataset, flag_dataset
    except NotImplementedError as e:
        print(f"Still encountering an error: {e}")


# ------------------------------------------------------------------
# 1. master mapping supplied by you
flag_dict = {
    "Bolivia": "Bolivia, Plurinational State of",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "DR Congo": "Congo, the Democratic Republic of the",
    "Falkland Islands": "Falkland Islands (Malvinas)",
    "Heard and McDonald Islands": "Heard Island and McDonald Islands",
    "Iran": "Iran, Islamic Republic of",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Micronesia, Fed. Sts.": "Micronesia, Federated States of",
    "Moldova": "Moldova, Republic of",
    "Russia": "Russian Federation",
    "Sint Maarten": "Sint Maarten (Dutch part)",
    "South Georgia and South Sandwich Is.": (
        "South Georgia and the South Sandwich Islands"
    ),
    "St. Barths": "Saint Barthelemy",
    "St. Helena": "Saint Helena, Ascension and Tristan da Cunha",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Pierre and Miquelon": "Saint Pierre and Miquelon",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Syrian Arab Republic": "Syria",
    "Tanzania": "Tanzania, United Republic of",
    "Türkiye": "Turkey",
    "United States of America": "United States",
    "Venezuela": "Venezuela, Bolivarian Republic of",
}

# ------------------------------------------------------------------
# 2. a single normalisation helper
def normalize_country(name: str, mapping: dict = flag_dict) -> str:
    """
    Return the spelling that exists in the flag-dataset.
    If *name* is a key in *mapping*, use the mapped value,
    otherwise leave it unchanged.
    """
    return mapping.get(name, name)


DELETE_LIST = [
    'Aland Islands', 'Holy See (Vatican City State)','Bonaire, Saint Eustatius and Saba',
    'British Virgin Islands','Congo Republic','Eswatini','Guam','Macau','Taiwan',
    'Saint-Martin',"St. Pierre and Miquelon",'United States Minor Outlying Islands',
    'United States Virgin Islands'
]


# ------------------------------------------------------------------
# 3. build continent / sub-region look-ups with *normalised* names
def create_geo_dictionaries(delete_list=None):
    """
    Returns:
        continent_dict[continent]  -> list of country names
        subregion_dict[subregion] -> list of country names
    """
    if delete_list is None:
        delete_list = DELETE_LIST
    cc = coco.CountryConverter()
    raw_countries = set(cc.data["name_short"]) - set(delete_list)

    continent_dict, subregion_dict = {}, {}

    for c in raw_countries:
        country = normalize_country(c)  # *** single point of truth ***
        try:
            continent = cc.convert(names=c, to="continent")
            subregion = cc.convert(names=c, to="UNregion")
        except Exception:
            # skip corner-cases the converter can’t handle
            continue

        continent_dict.setdefault(continent, []).append(country)
        subregion_dict.setdefault(subregion, []).append(country)

    return continent_dict, subregion_dict


# ------------------------------------------------------------------
# 4. Q-generation helpers – always work with normalised lists
def generate_q1(
    item,
    ans_geography: str,
    continent_dict: dict,
    subregion_dict: dict,
):
    """
    One-answer MCQ:
        – pick one country in same continent/sub-region but
          NOT already present in the image's list.
        – options = existing countries (+ distractor) + correct answer
    """
    countries = [normalize_country(c) for c in literal_eval(item["countries"])]
    countries = [c for c in countries if c not in DELETE_LIST]
    continents = literal_eval(item["continents"])
    subregions = literal_eval(item["un_subregion"])

    # ------------- choose the candidate pool -----------------
    if ans_geography == "continent":
        pool = set(continent_dict.get(continents[0], []))
    elif ans_geography == "subregion":
        pool = set(subregion_dict.get(subregions[0], []))
    else:
        raise ValueError("ans_geography must be 'continent' or 'subregion'")

    pool -= set(countries)  # never repeat what’s already in the image
    pool -= set(DELETE_LIST) # remove any countries that are in the delete list
    if not pool:
        return None, None  # give up if nothing to pick from

    correct = random.choice(list(pool))

    # ------------- compose and shuffle options ---------------
    if len(countries) <= 2:
        options = countries + [correct]
    else:
        options = random.sample(countries, 2) + [correct]

    random.shuffle(options)
    mcq = {opt: chr(65 + i) for i, opt in enumerate(options)}

    img_id = getattr(item["image"], "filename", "unknown_image")
    answer_key = {img_id: mcq[correct]}
    return mcq, answer_key


def generate_q2(
    item,
    ans_geography: str,
    continent_dict: dict,
    subregion_dict: dict,
):
    """
    Two-correct-answers version (select 2 countries in same region, etc.)
    """
    countries = [normalize_country(c) for c in literal_eval(item["countries"])]
    countries = [c for c in countries if c not in DELETE_LIST]
    continents = literal_eval(item["continents"])
    subregions = literal_eval(item["un_subregion"])

    if ans_geography == "continent":
        pool = set(continent_dict.get(continents[0], []))
    elif ans_geography == "subregion":
        pool = set(subregion_dict.get(subregions[0], []))
    else:
        raise ValueError("ans_geography must be 'continent' or 'subregion'")

    pool -= set(countries)
    pool -= set(DELETE_LIST) # remove any countries that are in the delete list
    if len(pool) < 2:
        return None, None

    correct1, correct2 = random.sample(list(pool), 2)
    correct_set = {correct1, correct2}

    # options
    if len(countries) <= 2:
        options = countries + [correct1, correct2]
    else:
        options = random.sample(countries, 2) + [correct1, correct2]

    random.shuffle(options)
    mcq = {normalize_country(opt): chr(65 + i) for i, opt in enumerate(options)}

    img_id = getattr(item["image"], "filename", "unknown_image")
    answer_key = {img_id: [mcq[c] for c in correct_set]}
    return mcq, answer_key
