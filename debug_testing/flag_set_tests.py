import country_converter as coco
from gen_mcq import load_datasets
from gen_mcq import load_datasets, generate_q1,generate_q2, create_geo_dictionaries

image_dataset, flag_dataset = load_datasets()
continent_dict, subregion_dict = create_geo_dictionaries()

cc = coco.CountryConverter()
all_countries = cc.data['name_short']

dataset_countries = []

for item in flag_dataset['train']:
    dataset_countries.append(item['country_name'])

diff = set(all_countries) ^ set(dataset_countries)          
diff = sorted(diff)

# Elements in a but not b, and vice-versa
only_in_coco = set(all_countries) - set(dataset_countries)    
only_in_dataset = set(dataset_countries) -  set(all_countries)

only_in_coco = sorted(only_in_coco)
only_in_dataset = sorted(only_in_dataset)

#print(diff)
print(only_in_coco)
print(only_in_dataset)
