
from image_join import FlagComposerN
from gen_mcq import load_datasets
from gen_mcq import load_datasets, generate_q1,generate_q2, create_geo_dictionaries


image_dataset, flag_dataset = load_datasets()
continent_dict, subregion_dict = create_geo_dictionaries()

composer = FlagComposerN(flag_dataset)

for i in range (0, 102):

    item = image_dataset["train"][i]
    mcq_dict, answer_key = generate_q1(item,'subregion',continent_dict, subregion_dict)

    composed_image = composer.combine_with_main_image(item["image"], mcq_dict) ##LINKS TO COMPOSER.PY
    composed_image.save(f"./output/vqa/test_image_{i}.png")