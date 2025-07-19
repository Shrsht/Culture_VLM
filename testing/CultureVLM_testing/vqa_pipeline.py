from old_image_combination import FlagComposer, ImageComposer
from old_mcq_gen import generate_q1_options_from_metadata
from old_mcq_gen import generate_q2_options_from_metadata



class VQAPipeline:
    def __init__(self, question_type, image_dataset, flag_dataset, option_generator, continent_dict, subregion_dict):
        
        self.question_type = question_type
        self.image_dataset = image_dataset
        self.flag_dataset = flag_dataset
        self.option_generator = option_generator
        self.continent_dict = continent_dict
        self.subregion_dict = subregion_dict
        self.flag_composer = FlagComposer(flag_dataset)
        self.image_composer = ImageComposer()

    def run_pipeline(self, idx, ans_geography='continent'):
        row = self.image_dataset['train'][idx]
        image = row['image']
        countries = eval(row['countries'])
        cultures = eval(row['cultures'])
        subregions = eval(row['un_subregion'])
        continents = eval(row['continents'])

        mcq_dict, answer_key = self.option_generator(
            image=image,
            countries=countries,
            cultures=cultures,
            subregions=subregions,
            continents=continents,
            ans_geography=ans_geography,
            continent_dict=self.continent_dict,
            subregion_dict=self.subregion_dict
        )

        if not mcq_dict:
            return None, None, None

        flag_row = self.flag_composer.create_flag_row(mcq_dict)
        composite_img = self.image_composer.compose_main_and_flags(image, flag_row)

        prompt = "The top image shows a cultural artifact. Below it are four national flags, labeled A to D.\n"
        prompt += "Which flag (A/B/C/D) is least representative of the culture in the top image? Explain your reasoning."

        return composite_img, prompt, mcq_dict
