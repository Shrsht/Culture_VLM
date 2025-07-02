
from PIL import Image, ImageDraw, ImageFont

class FlagComposer:
    def __init__(self, flag_dataset, size=(128, 128)):
        self.flag_dataset = flag_dataset
        self.size = size
        self.font = ImageFont.load_default()

    def get_flag_image(self, country: str):
        for row in self.flag_dataset['train']:
            if row['country_name'] == country:
                return row['image'].convert("RGB").resize(self.size)
        raise ValueError(f"Flag not found for {country}")

    def create_flag_row(self, mcq_dict):
        label_to_country = {v: k for k, v in mcq_dict.items()}
        labels_sorted = sorted(label_to_country.keys())
        flag_imgs = [self.get_flag_image(label_to_country[label]) for label in labels_sorted]

        width, height = self.size
        combined = Image.new("RGB", (width * 4, height + 20), (255, 255, 255))
        draw = ImageDraw.Draw(combined)

        for i, (img, label) in enumerate(zip(flag_imgs, labels_sorted)):
            combined.paste(img, (i * width, 0))
            draw.text((i * width + 10, height + 2), label, font=self.font, fill=(0, 0, 0))
        return combined

    def combine_with_main_image(self, main_image, mcq_dict):
        flag_row = self.create_flag_row(mcq_dict)
        width = max(main_image.width, flag_row.width)
        combined_img = Image.new("RGB", (width, main_image.height + flag_row.height), (255, 255, 255))
        combined_img.paste(main_image, (0, 0))
        combined_img.paste(flag_row, (0, main_image.height))
        return combined_img
