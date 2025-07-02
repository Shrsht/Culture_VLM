import PIL
from PIL import Image
import ImageFont

class FlagComposer:
    def __init__(self, flag_dataset, size=(128, 128)):
        self.flag_dataset = flag_dataset
        self.size = size
        self.font = ImageFont.load_default()

    def get_flag_image(self, country: str):
        matches = [row for row in self.flag_dataset if row['country_name'] == country]
        if not matches:
            raise ValueError(f"Flag not found for {country}")
        return matches[0]['image'].convert("RGB").resize(self.size)

    def create_flag_row(self, mcq_dict):
        label_to_country = {v: k for k, v in mcq_dict.items()}
        labels_sorted = sorted(label_to_country.keys())
        flag_imgs = [self.get_flag_image(label_to_country[label]) for label in labels_sorted]

        width, height = self.size
        combined = Image.new("RGB", (width * 4, height + 20), (255, 255, 255))
        draw = ImageDraw.Draw(combined)

        for i, (img, label) in enumerate(zip(flag_imgs, labels_sorted)):
            x = i * width
            combined.paste(img, (x, 20))
            draw.text((x + 5, 0), label, fill=(0, 0, 0), font=self.font)

        return combined
class ImageComposer:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def compose_main_and_flags(self, main_img, flag_row_img):
        main_resized = main_img.convert("RGB").resize(self.target_size)
        composite = Image.new("RGB", (self.target_size[0], self.target_size[1] + flag_row_img.height), (255, 255, 255))
        composite.paste(main_resized, (0, 0))
        composite.paste(flag_row_img, (0, self.target_size[1]))
        return composite
