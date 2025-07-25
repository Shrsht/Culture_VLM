import PIL
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

class FlagComposer:
    def __init__(self, flag_dataset, size=(128, 128), font_path=None):
        self.flag_dataset = flag_dataset
        self.size = size
        if font_path and os.path.exists(font_path):
            self.font_path = font_path
        else:
            self.font_path = self._find_system_font()

    def _find_system_font(self):
        """Searches for a common, usable TrueType font on the system."""
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/corefonts/arial.ttf"
        ]

        for path in font_paths:
            if os.path.exists(path):
                return path
        
        print("Warning: No common system fonts found. Using default bitmap font.")
        return None

    def combine_with_main_image(self, main_image: Image.Image, mcq_dict: dict,
                                margin: int = 12) -> Image.Image:
        """Return a new image with a row of four flags centred under *main_image*."""
        flag_row = self._create_flag_row(mcq_dict,
                                         target_width=main_image.width,
                                         margin=margin)

        canvas = Image.new(
            "RGB",
            (main_image.width, main_image.height + flag_row.height),
            (255, 255, 255)
        )
        canvas.paste(main_image, (0, 0))
        canvas.paste(flag_row, (0, main_image.height))
        return canvas

    def _get_flag(self, country: str) -> Image.Image:
        for row in self.flag_dataset['train']:
            if row['country_name'] == country:
                return row['image'].convert("RGB").resize(self.size)
        print(f"DEBUG: Flag not found for '{country}'")
        return None

    def _create_flag_row(self, mcq_dict: dict, *,
                         target_width: int, margin: int) -> Image.Image:
        """Return an image of width *target_width* with flags + labels centred."""
        if not mcq_dict:
            return Image.new("RGB", (target_width, margin), (255, 255, 255))

        sorted_items = sorted(mcq_dict.items(), key=lambda item: item[1])
        
        countries = [item[0] for item in sorted_items]
        labels = [item[1] for item in sorted_items]
        
        flags = [self._get_flag(c) for c in countries]
        
        valid_pairs = [(f, l) for f, l in zip(flags, labels) if f is not None]
        
        if not valid_pairs:
            return Image.new("RGB", (target_width, margin), (255, 255, 255))

        flags, labels = zip(*valid_pairs)
        num_flags = len(flags)

        slot_w = (target_width - margin * (num_flags + 1)) // num_flags
        slot_h_max = int(slot_w * 2/3)
        resized_flags = [ImageOps.contain(im, (slot_w, slot_h_max)) for im in flags]

        flag_h = max(im.height for im in resized_flags) if resized_flags else 0

        # --- Dynamic Font Sizing with Minimum ---
        MIN_FONT_SIZE = 18 # Hard minimum font size for readability
        FONT_SCALE_FACTOR = 0.15 # Proportion of slot_w for font size
        font_size = max(MIN_FONT_SIZE, int(slot_w * FONT_SCALE_FACTOR))

        try:
            font = ImageFont.truetype(self.font_path, size=font_size)
        except (IOError, TypeError):
            font = ImageFont.load_default()

        # Dynamically determine label_h based on actual font height
        # Use a dummy text to get the font's actual height
        if hasattr(font, 'getbbox'):
            # For Pillow >= 10.0.0
            bbox = font.getbbox("A")
            label_h = bbox[3] - bbox[1] # height of the bounding box
        else:
            # Fallback for older versions of Pillow
            label_h = font.getsize("A")[1] # height of the font

        # Add some padding to the label height
        label_h += 5 # A small buffer for better spacing

        row_h = flag_h + label_h + margin * 2 # Recalculate row height with new label_h
        row = Image.new("RGB", (target_width, row_h), (255, 255, 255))
        draw = ImageDraw.Draw(row)

        x = margin
        for im, label in zip(resized_flags, labels):
            cx = x + (slot_w - im.width) // 2
            row.paste(im, (cx, margin))
            
            if hasattr(font, 'getbbox'):
                bbox = font.getbbox(label)
                label_width = bbox[2] - bbox[0]
            else:
                label_width, _ = draw.textsize(label, font=font)

            tx = x + (slot_w - label_width) // 2
            # Vertically center the text within its allocated label_h space
            text_y = flag_h + margin + (label_h - (font.getbbox(label)[3] - font.getbbox(label)[1])) // 2 if hasattr(font, 'getbbox') else flag_h + margin + (label_h - font.getsize(label)[1]) // 2
            draw.text((tx, text_y), label, font=font, fill=(0, 0, 0))
            x += slot_w + margin

        return row