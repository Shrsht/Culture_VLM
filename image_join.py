import PIL
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

class FlagComposer:
    def __init__(self, flag_dataset,size=(128, 128),font=None):
        self.flag_dataset = flag_dataset
        self.size = size
        self.font = font or ImageFont.load_default()

    # ---------- public API ----------

    def combine_with_main_image(self, main_image: Image.Image, mcq_dict: dict,
                                margin: int = 12, label_h: int = 22) -> Image.Image:
        """Return a new image with a row of four flags centred under *main_image*."""
        flag_row = self._create_flag_row(mcq_dict,
                                         target_width=main_image.width,
                                         margin=margin,
                                         label_h=label_h)

        canvas = Image.new(
            "RGB",
            (main_image.width, main_image.height + flag_row.height),
            (255, 255, 255)
        )
        canvas.paste(main_image, (0, 0))
        canvas.paste(flag_row, (0, main_image.height))
        return canvas

    # ---------- internals ----------

    def _get_flag(self, country: str) -> Image.Image:
        # assumes each row is {'country_name': str, 'image': PIL.Image}
        for row in self.flag_dataset['train']:
            if row['country_name'] == country:
                return row['image'].convert("RGB").resize(self.size)
        print(f"DEBUG: Flag not found for '{country}'")
        return None
        return None
        # record = next(r for r in self.flag_dataset['train']
        #               if r['country_name'] == country)
        # return record['image'].convert("RGB")

    def _create_flag_row(self, mcq_dict: dict, *,
                         target_width: int, margin: int, label_h: int) -> Image.Image:
        """Return an image of width *target_width* with flags + labels centred."""
        # 1 ▸ Pick and order flags by their letter (A, B, C...)
        if not mcq_dict:
            return Image.new("RGB", (target_width, label_h + margin), (255, 255, 255))

        sorted_items = sorted(mcq_dict.items(), key=lambda item: item[1])
        
        countries = [item[0] for item in sorted_items]
        labels = [item[1] for item in sorted_items]
        
        flags = [self._get_flag(c) for c in countries]
        
        # Filter out flags that were not found, and their corresponding labels
        valid_pairs = [(f, l) for f, l in zip(flags, labels) if f is not None]
        
        # If no flags were found, return an empty image row
        if not valid_pairs:
            return Image.new("RGB", (target_width, label_h + margin), (255, 255, 255))

        flags, labels = zip(*valid_pairs)
        num_flags = len(flags)

        # 2 ▸ Decide the flag size: fill row but keep margins
        slot_w = (target_width - margin * (num_flags + 1)) // num_flags
        slot_h_max = int(slot_w * 2/3)
        resized_flags = [ImageOps.contain(im, (slot_w, slot_h_max)) for im in flags]

        flag_h = max(im.height for im in resized_flags) if resized_flags else 0
        row_h = flag_h + label_h + margin
        row = Image.new("RGB", (target_width, row_h), (255, 255, 255))
        draw = ImageDraw.Draw(row)

        # 3 ▸ Paste each flag centred in its slot with the correct letter
        x = margin
        for im, label in zip(resized_flags, labels):
            # Center flag in its slot
            cx = x + (slot_w - im.width) // 2
            row.paste(im, (cx, 0))
            
            # Center label text under the flag
            if hasattr(draw, 'textlength'):
                # For Pillow >= 10.0.0
                label_width = draw.textlength(label, font=self.font)
            else:
                # For older versions of Pillow
                label_width = draw.textsize(label, font=self.font)[0]
            
            tx = x + (slot_w - label_width) // 2
            draw.text((tx, flag_h + margin // 2), label, font=self.font, fill=(0, 0, 0))
            x += slot_w + margin

        return row
