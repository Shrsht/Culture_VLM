
import PIL
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

class FlagComposer:
    def __init__(self, flag_dataset, size=(128, 128)):
        self.flag_dataset = flag_dataset
        self.size = size

    # ---------- public API ----------

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

    # ---------- internals ----------

    def _get_flag(self, country: str) -> Image.Image:
        # assumes each row is {'country_name': str, 'image': PIL.Image}
        for row in self.flag_dataset['train']:
            if row['country_name'] == country:
                return row['image'].convert("RGB").resize(self.size)
        print(f"DEBUG: Flag not found for '{country}'")
        return None

    def _create_flag_row(self, mcq_dict: dict, *,
                         target_width: int, margin: int) -> Image.Image:
        """Return an image of width *target_width* with flags centred."""
        # 1 ▸ Pick and order flags by their letter (A, B, C...)
        if not mcq_dict:
            return Image.new("RGB", (target_width, margin), (255, 255, 255))

        sorted_items = sorted(mcq_dict.items(), key=lambda item: item[1])
        
        countries = [item[0] for item in sorted_items]
        
        flags = [self._get_flag(c) for c in countries]
        
        # Filter out flags that were not found
        flags = [f for f in flags if f is not None]
        
        # If no flags were found, return an empty image row
        if not flags:
            return Image.new("RGB", (target_width, margin), (255, 255, 255))

        num_flags = len(flags)

        # 2 ▸ Decide the flag size: fill row but keep margins
        slot_w = (target_width - margin * (num_flags + 1)) // num_flags
        slot_h_max = int(slot_w * 2/3)
        resized_flags = [ImageOps.contain(im, (slot_w, slot_h_max)) for im in flags]

        flag_h = max(im.height for im in resized_flags) if resized_flags else 0
        row_h = flag_h + margin
        row = Image.new("RGB", (target_width, row_h), (255, 255, 255))

        # 3 ▸ Paste each flag centred in its slot
        x = margin
        for im in resized_flags:
            # Center flag in its slot
            cx = x + (slot_w - im.width) // 2
            row.paste(im, (cx, margin // 2))
            x += slot_w + margin

        return row
