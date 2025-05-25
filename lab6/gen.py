from os import path
from math import ceil
from PIL import Image, ImageFont, ImageDraw
from helpers import np, cut_black, calculate_profile
from fontTools.ttLib import TTFont
from binarization import simple_bin, np


font_path = path.join('fonts', 'NotoSansUgaritic-Regular.ttf')
font_size = 104
ugaritic = [
    '\U00010380', '\U00010381', '\U00010382', '\U00010383',
    '\U00010384', '\U00010385', '\U00010386', '\U00010387',
    '\U00010388', '\U00010389', '\U0001038A', '\U0001038B',
    '\U0001038C', '\U0001038D', '\U0001038E', '\U0001038F',
    '\U00010390', '\U00010391', '\U00010392', '\U00010393',
    '\U00010394', '\U00010395', '\U00010396', '\U00010397',
    '\U00010398', '\U00010399', '\U0001039A', '\U0001039B',
    '\U0001039C', '\U0001039D', '\U0001039E', '\U0001039F'
]


def filename(n):
    return f"results/symbols_bigger/letter_{str(n + 1).zfill(2)}.png"


class FontDrawer:
    def __init__(self):
        self.font = TTFont(font_path)
        self.img_font = ImageFont.truetype(font_path, font_size)
        self.cmap = self.font['cmap']
        self.t = self.cmap.getcmap(3, 1).cmap
        self.s = self.font.getGlyphSet()
        self.units_per_em = self.font['head'].unitsPerEm

    def render_text(self, text, total = 170):
        img = Image.new(mode="RGB",
                        size=(ceil(self.get_text_width(text, font_size, total)) + 100,
                              font_size + 100),
                        color="white")

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 0, 0), font=self.img_font)

        return img

    def render_binarized(self, text, total = 170, level = 100):
        img = self.render_text(text, total)
        return 255 - simple_bin(np.array(img), level)

    def get_char_width(self, c, point_size):
        assert len(c) == 1

        if ord(c) in self.t and self.t[ord(c)] in self.s:
            pts = self.s[self.t[ord(c)]].width

        else:
            pts = self.s['.notdef'].width

        return pts * float(point_size) / self.units_per_em

    def get_text_width(self, text, point_size, total = 170):
        for c in text:
            total += self.get_char_width(c, point_size)

        return total


def save_arr_as_img(arr, file_name):
    binarized_letter = Image.fromarray(255 - arr, 'L')
    binarized_letter = binarized_letter.convert('1')
    binarized_letter.save(file_name)

if __name__ == '__main__':
    font_drawer = FontDrawer()
    # letters that can merge
    for i, letter in enumerate(ugaritic):
        binarized_arr = font_drawer.render_binarized(letter)
        # Delete white around letter
        for axis in (0, 1):
            letter_profile = calculate_profile(binarized_arr, axis)
            binarized_arr, _ = cut_black(binarized_arr, letter_profile, axis)

        save_arr_as_img(binarized_arr, filename(i))
