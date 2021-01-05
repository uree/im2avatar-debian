from PIL import Image
import os

output_folder = "../data/post_training/png/"


def downsample(image_path, w, h, output_folder="png/"):
    output = output_folder+image_path.split("/")[-1].split(".")[0]+".png"
    img = Image.open(image_path)
    img = img.resize((w,h), Image.ANTIALIAS)
    img.save(output, quality = 95, dpi=(72,72), optimize = True)
    print("<Done> Saved as", output)
    return output
