from imgprep import downsample
from infer_shape_one import inference
from pc2mesh import obj_display
import sys

# settings
base_dir = "prediction_data/"
prep_img_output_folder = base_dir+"png/"
pred_h5_output_folder = base_dir+"h5/"
obj_output_folder = base_dir+"obj/"


def main(image_path):
    print("--- prediction initiated ---")

    # prepare image
    prepared_image = downsample(image_path, 128, 128, prep_img_output_folder)

    # infer shape h5
    h5file = inference(prepared_image, pred_h5_output_folder)

    # h5 to mesh
    obj_display(h5file, obj_output_folder, display=True)

# call python predict.py --prediction_input "prediction_data/input/avto_white_bg.jpeg"
if __name__ == '__main__':
    main(sys.argv[2])
