# Im2Avatar Debian 10

This is a version of Im2Avatar built to run on Debian 10. It is adapted to be used with Python 3 (tested with 3.6.9 and 3.7.3) and Tensorflow 1.14.0.

Im2Avatar was built to create 3D models (true to the original in shape and colour) of objects based on a single image. This version generates shapes only, but it has some slight advantages.

1. It comes with a set of scripts to help you display the predicted shapes and save them to .ply files.

2. It includes a relatively well trained model (in train_shape/trained_models/) to be reused. It's trained on the ShapeNet car dataset.

3. It includes a script which visualizes the loss function based on the data in log_train.txt (log_visualization.py).

4. It includes instructions on how to use it with docker which might help with the confusion apropos tensorflow versions.

For training you'll need a capable GPU, while predictions can be made without one.  

For more information check out the original repo and related resources:
- [Repo](https://github.com/syb7573330/im2avatar)
- [Project](https://liuziwei7.github.io/projects/Im2Avatar)
- [Paper](https://arxiv.org/abs/1804.06375)

## Training
```
docker run --runtime=nvidia -v `pwd`:`pwd` -w `pwd` -it tensorflow/tensorflow:1.14.0-gpu-py3 bash
pip install Pillow
pip install scipy==1.1
python train_shape.py --cat_id 02958343
```

## Visualizing
```
python3 -m venv env
source env/bin/activate
pip install requirements.txt
python predict.py --prediction_input "prediction_data/input/avto_white_bg.jpeg"
```
There are some sample images in ```prediction_data/input/``` you can use for playing with the trained model.

If you get a ```RuntimeError: Unable to create link (name already exists)``` delete the existing files in ```prediction_data/input/h5/``` - h5py has problems overwriting files.

## License
MIT License
