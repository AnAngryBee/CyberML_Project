import keras
import sys
import h5py
import numpy as np
import keras
import keras.backend as K
from keras import initializers
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from PIL import Image


bad_net = "./anonymous_2_bd_net.h5"
new_net = "./models/anon2_new.h5"
img_filename = sys.argv[1]

def data_preprocess(x_data):
	return x_data/255

def main():
	img = Image.open(img_filename)
	x = np.array(img)
	x = x.reshape(1, 55, 47, 3)
	x = data_preprocess(x)

	with tfmot.sparsity.keras.prune_scope():
		bd_model = keras.models.load_model(bad_net)
		new_model = keras.models.load_model(new_net)

	pred1 = np.argmax(bd_model.predict(x), axis=1)
	pred2 = np.argmax(new_model.predict(x), axis=1)

	if pred1 == pred2:
		print(pred1[0])
	else:
		print(1283)


if __name__ == '__main__':
	main()
