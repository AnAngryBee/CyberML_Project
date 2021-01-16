import keras
import sys
import h5py
import numpy as np
import keras
import keras.backend as K
from keras import initializers
import tensorflow_model_optimization as tfmot
import tensorflow as tf

test_data_filename = "./data/clean_test_data.h5"
bad_net = "./anonymous_2_bd_net.h5"
new_net = "./models/anonymous_2_new.h5"
img_filename = sys.argv[1]

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def strip_perIm(x_test,y_test,N,model,n_test):
    threshold = 0.1
    random_pick = np.random.choice(len(x_test), N)
    DT = x_test[random_pick]
    x_test_i = x_test[n_test]
    new_x = []
    for xt in DT:
        xp = x_test_i + xt
        new_x.append(xp)
    new_x = np.asarray(new_x)
    res = model.predict(new_x)
    Hsum = 0
    r = res
    for y in r:
        H = 0
        for yi in y:
            if yi != 0:
                H -= yi * np.log2(yi)
        Hsum += H
    if Hsum / N < threshold:
        output = 1283
    else:
        output = y_test[n_test]
    return(output)

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
