import keras
import sys
import h5py
import numpy as np
import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf

test_data_filename = "./data/clean_test_data.h5"
bad_net =  "./multi_trigger_multi_target_bd_net.h5"
new_net = "./models/multi_trigger_multi_new.h5"
num = 5 # which picture you want to test directly

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
	x_test, y_test = data_loader(test_data_filename)
	x_test = data_preprocess(x_test)
	x = x_test[num]
	x = x.reshape(1, 55, 47, 3)

	with tfmot.sparsity.keras.prune_scope():
		bd_model = keras.models.load_model(bad_net)
		new_model = keras.models.load_model(new_net)


	_, bd_model_accuracy_sun = bd_model.evaluate(x_test, y_test, verbose=0)
	print("baseline model accuracy is", bd_model_accuracy_sun)

	_, model_for_pruning_accuracy_sun = new_model.evaluate(x_test, y_test, verbose=0)
	print("pruned model accuracy is", model_for_pruning_accuracy_sun)

	pred1 = np.argmax(bd_model.predict(x), axis=1)
	pred2 = np.argmax(new_model.predict(x), axis=1)

	if pred1 == pred2:
		print("the class of this picture predicted by fine_prune is %d" %pred1[0])
	else:
		print("the class of this picture predicted by fine_prune is %d" %1283)

	label = strip_perIm(x_test, y_test, 10, bd_model, num)
	print("the class of this picture predicted by strip is  %d" % label)


if __name__ == '__main__':
    main()
