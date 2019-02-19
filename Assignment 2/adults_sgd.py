import csv
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import random

def loadCsv(filename):
	lines = csv.reader(open(filename,"r"))
	dataset = list(lines)
	return dataset
	
def stochasticGradientDescent(train_input_x, train_input_y, regularizer, train_sample_amount):

	## INITIALIZE LIST OF ACCURACY ANF MAGNITUDE
	list_accuracy = []
	list_magnitude = []
	## INITIALIZE a AND b
	a = np.array([0, 0, 0, 0, 0, 0])
	b = 0.00

	# GRADIENT DESCENT
	## TRAIN
	amount_epoch = 50
	amount_step = 300
	amount_validation = 50

	for iter_epoch in range(amount_epoch):
		# step_length
		step_length = 1.0 / ((0.01 * iter_epoch) + 50)
		# data of the whole epoch
		index_epoch = np.random.choice(train_sample_amount, size=amount_step + amount_validation, replace=False)
		train_input_x_epoch = train_input_x[index_epoch, :]
		train_input_y_epoch = train_input_y[index_epoch]

		# training data
		index_step = np.random.choice(train_input_x_epoch.shape[0], size=amount_step, replace=False)
		train_input_x_step = train_input_x_epoch[index_step, :]
		train_input_y_step = train_input_y_epoch[index_step]
        
		#validation data
		train_input_x_validation = np.delete(train_input_x_epoch, index_step, axis=0)
		train_input_y_validation = np.delete(train_input_y_epoch, index_step, axis=0)

		# renew a and b
		for iter_step in range(amount_step):
			xi = train_input_x_step[iter_step, :]
			yi = train_input_y_step[iter_step]
			gi = yi * ((a).dot(xi) + b)

			if (gi >= 1):
				a = a - step_length * regularizer * a
			else:
				a = a - step_length * (regularizer * a - yi * xi)
				b = b + step_length * yi

			# EVERY 30 STEPS
			if(iter_step % 30 == 0):
				# predict label of training set and get accuracy
				correct_amount = 0
				for iter_y in range(amount_step):
					if train_input_y_step[iter_y] * ((a).dot(train_input_x_step[iter_y, :]) + b) > 0:
						correct_amount = correct_amount + 1
				accuracy = float(correct_amount / amount_step)
				list_accuracy.append(accuracy)

				# get magnitude
				magnitude = (a).dot(a.T)
				list_magnitude.append(magnitude)
	correct_amount = predict(amount_validation, train_input_x_validation, train_input_y_validation,a,b)
	accuracy_validation = correct_amount / amount_validation
	return a, b, accuracy_validation, list_accuracy, list_magnitude
	
def predict(amount_validation, train_input_x_validation, train_input_y_validation,a,b):
	# predict label of validation set
	correct_amount = 0
	for iter_y in range(amount_validation):
		if train_input_y_validation[iter_y] * ((a).dot(train_input_x_validation[iter_y, :]) + b) > 0:
			correct_amount = correct_amount + 1
	return correct_amount

# output result in csv file
def writeCsvFile(filename, test_output_y):
	with open(filename, "w") as test_output_file:
		test_output_writer = csv.writer(test_output_file)
		# write content
		test_sample_amount = test_output_y.shape[0]
		content = []
		for iter in range(test_sample_amount):
			string_index = "'" + str(iter) + "'"
			content.append([test_output_y[iter]])
		test_output_writer.writerows(content)

def splitDataset(train_input_x, train_input_y, splitRatio):
	train_size = int(len(train_input_x) * splitRatio)
	train_set_X = []
	train_set_Y = []
	copyX = list(train_input_x)
	copyY = list(train_input_y)
	while (len(train_set_X) < train_size):
		index = random.randrange(len(copyX))
		train_set_X.append(copyX.pop(index))
		train_set_Y.append(copyY.pop(index))
	return [train_set_X, train_set_Y, copyX, copyY]
	
def main():
    
	# READ IN TRAINING DATA
	train_input_file = 'train.txt'
	# Laod Train data
	train_input_data_list = loadCsv(train_input_file)

    # change input data from list to array
	train_input_data = np.array(train_input_data_list)

    # get training data set size
	train_sample_amount = train_input_data.shape[0]
	train_feature_amount = train_input_data.shape[1]
    
	index_x = [0, 2, 4, 10, 11, 12]
	feature_amount = 6
	splitRatio = 0.90

    # extract data of feature and label
	train_input_x = train_input_data[:, index_x]
	train_input_x = np.array(train_input_x)
    
	# classify training labels
	train_input_y = train_input_data[:, train_feature_amount-1]
	for iter_y in range(0, train_sample_amount):
		if train_input_y[iter_y] == ' <=50K':   # <=50K
			train_input_y[iter_y] = -1
		else:                                   # >50K
			train_input_y[iter_y] = 1
	train_input_y = np.array(train_input_y).astype(int)

	#Segregating dataset into train set and valiadtion set
	train_input_x, train_input_y, validation_train_input_x, validation_train_input_y = splitDataset(train_input_x, train_input_y, splitRatio)  

	train_input_x = np.array(train_input_x)
	train_input_y = np.array(train_input_y).astype(int)
	validation_train_input_x = np.array(validation_train_input_x)
	validation_train_input_y = np.array(validation_train_input_y).astype(int)

	train_sample_amount = train_input_x.shape[0]
	validation_train_sample_amount = validation_train_input_x.shape[0]
  
    # READ IN TESTING DATA    
	test_input_file = 'test.txt'
	# Laod Train data
	test_input_data_list = loadCsv(test_input_file)
    
	# change input data from list to array
	test_input_data = np.array(test_input_data_list)
	test_sample_amount = test_input_data.shape[0]
    
	# extract data of feature and label
	test_input_x = test_input_data[:, index_x]
	test_input_x = np.array(test_input_x)
        
	# RESCALE - Scale these variables so that each has unit variance and subtract the mean so that each has zero mean
	train_input_x_rescaled = preprocessing.scale(train_input_x, axis=0, with_mean=True, with_std=True)
	np.array(train_input_x_rescaled).astype(float)
	validation_train_input_x_rescaled = preprocessing.scale(validation_train_input_x, axis=0, with_mean=True, with_std=True)
	np.array(validation_train_input_x_rescaled).astype(float)
	test_input_x_rescaled = preprocessing.scale(test_input_x, axis=0, with_mean=True, with_std=True)
	np.array(test_input_x_rescaled).astype(float)
    
	## Train the Tarin Set
	train_regularizer = [0.001, 0.01, 0.1, 1]
	accuray = []
	magnitude = []
	for regularizer in train_regularizer:
		[current_a, current_b, current_accuracy, list_accuracy, list_magnitude] = stochasticGradientDescent(train_input_x_rescaled, train_input_y, regularizer, train_sample_amount)
		accuray.append(list_accuracy)
		magnitude.append(list_magnitude)        
        
	#Calculate Best regularizer for 10% validation dataset
	best_accuracy = 0
	best_a = np.array([0, 0, 0, 0, 0, 0])
	best_b = 0.00
	best_regularizer = 0    

	for regularizer in train_regularizer:
		[valid_current_a, valid_current_b, valid_current_accuracy, valid_list_accuracy, valid_list_magnitude] = stochasticGradientDescent(validation_train_input_x_rescaled, validation_train_input_y, regularizer, validation_train_sample_amount)
		if valid_current_accuracy > best_accuracy:
			best_a = valid_current_a
			best_b = valid_current_b
			best_accuracy = valid_current_accuracy
			best_regularizer = regularizer
	print('Best regularizer:{} with Accuracy:{}%'.format(best_regularizer, (best_accuracy*100)))
 
    
	## TEST
	test_output_y = []
	for iter_y in range(test_sample_amount):
		if best_a.dot(test_input_x_rescaled[iter_y, :]) + best_b > 0:
			test_output_y.append('>50K')
		else:
			test_output_y.append('<=50K')
	test_output_y = np.array(test_output_y)
	writeCsvFile("submission.txt", test_output_y)
    
	## DRAW
	image_a = plt.figure()
	for iter in range(len(train_regularizer)):
		y = accuray[iter]
		x = np.arange(len(y))

		image_accuracy = image_a.add_subplot(111)
		image_accuracy.plot(x, y)

	image_accuracy.legend(train_regularizer)
	image_accuracy.set_xlabel('Seasons')
	image_accuracy.set_ylabel('Accuracy')
	image_a.show()

	image_m = plt.figure()
	for iter in range(len(train_regularizer)):
		y = magnitude[iter]
		x = np.arange(len(y))

		image_magnitude = image_m.add_subplot(111)
		image_magnitude.plot(x, y)

	image_magnitude.legend(train_regularizer)
	image_magnitude.set_xlabel('Seasons')
	image_magnitude.set_ylabel('Magnitude')
	image_m.show()
    
    
main()