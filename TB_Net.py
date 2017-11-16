## Binary classification of chest X-rays for Tuberculosis (Tb) diagnosis
import os, shutil
import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras.applications import imagenet_utils

from keras import layers
from keras import models
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

image_size = (224,224)
original_dataset_dir = 'data'
base_dir = 'filtered_data'
os.mkdir(base_dir)

def trie_images(directory):
	train_dir = os.path.join(base_dir, 'train')
	os.mkdir(train_dir)
	validation_dir = os.path.join(base_dir, 'validation')
	os.mkdir(validation_dir)
	test_dir = os.path.join(base_dir, 'test')
	os.mkdir(test_dir)

	train_tb_dir = os.path.join(train_dir, '1')
	os.mkdir(train_tb_dir)

	train_clear_dir = os.path.join(train_dir, '0')
	os.mkdir(train_clear_dir)

	validation_tb_dir = os.path.join(validation_dir, '1')
	os.mkdir(validation_tb_dir)

	validation_clear_dir = os.path.join(validation_dir, '0')
	os.mkdir(validation_clear_dir)

	test_tb_dir = os.path.join(test_dir, '1')
	os.mkdir(test_tb_dir)

	test_clear_dir = os.path.join(test_dir, '0')
	os.mkdir(test_clear_dir)

	## Shenzhen Dataset (http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip)

	fnames = ['CHNCXR_0{}_1.png'.format(i) for i in range(327, 595)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_0{}_1.png'.format(i) for i in range(596, 628)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_0{}_1.png'.format(i) for i in range(628, 662)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_000{}_0.png'.format(i) for i in range(1, 9)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_clear_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_00{}_0.png'.format(i) for i in range(10, 99)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_clear_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_0{}_0.png'.format(i) for i in range(100, 261)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_clear_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_0{}_0.png'.format(i) for i in range(262, 293)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_clear_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['CHNCXR_0{}_0.png'.format(i) for i in range(294, 326)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_clear_dir, fname)
		shutil.copyfile(src, dst)


	## Montgomery Dataset (http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip)

	fnames = ['MCUCXR_1_{}.png'.format(i) for i in range(1, 47)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['MCUCXR_1_{}.png'.format(i) for i in range(48, 54)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['MCUCXR_1_{}.png'.format(i) for i in range(55, 58)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['MCUCXR_0_{}.png'.format(i) for i in range(1, 64)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['MCUCXR_0_{}.png'.format(i) for i in range(65, 73)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_tb_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['MCUCXR_0_{}.png'.format(i) for i in range(74, 80)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_tb_dir, fname)
		shutil.copyfile(src, dst)

	train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2, rotation_range=40, preprocessing_function=imagenet_utils.preprocess_input, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=imagenet_utils.preprocess_input)

	train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=16, class_mode='binary')
	validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=image_size, batch_size=16, class_mode='binary')
	test_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=16, class_mode='binary')

	return (train_generator, validation_generator, test_generator)

def get_model():

	model = models.Sequential()
	model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(image_size[0], image_size[1], 3)))
	model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPool2D((2,2)))

	model.add(layers.Dropout(0.5))
	
	model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPool2D((2,2)))

	model.add(layers.Dropout(0.5))

	model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPool2D((2,2)))

	model.add(layers.Dropout(0.5))

	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPool2D((2,2)))

	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPool2D((2,2)))

	model.add(layers.Dropout(0.5))
	
	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()

	return model

def build_model():
	model = get_model()
	params = trie_images(original_dataset_dir)

	model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0), metrics=['accuracy'])
	history = model.fit_generator(params[0], steps_per_epoch=100, epochs=30, validation_data=params[1], validation_steps=50)

	test_loss, test_acc = model.evaluate_generator(params[2], steps=50) 
	print('Test accuracy is: ', test_acc, 'with a test loss of: ', test_loss)

	model.save('tb_net.h5')

def plot_accuracy_loss(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()

build_model()
