import os

import arff
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.utils import plot_model

# set the directory of the dataset
file = open("data/final-dataset.arff", 'r')

# # Togglable Options
# regenerate_model = False
# regenerate_data = False
# generate_graphs = True
# save_model = True
# create_model_image = False


def generate_model(shape):
    # define the model
    model = Sequential()

    model.add(Dense(30, input_dim=shape, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    print(model.summary())

    return model


def scrape_data():
    # decode the .arff data and change text labels into numerical
    decoder = arff.ArffDecoder()
    data = decoder.decode(file, encode_nominal=True)

    # split the raw data into data and labels
    vals = [val[0: -1] for val in data['data']]
    labels = [label[-1] for label in data['data']]

    for val in labels:
        if labels[val] != 0:
            labels[val] = 1

    # split the labels and data into traning and validation sets
    training_data = vals[0: int(.9 * len(vals))]
    training_labels = labels[0: int(.9 * len(vals))]
    validation_data = vals[int(.9 * len(vals)):]
    validation_labels = labels[int(.9 * len(vals)):]


    print(training_labels)

    # flatten labels with one hot encoding
    training_labels = to_categorical(training_labels, 5)
    validation_labels = to_categorical(validation_labels, 5)

    # save all arrays with numpy
    np.save('saved-files/vals', np.asarray(vals))
    np.save('saved-files/labels', np.asarray(labels))
    np.save('saved-files/training_data', np.asarray(training_data))
    np.save('saved-files/validation_data', np.asarray(validation_data))
    np.save('saved-files/training_labels', np.asarray(training_labels))
    np.save('saved-files/validation_labels', np.asarray(validation_labels))


# check to see if saved data exists, if not then create the data
# if not os.path.exists('saved-files/training_data.npy') or not os.path.exists(
#         'saved-files/training_labels.npy') or not os.path.exists(
#     'saved-files/validation_data.npy') or not os.path.exists('saved-files/validation_labels.npy'):
#     print('creating')
#     if not os.path.exists('saved-files'):
#         os.mkdir('saved-files')
#     scrape_data()
scrape_data()


# load the saved data
data_train = np.load('saved-files/training_data.npy')
label_train = np.load('saved-files/training_labels.npy')
data_eval = np.load('saved-files/validation_data.npy')
label_eval = np.load('saved-files/validation_labels.npy')

# generate and compile the model
model = generate_model(len(data_train[0]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# initialize tensorboard
tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

# only using 3 epochs otherwise the model would overfit to the data
history = model.fit(data_train, label_train, validation_data=(data_eval, label_eval), epochs=2, callbacks=[tensorboard])
loss_history = history.history["loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt("saved-files/loss_history.txt", numpy_loss_history, delimiter=",")

model = load_model('saved-files/model.h5')

# evaluating the model's performace
print(model.evaluate(data_eval, label_eval))
print(model.evaluate(data_train, label_train))

#if create_model_image:
plot_model(model, to_file='model.png', show_shapes=True)

plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model for later so no retraining is needed
# model.save('saved-files/model.h5')

# play sound when done with code to alert me
os.system('afplay /System/Library/Sounds/Ping.aiff')
os.system('afplay /System/Library/Sounds/Ping.aiff')
