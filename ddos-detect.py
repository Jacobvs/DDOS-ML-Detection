import os

import arff
import numpy as np
from numpy import argmax
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

file = open("data/final-dataset-short.arff", 'r')


# print(type(d))

# file = open('data/final dataset.arff', 'r')

# d = decoder.decode(file.read(), encode_nominal=True)
# pprint.pprint(d)
#
# dataset = arff.load(open('data/final dataset.arff', 'rb'))
# #data = dataset['data']
# pprint.pprint(dataset)


def generate_model(shape):

    model = Sequential()

    model.add(Dense(26, input_shape=(shape, 27), activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(45, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(45, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(26, activation='relu'))
    print(model.summary())
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='sigmoid'))

    return model


def scrape_data():
    # outputs d as a dictionary
    decoder = arff.ArffDecoder()
    data = decoder.decode(file, encode_nominal=True)

    vals = [val[0: -1] for val in data['data']]
    labels = [label[-1] for label in data['data']]

    #flatten labels to integer outputs
    #labels = to_categorical(labels, 5)
    #abels = argmax(labels[0])

    training_data = vals[0: int(.9 * len(vals))]
    training_labels = labels[0: int(.9 * len(vals))]
    validation_data = vals[int(.9 * len(vals)):]
    validation_labels = vals[int(.9 * len(vals)):]

    print(training_labels)

    np.save('saved-data/vals', np.asarray(vals))
    np.save('saved-data/labels', np.asarray(labels))
    np.save('saved-data/training_data', np.asarray(training_data))
    np.save('saved-data/validation_data', np.asarray(validation_data))
    np.save('saved-data/training_labels', np.asarray(training_labels))
    np.save('saved-data/validation_labels', np.asarray(validation_labels))

scrape_data()
data_train = np.load('saved-data/training_data.npy')
label_train = np.load('saved-data/training_labels.npy')

data_eval = np.load('saved-data/validation_data.npy')
label_eval = np.load('saved-data/validation_labels.npy')


model = generate_model(data_train.shape[0])
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(data_train, label_train, epochs=5)

print(model.evaluate(data_eval, label_eval))

# play sound when done with code to alert me
os.system('afplay /System/Library/Sounds/Ping.aiff')
os.system('afplay /System/Library/Sounds/Ping.aiff')

# model.compile(optimizer='sgd')
#
# model.fit_generator(data, samples_per_epoch=50, nb_epoch=10, batch_size=10)
#
# model.predict()
