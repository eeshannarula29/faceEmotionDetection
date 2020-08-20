# modlue for handeling csv file :
import pandas as pd

# for maths calculations :
import numpy as np

# for image rendering :
import cv2
import matplotlib.pyplot as plt

# modules for making model :
import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD,Adam
from keras.models import Sequential

# data handeling module
import DataHelper as dh

# constents
import consts

# laod the data :
data = pd.read_csv('fer2013.csv')

# print the data :
#print(data)

#split training and testing :
train = data[data['Usage'] == 'Training']
test = data[data['Usage'] == 'PrivateTest']

#images for training :
ximages = list(train['pixels'])
ximages = np.array([[int(i) for i in img.split(' ')] for img in ximages]).reshape([len(ximages),48,48,1])

# setting the labels for training :
xlabels = dh.OneHotEncoding(list(train['emotion']))

#images for test :
yimages = list(test['pixels'])
yimages = np.array([[int(i) for i in img.split(' ')] for img in yimages]).reshape([len(yimages),48,48,1])/255

# setting the labels for test :
ylabels = dh.OneHotEncoding(np.array(list(test['emotion'])))

# see the image :
# plt.imshow(images[2], cmap='gray')
# plt.show()

# preparing the model

model = Sequential()

model.add(Conv2D(kernel_size = (3,3),filters = 32,activation = 'tanh',input_shape = consts.input_shape))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(kernel_size = (3,3),filters = 64,activation = 'tanh'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(100, activation = 'tanh'))
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(16, activation = 'tanh'))

model.add(Dense(consts.classes,activation = consts.activation))

optimizer = Adam(lr = consts.lr)#SGD(lr = consts.lr,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer,loss = consts.loss, metrics=["accuracy"])

## training the model
model.fit(ximages/255,xlabels,epochs=consts.epochs,shuffle=True,validation_data=(yimages,ylabels))

prediction = dh.FromOneHot(model.predict(yimages))
correct = dh.FromOneHot(ylabels)


acc = (np.sum(prediction == correct)/len(list(yimages))) * 100
print('Acc :' + str(acc) + '%')

model.save('emotions' + str(round(acc)) + '.h5')
