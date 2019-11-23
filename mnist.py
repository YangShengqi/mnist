import numpy as np
from keras.datasets import mnist
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

#数据准备
img_rows, img_cols = 28, 28
nb_classes = 10
(x_train,y_train),(x_test,y_test) = mnist.load_data()
if backend.image_data_format()=='channels_first':
    x_train = np.reshape(x_train,[x_train.shape[0],1,img_rows,img_cols])
    x_test = np.reshape(x_test, [x_test.shape[0], 1,img_rows, img_cols])
    input_shape = (1,img_rows,img_cols)
else:
    x_train = np.reshape(x_train,[x_train.shape[0],img_rows,img_cols,1])
    x_test = np.reshape(x_test, [x_test.shape[0],img_rows, img_cols,1])
    input_shape = (img_rows,img_cols,1)
#x_train = x_train.reshape(len(x_train),-1)
#x_test = x_test.reshape(len(x_test),-1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

#搭建网络
model = Sequential()

model.add(Conv2D(6, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(512,activation='relu',input_shape=input_shape))
model.add(Dropout(0.2))

model.add(Dense(512,activation='relu',input_shape=input_shape))
model.add(Dropout(0.2))

model.add(Dense(nb_classes,activation='softmax'))

#配置
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#训练
model.fit(x_train,y_train,batch_size=64,epochs=5,verbose=2,validation_split=0.05)

#测试
loss, accuracy = model.evaluate(x_test,y_test)
print('Test loss:',loss)
print('Accuracy:',accuracy)