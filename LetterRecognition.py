import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.saving.save import load_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv(
    r"C:\Users\Andrei\Documents\UBB\2022-2023\Sem1\CVDL\project\Datasets\A_Z Handwritten Data.csv").astype('float32')
# print(data.head(10))

images = data.drop('0', axis=1)
labels = data['0']

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2)

train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
             13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: 'Z'}

y_int = np.int0(labels)
count = np.zeros(len(word_dict), dtype='int')
for i in y_int:
    count[i] += 1
alphabets = []
for i in word_dict.values():
    alphabets.append(i)

# fig, ax = plt.subplots(1, 1, figsize=(10,10))
# ax.barh(alphabets, count)
# plt.xlabel("Number of elements ")
# plt.ylabel("Alphabets")
# plt.grid()
# plt.show()

# shuff = shuffle(train_x[:100])
# fig, ax = plt.subplots(5,5, figsize = (10,10))
# axes = ax.flatten()
# for i in range(25):
#     _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
#     axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
# plt.show()

# fig, ax = plt.subplots(6, 6, figsize = (10,10))
# axes = ax.flatten()
# values = train_y.values
# added = list()
# imgs = list()
# for i in range(len(values)):
#     if int(values[i]) not in added:
#         added.append(int(values[i]))
#         imgs.append(train_x[i])
#         _, shu = cv2.threshold(train_x[i], 30, 200, cv2.THRESH_BINARY)
#         axes[len(added) - 1].imshow(np.reshape(train_x[i], (28,28)), cmap="Greys")
#
#         img = (255 - imgs[-1])
#         img = Image.fromarray(img).convert("RGB")
#         img.save("Datasets/" + word_dict[added[-1]] + ".jpeg")
# plt.show()
# added.sort()
# added = [word_dict[element] for element in added]
# print(added)
# print(len(added))
#
# for img in imgs:
#     img = (255 - img)
#     cv2.imshow("Img", img)
#     cv2.waitKey()


train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
print("New shape of train data: ", train_X.shape)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
print("New shape of train data: ", test_X.shape)

train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
print("New shape of train labels: ", train_yOHE.shape)
test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_yOHE, epochs=5, validation_data=(test_X, test_yOHE))

model.summary()
model.save(r'model_hand.h5')


print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

fig, axes = plt.subplots(3, 3, figsize=(8, 9))
axes = axes.flatten()
for i, ax in enumerate(axes):
    img = np.reshape(test_X[i], (28, 28))
    ax.imshow(img, cmap="Greys")

    pred = word_dict[np.argmax(test_yOHE[i])]
    ax.set_title("Prediction: " + pred)
    ax.grid()
