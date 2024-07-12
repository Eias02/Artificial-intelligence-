import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# set default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# load and normalize cifar-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# class names for cifar-10
class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# display some sample images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    label = class_name[training_labels[i][0]]
    plt.xlabel(label.encode('utf-8').decode('utf-8'))
# plt.show()

# trim dataset for faster training (optional)
training_images = training_images[:10000]
training_labels = training_labels[:10000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# define the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# evaluate the model on the testing dataset
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
print(f'\ntest accuracy: {test_acc}')

# make a prediction on a custom image
# replace "put-your-pic-here" with the path to your image file
img=cv.imread("put-your-pic-here")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)

# preprocess the image and make a prediction
prediction=model.predict(np.array([img]/255))
index=np.argmax(prediction)
print(f"the prediction is {class_name[index]}")

plt.show()
