import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
mnist = tf.keras.datasets.fashion_mnist
(train_image, train_label),(test_image, test_label) = mnist.load_data()
categories = {"T-shirt/top": 0, "Trouser": 1,"Pullover":2, "Dress":3, "Coat":4, "Sandel": 5, "Shirt": 6, "Sneaker": 7, "Bag": 8, "Ankle_Boat" : 9}
print(categories)
train_image = train_image/255.0
test_image = test_image/255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_image,train_label, epochs=15)
test_loss, test_accuracy = model.evaluate(test_image, test_label)
prediction = model.predict(test_image)
print(prediction[100])
print(np.argmax(prediction[100]))
