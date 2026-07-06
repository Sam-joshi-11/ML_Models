# creation convolutional Layer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core import history
from sklearn.metrics import confusion_matrix,classification_report

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

print(f"Train Data shape:{x_train.shape}\n")
print(f"Test Data Shape:{x_test.shape}\n")
print(y_train[:10])

plt.figure(figsize=(12,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title([y_train[i][0]])
    plt.axis("off")
plt.tight_layout()

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

print(x_train[0])

x_train = x_train/255.0
x_test = x_test/255.0
print("Minimum:",x_train.min())
print("Maximum:",x_train.max())

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(32,32,3)
    ),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        64,
        activation='relu',
    ),
    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=23,
    validation_split=0.2
)

print(history.history.keys())

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],label="Training Accuracy")
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label='Validation loss')
plt.show()

predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
print("Predicted Digit :",predicted_label)
print("Actual Digit :",y_test[0])

y_pred = np.argmax(predictions,axis=1)
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='d'
)
plt.show()  
print("Classification Report:",classification_report(y_test,y_pred))