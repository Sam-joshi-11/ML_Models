import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
print("Training Images :",X_train.shape)

plt.imshow(X_train[0],cmap='gray')
plt.show()
X_train=X_train/255.0
X_test=X_test/255.0

print("Max Pixel:",X_train.max())

#X_train=X_train.reshape(-1,28,28,1)
#X_test=X_test.reshape(-1,28,28,1)
X_train = X_train.reshape(60000,784)
print(X_train[0])
print(y_train[:10])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu',input_shape=(784,)),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
from IPython.core import history
history= model.fit(X_train,y_train,epochs=10,batch_size=50,validation_split=0.2)

print(history.history.keys())

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history["val_accuracy"],label='val acc')
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.legend()
plt.show()

X_test = X_test.reshape(-1, 784)
y_pred = model.predict(X_test)
print(y_pred[0])

print(model.evaluate(X_test,y_test))
print(model.summary())

pred_label = np.argmax(y_pred[0])
print("pred label",pred_label)
print("actual value",y_test[0])


predictions = np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_test,predictions)

plt.figure(figsize=(8,3))
import seaborn as sns
sns.heatmap(cm,cmap='Blues',fmt='d')
plt.show()

print("\nClassification report:",classification_report(y_test,predictions))

'''model.save("mnist.keras")
print("saved")

load_model = tf.keras.model("mnist.keras")
print("model loaded")'''