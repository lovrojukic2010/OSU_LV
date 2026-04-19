import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#BITNO PRIJE POKRETANJA tf_env\Scripts\activate

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1) #1 kanal jer su grayscale slike, a ne RGB

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
fig, axs = plt.subplots(2, 3, figsize=(10, 7))
brojac = 0
for i in range(2):
    for j in range(3):
        axs[i, j].imshow(x_train[brojac], cmap='gray')
        axs[i, j].set_title(f'Oznaka: {y_train[brojac]}')
        axs[i, j].axis('off')
        brojac += 1
plt.tight_layout()
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

#one-hot encoding
# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# TODO: provedi ucenje mreze
history = model.fit(
    x_train_s,
    y_train_s,
    batch_size=128,
    epochs=15,
    validation_split=0.1
)


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f'loss = {score[0]}')
print(f'accuracy = {score[1]}')

y_pred = model.predict(x_test_s)#predict vraća vjerojatnost svih 10 klasa za svaku sliku
y_pred_classes = np.argmax(y_pred, axis=1)#ovdje se uzima indeks argumenta sa najvecom "vjerojatnoscu"

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()


# TODO: spremi model
model.save("Model/model.keras")
