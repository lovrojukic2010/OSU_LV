import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

#X_train_n = X_train_n[0:25000,:,:,:] #izdvajanje polovine skupa podataka za ucenje u svrhu zadatka


# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza, broj parametara = 1 122 758
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout_earlyStop',
                                update_freq = 100),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
]

optimizer = keras.optimizers.Adam() #kontruktoru predati proizvoljni learning_rate, postoji default
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#za tensorboard, otvoriti novi terminal (New terminal) i upisati tensorboard --logdir logs/cnn_dropout, tensorboard --logdir logs/cnn_dropout_earlyStop 
#logs-direktorij u koji se sprema, naveden u Callbacku
#otvoriti dani link i prikazani su grafovi

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)

#dogodio se overfit, na validacijskom skupu vidljiv je pad točnosti i porast gubitka već nakon prvih 6-7 epoha, 
#dok se na skupu za učenje se sve metrike povećavaju (loss je blizu 0, accuracy 1)
#nakon primjene dropout sloja, točnost na testnom skupu se bitno povećala, učinak overfita je smanjen (iako je i dalje je 6-7 epoha optimalan broj)
#earlyStop gleda najmanji prosjecni loss, a ne razliku izmedu lossa 2 susjedne epohe, ako se za 5 ne poboljsa(smanji) najbolji, zaustavlja se

#jako velika velicina batcha - manje iteracija, krace trajanje epohe, losija tocnost i veci loss 
#jako mala velicina batcha - vise iteracija, duze trajanje epohe (predugo)
#jako mala vrijednost stope ucenja - ucenje izrazito sporo konvergira, loss jedva pada, accuracy jedva raste 
#jako velika vrijednost stope ucenja - loss velik, accuracy mali, i ne mijenjaju se 
#izbacivanje slojeva iz mreze za manju mrezu - izbacivanjem jednog dense, conv2d i maxpooling sloja, epoha traje krace, losiji accuracy i loss veci
#50% manja velicina skupa za ucenje - upola manje iteracija, veci loss i losiji accuracy, epoha traje krace

score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}%')
#Dobivena točnost nakon 40 epoha na testnom skupu = 73.27%
#Dobivena točnost nakon 20 epoha na testnom skupu uz dropout sloj izmedu 2 potpuno povezana sloja uz rate 0.3 = 76.04%
#Dobivena točnost nakon 11-12 epoha na testnom skupu uz dropout sloj izmedu 2 potpuno povezana sloja uz rate 0.3 i earlyStop(patience=5) = 74.23%
#Stane nakon 11. epohe zbog earlyStoppinga



