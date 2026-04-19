from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


#BITNO PRIJE POKRETANJA tf_env\Scripts\activate


#ucitavanje modela
model = load_model('Model/model.keras')
model.summary()

#ucitavanje slike, u MNIST-u su brojevi bijelom bojom na crnoj pozadini pa je potrebna zamjena vrijednosti crne i bijele boje jer je ova slika crna olovka bijela pozadina
img = plt.imread('nesto.png')[:,:,0]*255   #bude izmedu 0 i 1 pa treba mnozit s 255, bude rgb, pa treba odstranit nepotrebno jer je crno bijelo
img = img.astype('uint8')
img = (255 - img) / 255.0      #zamjena, da 255 simbolizira gdje nesto pise, a 0 gdje nema nista (kada ucita je obrnuto jer je crni tekst(0) na bijeloj pozadini(255))
img = np.expand_dims(img, axis=(0, -1))

#predikcija
img_prediction = model.predict(img)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (1,10)
img_prediction = np.argmax(img_prediction, axis=1) #vrati (1,) (izdvaja max index u svakom retku pa se dobije 1d)

#prikaz slike
img_display = np.squeeze(img, axis=(0, -1))  # za prikaz: (H, W)
plt.imshow(img_display, cmap='gray')
plt.title(f'Stvarni broj: 2, predikcija: {img_prediction[0]}')
plt.axis('off')
plt.show()