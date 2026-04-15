import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
h,w,d = img.shape
img_array = np.reshape(img, (h*w, d))

# rezultatna slika
img_array_aprox = img_array.copy()

print(f'Broj boja u originalnoj slici: {len(np.unique(img_array_aprox, axis=0))}') #ili .shape[0] za broj redova

"""#trazenje lakta, optimalnog broja grupa (K), lakat je uočljiv
squareSums = []
for i in range (1,10):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(img_array_aprox)
    squareSums.append(kmeans.inertia_)

plt.plot(range(1,10), squareSums)
plt.xlabel('K')
plt.show()
"""

#racunanje kmeans za K=5 slike (ne mora znacit da je to optimalan K, nije potvrden laktom iznad)
kmeans = KMeans(n_clusters=5, init='random')
kmeans.fit(img_array_aprox)
labels=kmeans.predict(img_array_aprox)

for i in range(len(labels)):
    img_array_aprox[i]=kmeans.cluster_centers_[labels[i]]    #promjena svakog retka(boje) u jednu od k boja (najblizi i odgovarajuci centroid) - kvantizacija slike

print(f'Broj boja u aproksimiranoj slici: {len(np.unique(img_array_aprox, axis=0))} (jednak predodredenom broju grupa)')
img_aprox = np.reshape(img_array_aprox, (h,w,d))    #povratak na originalnu dimenziju slike 
img_aprox = (img_aprox*255).astype(np.uint8)        #povratak iz raspona 0 do 1 u int
plt.figure()
plt.title("Aproksimirana slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()
#povecavanjem broja grupa, vise je boja, slika sve vise izgleda kao originalna, smanjivanjem manje boja i sve dalje od originala
#duljina izvodenja programa se bitno povecava s povecanjem broja grupa
#test4 - cudne vrijednosti piksela, linija 36 maknuti astype jer su pikseli u floatu


labels_unique = np.unique(labels)
for i in range(len(labels_unique)):
    binary_image = labels==labels_unique[i] #labels je n_pixela x 1 shape
    binary_image = np.reshape(binary_image, (h,w)) #potrebno reshapeat za prikaz nazad u normalne dimenzije slike(bez rgb dimenzije)
    plt.figure()
    plt.title(f"Binarna slika {i+1}. grupe boja")
    plt.imshow(binary_image)
    plt.tight_layout()
    plt.show()
#prikazom binarnih slika svake grupe, primjecuje se da su grupe disjunktni skupovi, tj. svaka grupa predstavlja jednu boju na slici








