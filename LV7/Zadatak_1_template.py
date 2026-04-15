import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)
# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()
#broj grupa u podacima se lako moze prepoznati uz pomoc vizualizacije (dijagram rasprsenja) za svaki od nacina generiranja podataka (1-5)

kmeans = KMeans(n_clusters=3, init ='random')
kmeans.fit(X)
labels = kmeans.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Grupirani podatkovni primjeri')
plt.show()

"""
squareSums = []
for i in range (1,10):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(X)
    squareSums.append(kmeans.inertia_)

plt.plot(range(1,10), squareSums)
plt.xlabel('K')
plt.show()
""" #lakat metoda provjere optimalnog K 
#neispravnim postavljanjem broja k dobija se previše ili premalo grupa
#kmeans kod nekih primjera ne grupira kako treba jer pretpostavlja da su grupe sferične, podjednake velicine i slicne gustoce,
#ne radi dobro s grupama nepravilnih oblika (jer radi na principu udaljenosti) (uz primjenu optimalnih vrijednosti k)
#kada flagc=1, radi dobro jer su grupe sfericne
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],c='red', s=120, marker='x') - označavanje centara

