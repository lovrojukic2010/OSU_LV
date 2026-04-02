import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
################
colors=['blue', 'red']
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors)) #redom, 0 blue, 1 red
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker="x", cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Podaci za treniranje(.) i testiranje(x)")
cbar=plt.colorbar(ticks=[0,1])
plt.show()

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)

bias=logisticRegression.intercept_ #teta0, coef vraca samo parametre uz ulazne velicine (za linearnu isto! (lv4))
coefs=logisticRegression.coef_    #provjeriti pravac odluke
print(coefs.shape)
a = -coefs[0,0]/coefs[0,1]
c = -bias/coefs[0,1]
x1x2min = X_train.min().min()-0.5
x1x2max = X_train.max().max()+0.5
xd = np.array([x1x2min, x1x2max]) #za pravac dovoljne dvije tocke
yd = a*xd + c
plt.plot(xd, yd, linestyle='--')
plt.fill_between(xd, yd, x1x2min, color='red', alpha=0.2) #1
plt.fill_between(xd, yd, x1x2max, color='blue', alpha=0.2) #0
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="white")
plt.xlim(x1x2min, x1x2max)
plt.ylim(x1x2min, x1x2max)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci za treniranje i granica odluke')
cbar=plt.colorbar(ticks=[0,1])
plt.show()

y_prediction=logisticRegression.predict(X_test)
cm=confusion_matrix(y_test,y_prediction)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Matrica zabune')
plt.show()
print(f'Točnost: {accuracy_score(y_test,y_prediction)}')
print(f'Preciznost: {precision_score(y_test,y_prediction)}')
print(f'Odziv: {recall_score(y_test,y_prediction)}')

colorsEvaluation=['black', 'green']
plt.scatter(X_test[:,0], X_test[:,1], c=y_test==y_prediction, cmap=matplotlib.colors.ListedColormap(colorsEvaluation)) #redom, false black, true green
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Točnost predikcije na podacima za testiranje')
cbar=plt.colorbar(ticks=[0,1])
cbar.ax.set_yticklabels(['Netočno','Točno'])
plt.show()