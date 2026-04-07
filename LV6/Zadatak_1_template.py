import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning) ##da ne ispisuje futurewarning za scikitlearn kod knn-a

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine, VRLO VAZNO KOD KNN-a i SVM-a!!!!!!!!!!!!!!!!!!!!!!!!
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(LogReg-train): " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_n,y_train)
y_train_prediction_knn = knn_model.predict(X_train_n)
y_test_prediction_knn = knn_model.predict(X_test_n)
print("KNN: (K=5) ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_knn))))
##puno bolje od logisticke regresije
#granica odluke knn, primjecuje se bolja prilagodenost podacima (nelinearna) u odnosu na logisticku regresiju
plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(knn-train): " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
plt.tight_layout()
plt.show()
#granica odluke kada je K=1 overfit, K=100 underfit, vrlo komplicirana kada je K=1, jednostavnija kada je K=100

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train_n,y_train)
y_train_prediction_knn = knn_model.predict(X_train_n)
y_test_prediction_knn = knn_model.predict(X_test_n)
print("KNN: (K=1) ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_knn))))
plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(knn-train): " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
plt.tight_layout()
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=100)
knn_model.fit(X_train_n,y_train)
y_train_prediction_knn = knn_model.predict(X_train_n)
y_test_prediction_knn = knn_model.predict(X_test_n)
print("KNN: (K=100) ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_knn))))
plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(knn-train): " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
plt.tight_layout()
plt.show()


averageAccuracy = []
for i in range(1,51):
    scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=i), X=X_train_n, y=y_train, cv=5, scoring='accuracy')
    averageAccuracy.append(scores_knn.mean()) 
plt.plot(range(1,51),averageAccuracy)
plt.xlabel('Broj susjeda K')
plt.ylabel('Točnost')
plt.show()
#for k, acc in enumerate(averageAccuracy, start=1):
#    print(f"K={k}: prosječna točnost = {acc:.3f}")
print(f'Optimalan parametar ima prosječnu točnost {max(averageAccuracy)} i iznosi K={averageAccuracy.index(max(averageAccuracy))+1} (KNN)')


#svm_model = svm.SVC(kernel='rbf', C=1, gamma=1)
svm_model = svm.SVC(kernel='rbf', C=3, gamma=6)
#svm_model = svm.SVC(kernel='rbf', C=0.1, gamma=7)

#svm_model = svm.SVC(kernel='linear', C=1, gamma=1)
#svm_model = svm.SVC(kernel='poly', C=1, gamma=1)
#svm_model = svm.SVC(kernel='sigmoid', C=1, gamma=1)

#smanjivanjem C i gamme, dogada se underfit, povecanjem overfit, gamma je obicno oko 1, i C oko 1 za zadovoljavajucu tocnost
#promjenom kernela za iste parametre rezultati su drugaciji i granica odluke se bitno mijenja

svm_model.fit(X_train_n, y_train)
y_train_prediction_svm = svm_model.predict(X_train_n)
y_test_prediction_svm = svm_model.predict(X_test_n)
#granica odluke knn
print("SVM: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_svm))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_svm))))
plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(svm-train): " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_svm))))
plt.tight_layout()
plt.show()
#smanjivanjem C i gamme, dogada se underfit, povecanjem overfit, gamma je obicno oko 1, i C oko 1 za zadovoljavajucu tocnost
#promjenom kernela za iste parametre rezultati su drugaciji i granica odluke se bitno mijenja



param_grid_svm = {'C': [1,5,10,20,30], 'gamma': [0.01, 0.1, 1, 5, 10,15]}
svm_grid = GridSearchCV(svm.SVC(),param_grid_svm,cv=5,scoring='accuracy')
svm_grid.fit(X_train_n,y_train)
y_test_prediction_svm_grid = svm_grid.predict(X_test_n)
scores_svm=svm_grid.cv_results_
print(f'Rezultati GridSearcha za SVM za dane parametre:\n{pd.DataFrame(scores_svm)}')
print(f'Optimalni parametri su {svm_grid.best_params_} uz prosječnu točnost od {svm_grid.best_score_} (SVM)')
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_svm_grid))))


""" #ISCRTAVANJE I RAČUNANJE OPTIMALNOG HIPERPARAMETRA K ZA KNN (2. način)
param_grid_knn = {'n_neighbors': np.linspace(1,50, num=50, dtype=np.int64)}
knn_grid = GridSearchCV(KNeighborsClassifier(),param_grid_knn,cv=5,scoring='accuracy')
knn_grid.fit(X_train_n,y_train)
scores_knn=knn_grid.cv_results_
print(f'Rezultati GridSearcha za KNN za dane parametre:\n{pd.DataFrame(scores_knn)}')
print(f'Optimalan parametar je {knn_grid.best_params_}(broj susjeda) uz točnost od {knn_grid.best_score_}')
plt.plot(param_grid_knn['n_neighbors'], scores_knn['mean_test_score'])
plt.xlabel('Broj susjeda K')
plt.ylabel('Točnost')
plt.show()
"""




