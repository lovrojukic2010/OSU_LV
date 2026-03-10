import numpy as np
import matplotlib.pyplot as plt

firstLight = np.ones((50, 50))
secondLight = firstLight.copy()
firstDark = np.zeros((50, 50))
secondDark = firstDark.copy()

firstRow = np.hstack((firstDark, firstLight))
secondRow = np.hstack((secondLight, secondDark))
fullSquare = np.vstack((firstRow, secondRow))
plt.imshow(fullSquare, cmap="gray")
plt.title("Crno bijeli kvadrat")
plt.show()
