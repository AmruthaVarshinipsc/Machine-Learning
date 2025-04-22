from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
colormap = np.array(['red', 'lime', 'black'])

plt.figure(figsize=(14, 7))


plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=colormap[y])


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)


mapped_kmeans = np.zeros_like(kmeans_labels)
for i in range(3):
    mask = (kmeans_labels == i)
    if np.any(mask):
        mapped_kmeans[mask] = mode(y[mask], keepdims=True).mode[0]

plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=colormap[kmeans_labels])

print("The accuracy score of K-Mean: ", metrics.accuracy_score(y, mapped_kmeans))
print("The Confusion matrix of K-Mean:\n", metrics.confusion_matrix(y, mapped_kmeans))


gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X)


mapped_gmm = np.zeros_like(gmm_labels)
for i in range(3):
    mask = (gmm_labels == i)
    if np.any(mask):
        mapped_gmm[mask] = mode(y[mask], keepdims=True).mode[0]

plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=colormap[gmm_labels])

print("The accuracy score of EM: ", metrics.accuracy_score(y, mapped_gmm))
print("The Confusion matrix of EM:\n", metrics.confusion_matrix(y, mapped_gmm))

plt.tight_layout()
plt.show()


OUTPUT:
The accuracy score of K-Mean:  0.8866666666666667
The Confusion matrix of K-Mean:
 [[50  0  0]
 [ 0 47  3]
 [ 0 14 36]]
The accuracy score of EM:  0.9666666666666667
The Confusion matrix of EM:
 [[50  0  0]
 [ 0 45  5]
 [ 0  0 50]]

------draw graph---------------
