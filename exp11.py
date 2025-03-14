import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 
import pandas as pd 

#load data set
X=pd.read_csv("heart.csv")
x1 = X['chol'].values
x2 = X['trestbps'].values
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2) 

#code for EM
gmm = GaussianMixture(n_components=3) 
gmm.fit(X)
em_predictions = gmm.predict(X) 
print("\nEM predictions") 
print(em_predictions) 
print("mean:\n",gmm.means_)  
print('\n') 
print("Covariances\n",gmm.covariances_)


otp:
EM predictions
[0 0 0 0 2 0 2 0 2 0 0 2 0 0 2 0 2 0 2 0 0 0 0 0 0 2 0 0 1 0 0 0 0 2 0 0 2
 .............
 0 0 0 0 0 0 0]
mean:
 [[220.47052567 127.61093635]
 [427.84367395 138.21800435]
 [281.6497143  138.185068  ]]


Covariances
 [[[1101.66299046  -18.00192815]
  [ -18.00192815  209.05396436]]

 [[5192.27713546 -908.96436016]
  [-908.96436016  191.08744441]]

 [[1256.54862053  -84.44393471]
  [ -84.44393471  406.67704735]]]




print(X)
plt.title('Exceptation Maximum') 
plt.scatter(X[:,0], X[:,1],c=em_predictions,s=50) 
plt.show()


otp:
[[233 145]
 [250 130]
 [204 130]
 ......
 [193 144]
 [131 130]
 [236 130]]
-------draw graph---------------------------------------





#code for Kmeans
import matplotlib.pyplot as plt1 
kmeans = KMeans(n_clusters=3) 
kmeans.fit(X) 
print(kmeans.cluster_centers_) 
print(kmeans.labels_) 
plt.title('KMEANS')
plt1.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt1.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')



otp:

[[213.64088398 129.8121547 ]
 [564.         115.        ]
 [292.43801653 134.47107438]]
[0 0 0 0 2 0 2 2 0 0 0 2 2 0 2 0 2 0 0 0 0 0 0 0 0 2 0 0 2 0 0 0 0 2 0 0 2
 .............
 0 0 0 2 0 0 0]
<matplotlib.collections.PathCollection at 0x2baa1377b10>
-----draw graph----------

