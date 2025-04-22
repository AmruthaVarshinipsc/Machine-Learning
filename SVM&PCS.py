from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
cancer=load_breast_cancer()
x=cancer.data[:, :2]
y=cancer.target
svm=SVC(kernel="linear")
svm.fit(x,y)
plt.scatter(x[:, 0],x[:, 1],
            c=y,
            s=20,edgecolors="k")



OUTPUT:
-----draw graph---
