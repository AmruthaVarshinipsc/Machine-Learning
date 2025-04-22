from mlxtend.data import iris_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold

x, y = iris_data()
clf = DecisionTreeClassifier(random_state=1)
scores = cross_val_score(clf, x, y, cv=KFold(n_splits=6))

print("Cross Validation:", scores)
print("Average CV Score:", scores.mean())
print("Number of CV Scores used in Average:", len(scores))

OUTPUT:
Cross Validation: [1.   1.   0.96 0.92 0.92 0.88]
Average CV Score: 0.9466666666666667
Number of CV Scores used in Average: 6
