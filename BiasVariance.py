from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
import numpy as np


X, y = iris_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

tree = DecisionTreeClassifier()

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    tree, X_train, y_train, X_test, y_test, loss='0-1_loss', num_rounds=100
)

print(f'Average Expected Loss: {round(avg_expected_loss, 4)}')
print(f'Average Bias: {round(avg_bias, 4)}')
print(f'Average Variance: {round(avg_var, 4)}')

print(f'Average Expected Loss: {round(avg_expected_loss * 100, 2)}%')
print(f'Average Bias: {round(avg_bias * 100, 2)}%')
print(f'Average Variance: {round(avg_var * 100, 2)}%')


otp:
Average Expected Loss: 0.0529
Average Bias: 0.0444
Average Variance: 0.0453
Average Expected Loss: 5.29%
Average Bias: 4.44%
Average Variance: 4.53%
