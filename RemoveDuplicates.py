import pandas as pd
data={
    "A":["TeamA","TeamB","TeamB","TeamC","TeamA"],
    "B":[50,40,40,30,50],
    "C":[True,False,False,False,True]
}
df=pd.DataFrame(data)
print(df)
display(df.drop_duplicates())

OUTPUT:
       A   B      C
0  TeamA  50   True
1  TeamB  40  False
2  TeamB  40  False
3  TeamC  30  False
4  TeamA  50   True
        A   B	  C
0	TeamA	50	True
1	TeamB	40	False
3	TeamC	30	False



import pandas as pd
from mlxtend.data import iris_data

x, y = iris_data()
df = pd.DataFrame(x, columns=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])
df["Species"] = y

print("Original Iris DataFrame:")
print(df)
print(f"\nTotal records in original dataframe: {len(df)}")

df = df.drop_duplicates()

print("\nIris dataframe after removing duplicates:")
print(df)
print(f"\nTotal records after removing duplicates: {len(df)}")


OUTPUT:
Original Iris DataFrame:
     Sepal_Length  Sepal_Width  Petal_Length  Petal_Width  Species
0             5.1          3.5           1.4          0.2        0
1             4.9          3.0           1.4          0.2        0
2             4.7          3.2           1.3          0.2        0
3             4.6          3.1           1.5          0.2        0
4             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
145           6.7          3.0           5.2          2.3        2
146           6.3          2.5           5.0          1.9        2
147           6.5          3.0           5.2          2.0        2
148           6.2          3.4           5.4          2.3        2
149           5.9          3.0           5.1          1.8        2

[150 rows x 5 columns]

Total records in original dataframe: 150

Iris dataframe after removing duplicates:
     Sepal_Length  Sepal_Width  Petal_Length  Petal_Width  Species
0             5.1          3.5           1.4          0.2        0
1             4.9          3.0           1.4          0.2        0
2             4.7          3.2           1.3          0.2        0
3             4.6          3.1           1.5          0.2        0
4             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
145           6.7          3.0           5.2          2.3        2
146           6.3          2.5           5.0          1.9        2
147           6.5          3.0           5.2          2.0        2
148           6.2          3.4           5.4          2.3        2
149           5.9          3.0           5.1          1.8        2

[147 rows x 5 columns]

Total records after removing duplicates: 147
