import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
df = pd.read_csv("StudentsPerformance_M4YygK2.csv")
scores = df[["math score", "reading score", "writing score"]]
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print("duplicate:", df.duplicated().sum())
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns
print("numeric:", list(num_cols))
print("categorical:", list(cat_cols))
# Task A
for col in cat_cols:
    print(col, df[col].nunique(), list(df[col].unique()))
print(df.describe())
for col in num_cols:
    print(f"{col:15s} mean={df[col].mean():.2f}  median={df[col].median():.2f}  mode={df[col].mode()[0]}")
for col in num_cols:
    print(f"{col:15s} variance={df[col].var():.2f}  std={df[col].std():.2f}")
for col in num_cols:
    m, med = df[col].mean(), df[col].median()
    if abs(m - med) < 0.5:
        shape = "Symmetric"
    elif m < med:
        shape = "Left-skewed"
    else:
        shape = "Right-skewed"
    print(f"{col:15s} mean={m:.2f} median={med:.2f} -> {shape}")
# Graphs
plt.figure()
df["race/ethnicity"].value_counts().sort_index().plot(kind="bar")
plt.title("Bar Chart: Race/Ethnicity Group")
plt.xlabel("Group")
plt.ylabel("Count")
plt.show()
plt.figure()
plt.hist(df["reading score"], bins=10)
plt.title("Histogram: Reading Score")
plt.xlabel("Reading Score")
plt.ylabel("Count")
plt.show()
df.boxplot(column="writing score", by="test preparation course")
plt.title("Writing Score by Test Preparation Course")
plt.suptitle("")
plt.xlabel("Test Preparation Course")
plt.ylabel("Writing Score")
plt.show()
plt.figure()
plt.scatter(df["reading score"], df["writing score"])
plt.title("Scatter: Reading vs Writing Score")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.show()
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def manhattan(a, b):
    return np.sum(np.abs(a - b))
sample = df.loc[0:2, ["math score", "reading score", "writing score"]]
sample.index = ["A", "B", "C"]
print(sample)
for i, j in combinations(sample.index, 2):
    a, b = sample.loc[i].values, sample.loc[j].values
    print(f"{i}-{j}:  Euclidean={euclidean(a, b):.2f}   Manhattan={manhattan(a, b):.2f}")
# Task B
sample2 = df.loc[3:5, ["math score", "reading score", "writing score"]]
sample2.index = ["D", "E", "F"]
print(sample2)
for i, j in combinations(sample2.index, 2):
    a, b = sample2.loc[i].values, sample2.loc[j].values
    print(f"{i}-{j}: Euclidean={euclidean(a, b):.2f}  Manhattan={manhattan(a, b):.2f}")
minmax = (scores - scores.min()) / (scores.max() - scores.min())
zscore = (scores - scores.mean()) / scores.std()
print(minmax.head())
print(zscore.head())
# Task C
minmax3 = minmax.head(3)
minmax3.index = ["A", "B", "C"]
for i, j in combinations(minmax3.index, 2):
    a, b = minmax3.loc[i].values, minmax3.loc[j].values
    print(f"{i}-{j}: Euclidean={euclidean(a, b):.4f}")
# Task D
for col in ["gender", "lunch", "test preparation course"]:
    m = df.groupby(col)["math score"].mean()
    print(m)
    print("зөрүү:", round(m.max() - m.min(), 2), "\n")
print(df[["math score", "reading score", "writing score"]].var())