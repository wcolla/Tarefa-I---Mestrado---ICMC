import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import zscore

#carregar os dados
df = pd.read_csv("/content/Steel_industry_data_task01.csv")
df.head()

# Apenas as linhas que tem NaN
nan_rows = df[df.isna().any(axis=1)]
print("Apenas linhas com NaN:")
print(nan_rows)

# NaN por coluna
print("Quantidade de NaN por coluna:")
print(df.isna().sum().sort_values(ascending=False))

# Dataframe booleano com a localização dos NaNs
print("Localização dos NaNs (True = é NaN):")
print(df.isna())

# Colunas que tem algum NaN
print("Colunas com pelo menos um NaN:")
print(df.columns[df.isna().any()].tolist())

# 10 primeiras linhas com NaN
print("Amostra de linhas com NaN:")
print(df[df.isna().any(axis=1)].head(10))

df_copy = df.copy()
