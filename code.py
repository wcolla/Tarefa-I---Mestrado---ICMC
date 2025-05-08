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
df.info()
df.isnull().sum()
df.shape

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


# Tratar outliers
# Converter a coluna de data
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Preencher NaNs numéricos com a mediana
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Preencher NaNs categóricos com a moda
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))


# Utilizar técnicas apresentadas em aula para substituição de dados e tratamento de outliers
# Função para tratamento de outliers usando IQR
def treat_outliers_iqr(series):
  q1 = series.quantile(0.25)
  q3 = series.quantile(0.75)
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  return np.where(series < lower, lower, np.where(series > upper, upper, series))

# Aplicar para colunas numéricas
for col in numeric_cols:
  df[col] = treat_outliers_iqr(df[col])


# Realizar avaliação estatística dos dados
# Estatíticas

# Moda
print("Moda (mais frequente):")
print(df.mode().iloc[0])

# Variância
print("\nVariância:")
print(df.var(numeric_only=True))

# Assimetria
print("\nAssimetria (Skew):")
print(df.skew(numeric_only=True))

# Curtose
print("\nCurtose:")
print(df.kurt(numeric_only=True))

# Matriz de correlação
print("Matriz de correlação:")
print(df.corr(numeric_only=True))

# Verificação de valores negativos (em colunas que não deveriam tê-los)
print("Colunas com valores negativos:")
for col in df.select_dtypes(include=[np.number]):
    if (df[col] < 0).any():
        print(f"{col} tem valores negativos.")

# Comparar média vs. mediana (detecção de assimetria ou outliers)
print("\n Média vs Mediana por coluna numérica:")
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"{col}: média = {df[col].mean():.2f}, mediana = {df[col].median():.2f}")

# Medidas de Dispersão
print("\n📊 Medidas de dispersão:")
print("Desvio padrão:\n", df.std(numeric_only=True))
print("\nCoeficiente de variação (%):")
print((df.std(numeric_only=True) / df.mean(numeric_only=True)) * 100)

# Teste de Normalidade
print("\n📈 Teste de normalidade (Shapiro-Wilk):")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].count() >= 5000:  # shapiro tem limitação
        sample = df[col].dropna().sample(5000, random_state=42)
    else:
        sample = df[col].dropna()
    stat, p = shapiro(sample)
    print(f"{col}: stat={stat:.4f}, p={p:.4f} {'(Normal)' if p > 0.05 else '(Não normal)'}")

# Teste de Hipótese
print("\n🧪 Teste t de hipótese: Usage_kWh entre 'Weekday' vs 'Weekend'")
df = df.dropna(subset=['Usage_kWh', 'WeekStatus'])

weekday = df[df['WeekStatus'] == 'Weekday']['Usage_kWh']
weekend = df[df['WeekStatus'] == 'Weekend']['Usage_kWh']

t_stat, p_val = ttest_ind(weekday, weekend, equal_var=False)
print(f"T-stat: {t_stat:.4f}, p-valor: {p_val:.4f} {'(Diferença significativa)' if p_val < 0.05 else '(Sem diferença significativa)'}")




# Codificação de variáveis categóricas
le_target = LabelEncoder()
df['Load_Type'] = le_target.fit_transform(df['Load_Type'])  # Alvo

# Codifique outras colunas com LabelEncoder novos
df['WeekStatus'] = LabelEncoder().fit_transform(df['WeekStatus'])
df['Day_of_week'] = LabelEncoder().fit_transform(df['Day_of_week'])

# Separando features e alvo
X = df.drop(columns=['Load_Type', 'date'])
y = df['Load_Type']

# Separação em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dicionário com os modelos
modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Regressão Logística': LogisticRegression(max_iter=1000),
    'Árvore de Decisão (CART)': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', C=1, gamma='scale')
}

# Codificando os rótulos para os nomes de classe legíveis, se necessário
target_names = [str(label) for label in le_target.classes_]

# Treinamento e avaliação com for
for nome, modelo in modelos.items():
    print(f"\n🔍 Modelo: {nome}")

    # Alguns modelos usam dados normalizados, outros não
    if nome in ['KNN', 'Regressão Logística', 'SVM']:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)



    # Avaliação
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=target_names))



# Codificação de variáveis categóricas
le_target = LabelEncoder()
df_copy['Load_Type'] = le_target.fit_transform(df_copy['Load_Type'])
df_copy['WeekStatus'] = LabelEncoder().fit_transform(df_copy['WeekStatus'])
df_copy['Day_of_week'] = LabelEncoder().fit_transform(df_copy['Day_of_week'])

# Separar X e y antes da remoção de outliers
X = df_copy.drop(columns=['Load_Type', 'date'])
y = df_copy['Load_Type']

# Normalizar para aplicar Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular Z-score e remover outliers (Z > 3 ou < -3)
z_scores = zscore(X_scaled)
mask = ~(z_scores > 3).any(axis=1)  # Mantém apenas linhas sem outliers

X_no_outliers = X[mask]
y_no_outliers = y[mask]

# Garantir que não restaram NaNs
X_no_outliers = X_no_outliers.dropna()
y_no_outliers = y_no_outliers.loc[X_no_outliers.index]

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_no_outliers, y_no_outliers, test_size=0.3, stratify=y_no_outliers, random_state=42)

# Normalização final
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dicionário de modelos
modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Regressão Logística': LogisticRegression(max_iter=1000),
    'Árvore de Decisão (CART)': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', C=1, gamma='scale')
}

# Nomes das classes
target_names = [str(c) for c in le_target.classes_]

# Avaliação dos modelos
for nome, modelo in modelos.items():
    print(f"\n🔍 Modelo: {nome}")

    if nome in ['KNN', 'Regressão Logística', 'SVM']:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=target_names))
