import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("menu.csv")
df = df[['Item', 'Calories', 'Total Fat', 'Carbohydrates', 'Protein', 'Sodium', 'Sugars']].dropna()

median_cal = df['Calories'].median()
median_fat = df['Total Fat'].median()
df['Label'] = df.apply(lambda row: 1 if (row['Calories'] > median_cal and row['Total Fat'] > median_fat) else 0, axis=1)

X = df[['Calories', 'Total Fat', 'Carbohydrates', 'Protein', 'Sodium', 'Sugars']].values
y = df['Label'].values
items = df['Item'].values

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

X = standardize(X)

def train_test_split_manual(X, y, items, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    idx = np.random.permutation(len(X))
    test_len = int(len(X) * test_size)
    return X[idx[test_len:]], X[idx[:test_len]], y[idx[test_len:]], y[idx[:test_len]], items[idx[test_len:]], items[idx[:test_len]]

X_train, X_test, y_train, y_test, items_train, items_test = train_test_split_manual(X, y, items)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=5):
    pred = []
    for x in X_test:
        dists = [euclidean_distance(x, x_tr) for x_tr in X_train]
        k_idx = np.argsort(dists)[:k]
        k_labels = [y_train[i] for i in k_idx]
        pred.append(Counter(k_labels).most_common(1)[0][0])
    return np.array(pred)

y_pred = knn_predict(X_train, y_train, X_test, k=5)

def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

cm = confusion_matrix_manual(y_test, y_pred)
acc = np.mean(y_pred == y_test)

tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
p1 = tp / (tp + fp) if (tp + fp) else 0
r1 = tp / (tp + fn) if (tp + fn) else 0
f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) else 0
p0 = tn / (tn + fn) if (tn + fn) else 0
r0 = tn / (tn + fp) if (tn + fp) else 0
f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) else 0

print("\nLaporan Klasifikasi:")
print(f"{'Label':<8}{'Precision':<10}{'Recall':<10}{'F1-score':<10}{'Support'}")
print(f"{'0':<8}{p0:<10.2f}{r0:<10.2f}{f1_0:<10.2f}{np.sum(y_test==0)}")
print(f"{'1':<8}{p1:<10.2f}{r1:<10.2f}{f1_1:<10.2f}{np.sum(y_test==1)}")
print(f"\nAccuracy: {acc:.2%}")

print("\nRangkuman:")
print(f"- {cm[0,0]} SEHAT benar")
print(f"- {cm[1,1]} TIDAK SEHAT benar")
print(f"- {cm[0,1]} SEHAT salah")
print(f"- {cm[1,0]} TIDAK SEHAT salah")

res = pd.DataFrame({'Item': items_test, 'Prediksi': y_pred})
sehat = res[res['Prediksi'] == 0]['Item'].tolist()
tdk_sehat = res[res['Prediksi'] == 1]['Item'].tolist()

print("\nMenu SEHAT:")
for i in sehat: print("-", i)

print("\nMenu TIDAK SEHAT:")
for i in tdk_sehat: print("-", i)

lbl = np.array([
    [f'Benar Sehat\n{cm[0,0]}', f'Salah Sehat\n{cm[0,1]}'],
    [f'Salah Tidak Sehat\n{cm[1,0]}', f'Benar Tidak Sehat\n{cm[1,1]}']
])

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=lbl, fmt='', cmap='RdYlGn', linewidths=2, linecolor='black', cbar=False, square=True)
plt.xlabel('Prediksi')
plt.ylabel('Label Sebenarnya')
plt.title(f'Matriks Kebingungan\nTotal: {len(y_test)} | Akurasi: {acc:.2%}')
plt.tight_layout()
plt.show()