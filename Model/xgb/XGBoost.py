# %%
import os
import pandas as pd
from obspy import read
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft
from scipy.stats import entropy

# %%
splits_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits-baru-XGB"
train_manifest = os.path.join(splits_root, "train_scaled.csv")
val_manifest  = os.path.join(splits_root, "val_scaled.csv")
test_manifest  = os.path.join(splits_root, "test_scaled.csv")

# %%
def load_features(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['path', 'label']).values  # Semua kolom selain 'path' dan 'label' adalah fitur
    y = df['label'].values                         # Label
    return X, y

# %%

X_train, y_train = load_features(train_manifest)
X_val, y_val = load_features(val_manifest)
X_test, y_test = load_features(test_manifest)

print("Train full shape:", X_train.shape)
print("Train shape:", X_train.shape)
print("val shape:", X_val.shape)
print("Test shape :", X_test.shape)

# %%
def class_distribution(labels, name):
    unique, counts = np.unique(labels, return_counts=True)
    dist = pd.DataFrame({
        "class": unique,
        "count": counts,
        "percentage": (counts / counts.sum() * 100).round(2)
    })
    print(f"\n{name} distribution:")
    print(dist.to_string(index=False))

class_distribution(y_train, "Train")
class_distribution(y_val, "Validation")
class_distribution(y_test, "Test")

# %%
le = LabelEncoder()

le.fit(y_train)

y_train = le.transform(y_train)
y_val   = le.transform(y_val)
y_test  = le.transform(y_test)

# %%
from xgboost import XGBClassifier

clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
)

eval_set = [(X_train, y_train), (X_val, y_val)]
clf.fit(
    X_train, y_train,
    eval_set=eval_set,
)

clf.save_model("Baseline_XGboost.json")


# %%
results = clf.evals_result()
train_loss = results['validation_0']['mlogloss']
val_loss = results['validation_1']['mlogloss']

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()


# %%
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

y_score = clf.predict_proba(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - XGBoost on AE feature")
plt.show()


for i, cls in enumerate(le.classes_):
    precision, recall, _ = precision_recall_curve(y_test == i, y_score[:, i])
    ap = average_precision_score(y_test == i, y_score[:, i])
    plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve per Class")
plt.legend()
plt.show()

# %% [markdown]
# 


