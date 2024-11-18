from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import time


def load_and_preprocess_data(root, apply_log_transform=True):
    NB15_1 = pd.read_csv(root + 'UNSW-NB15_1.csv', low_memory=False)
    NB15_2 = pd.read_csv(root + 'UNSW-NB15_2.csv', low_memory=False)
    NB15_3 = pd.read_csv(root + 'UNSW-NB15_3.csv', low_memory=False)
    NB15_4 = pd.read_csv(root + 'UNSW-NB15_4.csv', low_memory=False)
    NB15_features = pd.read_csv(
        root + 'NUSW-NB15_features.csv', encoding='cp1252')

    NB15_1.columns = NB15_features['Name']
    NB15_2.columns = NB15_features['Name']
    NB15_3.columns = NB15_features['Name']
    NB15_4.columns = NB15_features['Name']

    train_df = pd.concat([NB15_1, NB15_2, NB15_3, NB15_4], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df['attack_cat'] = train_df['attack_cat'].fillna(
        'normal').apply(lambda x: x.strip().lower())
    train_df['ct_flw_http_mthd'] = train_df['ct_flw_http_mthd'].fillna(0)
    train_df['is_ftp_login'] = train_df['is_ftp_login'].fillna(0)
    train_df['attack_cat'] = train_df['attack_cat'].replace(
        'backdoors', 'backdoor')

    label_mapping = {
        'analysis': 0, 'backdoor': 1, 'dos': 2, 'exploits': 3,
        'fuzzers': 4, 'generic': 5, 'normal': 6, 'reconnaissance': 7, 'shellcode': 8, 'worms': 9
    }
    train_df['attack_cat'] = train_df['attack_cat'].map(label_mapping)
    train_df = train_df.dropna(subset=['attack_cat'])
    train_df['attack_cat'] = train_df['attack_cat'].astype(int)

    numeric_cols = [
        'sport', 'dsport', 'ct_ftp_cmd', 'Ltime', 'Stime', 'sbytes', 'dbytes',
        'Spkts', 'Dpkts', 'Sload', 'Dload', 'Sjit', 'Djit',
        'tcprtt', 'synack', 'ackdat'
    ]
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])

    if apply_log_transform:
        for col in numeric_cols:
            train_df[col] = np.log1p(train_df[col])

    train_df['duration'] = train_df['Ltime'] - train_df['Stime']
    train_df['byte_ratio'] = train_df['sbytes'] / (train_df['dbytes'] + 1)
    train_df['pkt_ratio'] = train_df['Spkts'] / (train_df['Dpkts'] + 1)
    train_df['load_ratio'] = train_df['Sload'] / (train_df['Dload'] + 1)
    train_df['jit_ratio'] = train_df['Sjit'] / (train_df['Djit'] + 1)
    train_df['tcp_setup_ratio'] = train_df['tcprtt'] / \
        (train_df['synack'] + train_df['ackdat'] + 1)

    columns_to_drop = [
        'sport', 'dsport', 'proto', 'srcip', 'dstip', 'state', 'service',
        'swim', 'dwim', 'stcpb', 'dtcpb', 'Stime', 'Ltime'
    ]
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')

    X = train_df.drop(['attack_cat'], axis=1)
    y = train_df['attack_cat']

    iso = IsolationForest(contamination=0.01, random_state=42)
    y_pred_outliers = iso.fit_predict(X)
    mask = y_pred_outliers != -1
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


# Plot Functions

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.savefig('rf_confusion_matrix.png')
    plt.show()


def plot_roc_auc(y_true, y_pred_proba, num_classes):
    fpr, tpr, roc_auc = {}, {}, {}
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} AUC = {roc_auc[i]:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('rf_roc_auc.png')
    plt.show()


def plot_accuracy_loss(train_accuracies, test_accuracies, train_losses, test_losses):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies,
             label='Validation Accuracy', color='orange')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('rf_accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Validation Loss', color='orange')
    plt.xlabel('Number of Trees')
    plt.ylabel('Loss (1 - Accuracy)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('rf_loss.png')
    plt.show()

# Main Function


def main():
    root = "UNSW-NB15/"
    Xtrain, Xtest, ytrain, ytest = load_and_preprocess_data(root)

    train_accuracies, test_accuracies = [], []
    train_losses, test_losses = [], []
    num_trees = 50

    for n_estimators in range(1, num_trees + 1):
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42)
        rf_model.fit(Xtrain, ytrain)

        y_train_pred = rf_model.predict(Xtrain)
        y_test_pred = rf_model.predict(Xtest)

        train_acc = accuracy_score(ytrain, y_train_pred)
        test_acc = accuracy_score(ytest, y_test_pred)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        train_losses.append(1 - train_acc)
        test_losses.append(1 - test_acc)

    # Final model evaluation
    rf_model = RandomForestClassifier(n_estimators=num_trees, random_state=42)
    rf_model.fit(Xtrain, ytrain)

    y_train_pred = rf_model.predict(Xtrain)
    y_test_pred = rf_model.predict(Xtest)
    y_test_pred_proba = rf_model.predict_proba(Xtest)

    # Final metrics
    print("Final Classification Report:")
    print(classification_report(ytest, y_test_pred, digits=4))

    classes = sorted(np.unique(ytrain))
    plot_confusion_matrix(ytest, y_test_pred, classes)
    plot_roc_auc(ytest, y_test_pred_proba, len(classes))
    plot_accuracy_loss(train_accuracies, test_accuracies,
                       train_losses, test_losses)


if __name__ == "__main__":
    main()
