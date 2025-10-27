
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("can_bus_data.csv")

label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

features = df.drop(columns=['Class'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=40)
features_pca = pca.fit_transform(features_scaled)

onehot_encoder = OneHotEncoder(sparse=False)
labels_encoded = onehot_encoder.fit_transform(df[['Class']])

X = features_pca.reshape((features_pca.shape[0], 1, features_pca.shape[1]))
y = labels_encoded

# Define LSTM model
def build_lstm_model(params, input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(params['neurons1'], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(params['neurons2']))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Fitness function
def fitness_function(params, X_train, y_train, X_val, y_val):
    model = build_lstm_model(params, X_train.shape[1:], y_train.shape[1])
    model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
    y_pred = model.predict(X_val)
    return accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

# Crow Search Algorithm
def crow_search(X, y, pop_size=5, dim=5, iterations=3, fl=2, AP=0.1):
    population = np.random.rand(pop_size, dim)
    memory = population.copy()
    best_score = -np.inf
    best_params = None

    def decode(ind):
        return {
            'neurons1': int(ind[0] * 500),
            'neurons2': int(ind[1] * 500),
            'batch_size': int(ind[2] * 300 + 32),
            'epochs': int(ind[3] * 50 + 50),
            'learning_rate': ind[4] * 0.0099 + 0.0001
        }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    for _ in range(iterations):
        for i in range(pop_size):
            if np.random.rand() > AP:
                rand_crow = np.random.randint(pop_size)
                new_pos = population[i] + fl * (memory[rand_crow] - population[i])
            else:
                new_pos = np.random.rand(dim)
            new_pos = np.clip(new_pos, 0, 1)
            new_params = decode(new_pos)
            new_fitness = fitness_function(new_params, X_train, y_train, X_val, y_val)
            old_params = decode(population[i])
            old_fitness = fitness_function(old_params, X_train, y_train, X_val, y_val)
            if new_fitness > old_fitness:
                population[i] = new_pos
                memory[i] = new_pos
                if new_fitness > best_score:
                    best_score = new_fitness
                    best_params = new_params
    return best_params, best_score

# Run CSA
best_params, best_score = crow_search(X, y)
print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_score)

# Train and Evaluate Final Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_lstm_model(best_params, X_train.shape[1:], y_train.shape[1])
model.fit(X_train, y_train, batch_size=best_params['batch_size'], epochs=best_params['epochs'], verbose=1)

# Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true_classes, y_pred_classes)
prec = precision_score(y_true_classes, y_pred_classes, average='macro')
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, F1 Score: {f1:.4f}")

# ROC Curve (One-vs-all)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(y.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
