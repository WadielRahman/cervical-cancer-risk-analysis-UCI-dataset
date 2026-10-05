# Hands-On Deep Learning Pipeline
### Training a Neural Network with scikit-learn (MLPClassifier)

**United International University (UIU)**
**Department of Computer Science and Engineering**
**LAB HANDOUT — PART 4**

| Field | Value |
|---|---|
| Goal | Predict Titanic survival with a neural network |
| Dataset | Titanic |
| Prerequisite | Lab 3 (Machine Learning Pipeline) |
| Model | Multi-layer Perceptron from scikit-learn's neural network module |
| Libraries | pandas, numpy, seaborn, matplotlib, scikit-learn |
| Mode | Hands-on: copy each block, run it, and study the output |

---

## 1. From Machine Learning to Deep Learning

In Part 3 you trained classic models (Logistic Regression, Decision Tree, Random Forest). This lab takes the next step into deep learning by training a neural network on the same Titanic data.

A neural network is a model loosely inspired by the brain. It is built from layers of simple units called neurons. Data enters at the input layer, flows through one or more hidden layers that learn patterns, and produces an answer at the output layer. A network is called "deep" when it has several hidden layers stacked together — hence "deep learning".

> **An honest note on tools:** scikit-learn is a classic-ML library, but it includes a genuine neural network called the `MLPClassifier` (Multi-Layer Perceptron). It is perfect for learning how neural networks work. For large-scale deep learning on images, text or audio, professionals use TensorFlow/Keras or PyTorch (which add GPUs and specialized layers). The concepts you learn here — layers, neurons, activation, epochs, loss — carry directly to those tools.

## 2. Learning Objectives

By the end of this lab, you will be able to:

- Explain the parts of a neural network: neurons, layers, activation, and epochs.
- Prepare and scale data for a neural network (scaling is essential here).
- Build and train a Multi-Layer Perceptron classifier with scikit-learn.
- Evaluate the network and read its training loss curve.
- Experiment with network architecture (width and depth) and interpret the results.
- Bundle scaling and the network into a single Pipeline and make predictions.

## 3. Step 1: Setup and Preprocessing

We load the Titanic data and apply the same preprocessing as before: fill missing values and turn text into numbers.

```python
# ==================================================
# STEP 1 - Setup and preprocessing
# ==================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['survived', 'pclass', 'sex', 'age',
        'sibsp', 'parch', 'fare', 'embarked']
df = sns.load_dataset('titanic')[cols].copy()

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], prefix='emb', dtype=int)

print("Shape:", df.shape, "| Missing:", df.isnull().sum().sum())
print(df.head(3))
```

**Expected output**

```
Shape: (891, 10) | Missing: 0
   survived  pclass  sex   age  sibsp  parch     fare  emb_C  emb_Q  emb_S
0         0       3    0  22.0      1      0   7.2500      0      0      1
1         1       1    1  38.0      1      0  71.2833      1      0      0
2         1       3    1  26.0      0      0   7.9250      0      0      1
```

## 4. Step 2: Split and Scale (Essential for Neural Networks)

We split into train/test sets and then scale the features. For neural networks, scaling is **not optional** — it is essential. The network learns by gradient descent, which struggles badly when features are on very different scales (like age vs fare). Standardizing puts every feature on a comparable footing so training is stable and fast.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
```

**Expected output**

```
Train: (712, 9) Test: (179, 9)
```

> **Remember:** Fit the scaler on the training data only, then apply it to both sets — the same anti-leakage rule from Lab 3.

## 5. Step 3: Build and Train the Neural Network

We create a Multi-Layer Perceptron with two hidden layers: 16 neurons in the first, 8 in the second. Training is the same two-step pattern as any scikit-learn model: create, then fit.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),   # two hidden layers: 16 then 8 neurons
    activation='relu',            # activation function inside neurons
    solver='adam',                # the optimizer that adjusts weights
    max_iter=1000,                # max training epochs
    random_state=42)

mlp.fit(X_train_s, y_train)       # the network learns here
y_pred = mlp.predict(X_test_s)

print("Neural Network accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("Number of layers (input + hidden + output):", mlp.n_layers_)
print("Final training loss: %.4f" % mlp.loss_)
```

**Expected output**

```
Neural Network accuracy: 0.7709
Number of layers (input + hidden + output): 4
Final training loss: 0.3058
```

> **The key settings:** `hidden_layer_sizes=(16, 8)` defines the shape of the network. `activation='relu'` lets neurons model non-linear patterns. `solver='adam'` is the learning algorithm. `max_iter` is the number of training passes (epochs). Four layers = input + 2 hidden + output.

## 6. Step 4: Evaluate the Network

We evaluate exactly as in Part 3, using a confusion matrix and classification report.

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Died','Survived'],
            yticklabels=['Died','Survived'])
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.show()

print(classification_report(y_test, y_pred,
                            target_names=['Died', 'Survived']))
```

**Expected output**

```
[[91 19]
 [22 47]]
              precision    recall  f1-score   support

        Died       0.81      0.83      0.82       110
    Survived       0.71      0.68      0.70        69

    accuracy                           0.77       179
   macro avg       0.76      0.75      0.76       179
weighted avg       0.77      0.77      0.77       179
```

## 7. Step 5: The Training Loss Curve

This is the signature plot of deep learning. During training, the network repeatedly measures its error (the loss) and adjusts its weights to reduce it. Every `MLPClassifier` stores the loss at each epoch in `loss_curve_`. Plotting it shows the network learning.

```python
plt.figure(figsize=(5.6, 3.7))
plt.plot(mlp.loss_curve_, color='#c0392b', lw=1.8)
plt.title('Training Loss Curve (learning over epochs)')
plt.xlabel('epoch (iteration)')
plt.ylabel('training loss')
plt.tight_layout(); plt.show()
```

**Expected output:** a curve like this (steep drop, then flattening).

> **How to read it:** The loss falls steeply at first, meaning the network is learning fast, then flattens as it approaches the best weights it can find. A curve that drops and then levels off is exactly what healthy training looks like. If the curve were still falling steeply at the end, you would train for more epochs.

## 8. Step 6: Experiment with the Architecture

The power of a neural network comes from its architecture — meaning how many layers and neurons it has. Let us try four designs, from a small single-layer network to a deeper three-layer one, and compare.

```python
architectures = {
    '(8,)':        (8,),          # 1 hidden layer,  8 neurons
    '(16,)':       (16,),         # 1 hidden layer, 16 neurons
    '(16, 8)':     (16, 8),       # 2 hidden layers
    '(32, 16, 8)': (32, 16, 8),   # 3 hidden layers (deeper)
}

for name, hl in architectures.items():
    m = MLPClassifier(hidden_layer_sizes=hl, activation='relu',
                      solver='adam', max_iter=1000, random_state=42)
    m.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_s))
    print(f"hidden_layer_sizes={name:14s} -> accuracy {acc:.4f}")
```

**Expected output**

```
hidden_layer_sizes=(8,)           -> accuracy 0.8045
hidden_layer_sizes=(16,)          -> accuracy 0.7989
hidden_layer_sizes=(16, 8)        -> accuracy 0.7709
hidden_layer_sizes=(32, 16, 8)    -> accuracy 0.8045
```

> **Important lesson:** Bigger is not always better. The smallest network here (8 neurons) matches the deepest one, and the two-layer version actually scores lowest. On small, simple tabular data, adding layers gives little benefit and can even hurt. Choosing architecture is experimental — you try designs and compare, exactly as you did here.

## 9. Step 7: When Should You Use Deep Learning?

Compare this lab with Part 3: the Random Forest reached about 82%, while our neural network lands around 77–80%. On this dataset, classic machine learning is just as good or better and much faster to train.

This is a real and important lesson. Deep learning is not automatically superior. Its true strength appears on large, complex, unstructured data such as images, audio, video, and natural language, where classic models struggle and where millions of training examples are available. For small structured tables like Titanic, simpler models from Lab 3 are often the right professional choice.

> **Rule of thumb:** Small tabular data → classic ML (Part 3). Large unstructured data (images, text, speech) with lots of examples → deep learning with Keras or PyTorch. Always start simple, and only reach for a neural network when simpler models are not enough.

## 10. Step 8: Bundle into a Pipeline and Predict

Finally, we wrap the scaler and the network into a single `Pipeline` (as in Lab 3) and use it to predict a new passenger — including the survival probability the network assigns.

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('nn', MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                         solver='adam', max_iter=1000, random_state=42))])
pipe.fit(X_train, y_train)
print("Pipeline accuracy: %.4f" % pipe.score(X_test, y_test))

# predict a new passenger: 1st class female, age 28, fare 80
new = pd.DataFrame([{
    'pclass': 1, 'sex': 1, 'age': 28, 'sibsp': 0, 'parch': 0,
    'fare': 80, 'emb_C': 1, 'emb_Q': 0, 'emb_S': 0}])

pred = pipe.predict(new[X.columns])[0]
prob = pipe.predict_proba(new[X.columns])[0][1]
print("Prediction:", "SURVIVED" if pred == 1 else "DIED")
print("Survival probability: %.1f%%" % (prob * 100))
```

**Expected output**

```
Pipeline accuracy: 0.7709
Prediction: SURVIVED
Survival probability: 96.8%
```

> **Try it yourself:** Change the passenger to a 3rd-class male and re-run — the network's survival probability drops sharply, just as your earlier models predicted.

## 11. Quick Reference: Neural Network Settings

| Setting | What it controls |
|---|---|
| `hidden_layer_sizes` | Number and size of hidden layers, e.g. `(16, 8)` |
| `activation` | Neuron function; `'relu'` is the common default |
| `solver` | Optimizer that updates weights; `'adam'` is standard |
| `max_iter` | Maximum training epochs (passes over the data) |
| `random_state` | Fixes randomness for reproducible results |
| `.loss_curve_` | Training loss at each epoch (for plotting) |
| `.predict_proba()` | Probability the network assigns to each class |

## 12. Lab Exercises

Use the data and network from this lab. Write your code, run it, and paste the output or chart.

1. Retrain the network with `activation='tanh'` instead of `'relu'`. Does the accuracy change? Which activation worked better here?
2. Train a network with a single large hidden layer `(100,)`. Plot its loss curve. Does it reach a lower final loss than the `(16, 8)` network?
3. Set `max_iter=50` and retrain. Look at the loss curve — has the network finished learning, or was it stopped too early? Explain what you see.
4. Compare your best neural network against the Random Forest from Part 3 on the same test set. Which wins, and by how much?
5. Use the pipeline to predict survival for a 3rd-class male, age 40, fare 8. Report the survival probability and compare it with the 1st-class female example.
6. In one short paragraph, explain when you would choose deep learning over the classic models from Part 3, and why it did not help much on the Titanic data.

---

*End of Lab Handout: Hands-On Deep Learning Pipeline with scikit-learn (Part 4 of the Titanic series).*
