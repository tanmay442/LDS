"""Builder for unit5_deep_learning.ipynb (Unit V part 2: TensorFlow, Keras, PyTorch)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelper import md, code, write_notebook

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "deep-learning" / "unit5_deep_learning.ipynb"

cells = []

# 1. Title / intro / learning objectives
cells.append(md(r"""
# Unit V (Part 2) — TensorFlow, Keras and PyTorch

This notebook is a long-lived reference for the deep-learning portion of the Python Libraries for Data Science syllabus. It starts with the shared vocabulary (tensors, layers, activations, losses, optimizers, backpropagation, epochs, batches, and the overfitting controls dropout and batch normalization), then works through two frameworks. We use **TensorFlow + Keras 3** first: raw tensor ops and `tf.GradientTape`, then the Sequential and Functional model APIs, training loops via `fit`/`evaluate`/`predict`, callbacks, CNNs, and model persistence. We then rebuild the same ideas in **PyTorch**: `torch.Tensor`, autograd, `nn.Module`, an explicit training loop, `DataLoader`, a digits classifier, a small CNN, and `state_dict` persistence. Everything runs **offline** on synthetic arrays or `sklearn.datasets.load_digits`; no MNIST/CIFAR download is performed.

**Learning objectives**

- Define the core deep-learning vocabulary and explain how dropout and batch normalization reduce overfitting.
- Explain the Keras 3 multi-backend design, the TensorFlow backend, and the CPU-only torch build; check installed versions.
- Create and inspect `tf.constant` / `tf.Variable`, use `tf.shape` / `dtype`, basic ops and `tf.random`, and compute gradients with `tf.GradientTape`.
- Build models with the Keras **Sequential** and **Functional** APIs using `Dense` layers and activations.
- Compile models with `compile(loss, optimizer, metrics)` and train/evaluate/predict with `fit`, `evaluate`, `predict` and `validation_split`.
- Use callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`, and describe the `%load_ext tensorboard` workflow.
- Choose correct loss/activation pairings for binary, multiclass, multilabel classification and regression.
- Build a CNN with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, plus `BatchNormalization`; explain transfer learning.
- Save and load Keras models in the `.keras` format.
- Create PyTorch tensors with correct dtype/device, index them, use `.numpy()` / `torch.from_numpy`, and move data to GPU/CPU.
- Explain autograd: `requires_grad`, `backward()`, `.grad`, and `torch.no_grad()`.
- Define models with `nn.Module` / `nn.Linear`, apply `MSELoss` / `CrossEntropyLoss` and `SGD` / `Adam`, and write a complete explicit training loop.
- Use `TensorDataset` and `DataLoader`, train a digits classifier and a small `nn.Conv2d` CNN, do inference under `torch.no_grad()`, and save/load `state_dict`.
- Use `torch.manual_seed` for reproducibility and compare TensorFlow with PyTorch in a summary table.
"""))

# 2. Official documentation & references
cells.append(md(r"""
## Official documentation & references

- [TensorFlow guides](https://www.tensorflow.org/guide) — conceptual guides for the TensorFlow ecosystem.
- [TensorFlow Python API reference](https://www.tensorflow.org/api_docs/python/tf) — authoritative `tf.*` operator docs.
- [Keras getting started](https://keras.io/getting_started/) — first models with the Keras 3 API.
- [Keras guides](https://keras.io/guides/) — Sequential API, Functional API, callbacks, and training.
- [Keras layers API](https://keras.io/api/layers/) — `Dense`, `Conv2D`, `Dropout`, `BatchNormalization` and friends.
- [Keras applications (transfer learning)](https://keras.io/api/applications/) — pretrained model architectures.
- [TensorBoard](https://www.tensorflow.org/tensorboard) — official visualization tool documentation.
- [PyTorch beginner basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — tensors, datasets, models, autograd, optimization.
- [PyTorch stable documentation](https://pytorch.org/docs/stable/index.html) — full API reference.
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — how `requires_grad` and the graph work.
- [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) — the offline dataset used below.
"""))

# 3. Imports (first code cell)
cells.append(code(r"""
from pathlib import Path
DATA = Path("../../data")  # data lives two levels above this notebook

# Core numerics and the two deep-learning frameworks
import numpy as np
import tensorflow as tf
import keras
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Offline dataset helpers — load_digits ships bundled with scikit-learn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Reproducibility across all three random number generators
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

# Scratch directory for checkpoints, logs and saved models used in this notebook
TMP = Path("dl_tmp")
TMP.mkdir(exist_ok=True)

print("Imports ready. Scratch folder:", TMP.resolve())
"""))

# 4. Installation & backend notes
cells.append(md(r"""
## Installation, backends and version checks

**Keras 3** is a multi-backend API: the same `keras` code can run on **TensorFlow**, **JAX**, or **PyTorch**. In this environment the active backend is **TensorFlow** (the `tensorflow` package is installed, so `keras.backend.backend()` reports `"tensorflow"`). `tf.keras` and the standalone `keras` package both expose Keras 3 when TensorFlow ≥ 2.16; we import `keras` directly, which is the recommended Keras 3 style. PyTorch is installed as the **CPU build** (`torch-backend = "cpu"` in `pyproject.toml`), so `torch.cuda.is_available()` is expected to be `False`. CPU builds avoid the very large CUDA wheels and are fine for the tiny models here.

Versions are pinned in `pyproject.toml` as `tensorflow>=2.17`, `keras>=3.5`, `torch>=2.4`, `scikit-learn>=1.5`. Check them with the dunder attributes below.
"""))

# 5. Version checks
cells.append(code(r"""
print("tensorflow:", tf.__version__)
print("keras     :", keras.__version__)
print("torch     :", torch.__version__)
print("keras backend:", keras.backend.backend())
print("CUDA available (torch):", torch.cuda.is_available())
"""))

# 6. Vocabulary
cells.append(md(r"""
## Deep-learning vocabulary

- **Tensor** — a multi-dimensional array. A scalar is rank 0, a vector rank 1, a matrix rank 2, an image batch rank 4 `(batch, height, width, channels)`. Frameworks add shape, dtype, device and (in PyTorch/TF) gradient tracking.
- **Layer** — a callable transform with trainable weights. `Dense` computes `y = activation(x @ W + b)`; `Conv2D` slides learned filters over an image.
- **Activation function** — a non-linearity that lets a network model curved relationships. `relu`, `sigmoid`, `tanh` for hidden units; `softmax` for multiclass output; `linear` (none) for regression. Without non-linearity, stacked layers collapse into one linear map.
- **Loss (objective)** — the scalar a model minimizes: `mse` for regression, `binary_crossentropy` for binary/multilabel, `sparse_categorical_crossentropy` for integer multiclass labels, `categorical_crossentropy` for one-hot multiclass labels.
- **Optimizer** — the rule that updates weights from gradients. `SGD` steps along the negative gradient; `Adam` adapts a per-parameter learning rate and is the usual default.
- **Backpropagation** — the reverse-mode automatic differentiation pass that computes the loss gradient with respect to every weight. Keras hides it inside `fit`; raw TF uses `tf.GradientTape`, raw PyTorch uses `loss.backward()`.
- **Epoch** — one full pass over the training data. **Batch** — a subset of samples whose gradients are averaged into one weight update. **Batch size** controls the number of updates per epoch and the memory used.
- **Overfitting** — training loss keeps falling while validation loss rises; the model memorizes noise. Controls: more data, smaller model, weight decay, early stopping, dropout, batch normalization.
- **Dropout** — during training, randomly set a fraction of activations to zero (e.g. `Dropout(0.3)`). This prevents co-adaptation of neurons; at inference all units are active with scaled outputs.
- **Batch normalization** — normalize layer inputs using batch statistics during training and running statistics at inference; it stabilizes and speeds up training and has a mild regularizing effect.
"""))

# 7. TF constants and variables
cells.append(md(r"""
## TensorFlow basics — `tf.constant` and `tf.Variable`

A `tf.constant` is an immutable tensor; a `tf.Variable` is mutable and is what optimizers update. Both carry a `.shape`, a `.dtype`, and a `.numpy()` view of their value.
"""))

# 8. TF constants/variables code
cells.append(code(r"""
c = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)   # immutable tensor
v = tf.Variable([[1.0, 2.0], [3.0, 4.0]])           # trainable/mutable tensor

print("constant:", c.numpy(), "shape:", c.shape, "dtype:", c.dtype)
print("variable:", v.numpy(), "shape:", v.shape, "dtype:", v.dtype)

# Variables expose assign / assign_add / assign_sub for in-place updates
v.assign_add(tf.constant([[0.5, 0.5], [0.5, 0.5]]))
print("after assign_add:\n", v.numpy())
"""))

# 9. Shapes, dtypes, ops, random
cells.append(md(r"""
### Shapes, dtypes, reshaping, arithmetic and `tf.random`

Use `tf.shape(x)` for a dynamic shape *tensor*, `x.shape` for the static shape, `tf.rank` for the number of axes, `tf.reshape` to change layout and `tf.cast` to change dtype. TensorFlow overloads the usual arithmetic operators and provides equivalents in `tf.math` (`reduce_mean`, `reduce_sum`, `matmul`, `sqrt`, ...). `tf.random` supplies `normal`, `uniform`, and seed control via `tf.random.set_seed`.
"""))

# 10. Shapes/ops/random code
cells.append(code(r"""
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)

print("static shape:", x.shape)             # TensorShape([2, 3])
print("dynamic shape:", tf.shape(x).numpy())  # array([2, 3])
print("rank:", tf.rank(x).numpy())          # 2

y = tf.reshape(x, (3, 2))
print("reshaped:\n", y.numpy())
print("cast to int32:\n", tf.cast(x, tf.int32).numpy())

# Arithmetic: overloading and tf.math equivalents
print("x + 1:\n", (x + 1.0).numpy())
print("x squared:\n", tf.math.square(x).numpy())
print("mean of all entries:", tf.reduce_mean(x).numpy().round(3))
print("matrix product x @ y:\n", tf.matmul(x, y).numpy())

# Random tensors and reproducibility
a = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
b = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype=tf.int32)
print("normal:\n", a.numpy().round(3))
print("uniform ints:\n", b.numpy())
"""))

# 11. GradientTape md
cells.append(md(r"""
### Automatic differentiation with `tf.GradientTape`

Record operations on a `tf.Variable` inside a `with tf.GradientTape() as tape:` block, then call `tape.gradient(loss, variables)`. This is the raw-TF way to do backpropagation and is the foundation of custom training loops. Below we fit `y ≈ w*x + b` by repeatedly subtracting a small multiple of the gradient (gradient descent).
"""))

# 12. GradientTape code
cells.append(code(r"""
x_train = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0])
y_train = tf.constant([1.0, 3.1, 4.9, 7.2, 8.9])  # roughly 2x + 1

weight = tf.Variable(0.0)
bias = tf.Variable(0.0)
learning_rate = 0.05

for epoch in range(60):
    with tf.GradientTape() as tape:                       # record forward pass
        prediction = weight * x_train + bias
        loss = tf.reduce_mean(tf.square(prediction - y_train))  # MSE
    gradients = tape.gradient(loss, [weight, bias])        # backpropagation
    weight.assign_sub(learning_rate * gradients[0])        # gradient descent step
    bias.assign_sub(learning_rate * gradients[1])

print(f"learned weight = {weight.numpy():.3f}, bias = {bias.numpy():.3f} (target ~2.0, ~1.0)")
print("final MSE:", round(float(tf.reduce_mean(tf.square(weight * x_train + bias - y_train))), 4))
"""))

# 13. Keras Sequential md
cells.append(md(r"""
## Keras Sequential API

`keras.Sequential` is a linear stack of layers. Begin with a `keras.Input(shape=...)` layer so the model can build its weights immediately, then add `Dense(units, activation=...)` layers. `model.summary()` prints the architecture and parameter counts. `Dense(1)` is a single output; `Dense(10, activation="softmax")` is a ten-class output.
"""))

# 14. Sequential code
cells.append(code(r"""
sequential_model = keras.Sequential([
    keras.Input(shape=(8,)),                    # 8 input features, no batch dimension here
    keras.layers.Dense(16, activation="relu"),  # hidden layer
    keras.layers.Dense(8, activation="relu"),   # hidden layer
    keras.layers.Dense(1),                      # linear output (regression / logit)
], name="binary_classifier")

sequential_model.summary()
"""))

# 15. Activations + compile md
cells.append(md(r"""
### Activation functions and `compile(loss, optimizer, metrics)`

Activations are available as strings (`activation="relu"`) or callables (`keras.activations.relu`). `compile` configures the three ingredients of training: `optimizer` (e.g. `"adam"` or `keras.optimizers.Adam(learning_rate=0.001)`), `loss`, and `metrics` tracked but **not** used for gradient updates (`["accuracy"]`, `keras.metrics.AUC()`, ...). The loss must match the output activation; the next sections give the pairings.
"""))

# 16. Activations + compile code
cells.append(code(r"""
logits = tf.constant([-2.0, 0.0, 2.0])
print("relu   :", keras.activations.relu(logits).numpy())
print("sigmoid:", keras.activations.sigmoid(logits).numpy().round(3))
print("tanh   :", keras.activations.tanh(logits).numpy().round(3))
print("softmax:", keras.activations.softmax(tf.constant([1.0, 2.0, 3.0])).numpy().round(3))

model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),  # sigmoid -> probability in [0, 1]
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print("Compiled with loss =", model.loss, "| optimizer =", type(model.optimizer).__name__)
"""))

# 17. fit/evaluate/predict md
cells.append(md(r"""
### `fit`, `evaluate`, `predict` and `validation_split`

`fit(x, y, epochs=, batch_size=, validation_split=0.2, verbose=0)` trains and returns a `History` object whose `.history` dict holds per-epoch loss/metric values. `validation_split` holds out the last fraction of the data for monitoring. `evaluate(x, y, verbose=0)` returns `[loss, ...metrics]`. `predict(x, verbose=0)` returns raw model outputs (probabilities for a sigmoid/softmax output). We generate a small synthetic binary dataset so training stays under a second.
"""))

# 18. Binary classification code
cells.append(code(r"""
rng = np.random.default_rng(SEED)
X_bin = rng.normal(size=(400, 8)).astype("float32")
true_w = np.linspace(1, -1, 8)
logits_np = X_bin @ true_w
y_bin = (logits_np > 0).astype("float32")  # binary labels 0/1

history = model.fit(X_bin, y_bin, epochs=6, batch_size=32, validation_split=0.2, verbose=0)
print("History keys:", list(history.history))
print("final training loss / accuracy:",
      round(history.history["loss"][-1], 3), round(history.history["accuracy"][-1], 3))
print("final validation loss / accuracy:",
      round(history.history["val_loss"][-1], 3), round(history.history["val_accuracy"][-1], 3))

loss_value, accuracy_value = model.evaluate(X_bin, y_bin, verbose=0)
print(f"evaluate -> loss {loss_value:.3f}, accuracy {accuracy_value:.3f}")
probabilities = model.predict(X_bin[:5], verbose=0).ravel()
print("predicted probabilities:", probabilities.round(3))
print("predicted classes:", (probabilities > 0.5).astype(int))
"""))

# 19. Callbacks md
cells.append(md(r"""
### Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

`callbacks` are objects invoked at training milestones:

- `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` — stop when validation loss stops improving and roll back to the best epoch.
- `ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True)` — write the best model; Keras 3 requires the `.keras` extension.
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)` — lower the learning rate when progress stalls.
- `TensorBoard(log_dir=...)` — write event files for the TensorBoard dashboard.

**TensorBoard note:** in a Jupyter notebook, load the extension once with `%load_ext tensorboard` and display the dashboard with `%tensorboard --logdir dl_tmp/tb_logs`. These are IPython magics, so they are shown as comments in the code cell to keep it a valid plain-Python script too.
"""))

# 20. Callbacks code
cells.append(code(r"""
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath=str(TMP / "best_binary.keras"), monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    TensorBoard(log_dir=str(TMP / "tb_logs")),
]

callbacks_model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
callbacks_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cb_history = callbacks_model.fit(
    X_bin, y_bin, epochs=10, batch_size=32, validation_split=0.2,
    callbacks=callbacks, verbose=0,
)
print("epochs actually run:", len(cb_history.history["loss"]))
print("best checkpoint exists:", (TMP / "best_binary.keras").exists())

# In a notebook, uncomment the IPython magics to view the dashboard:
# %load_ext tensorboard
# %tensorboard --logdir dl_tmp/tb_logs
print("TensorBoard log directory:", (TMP / "tb_logs").resolve())
"""))

# 21. Functional API md
cells.append(md(r"""
## Keras Functional API

The Functional API builds a graph by calling layers on tensors: `inputs = keras.Input(...)`, `x = layer(inputs)`, `outputs = layer(x)`, `model = keras.Model(inputs, outputs)`. It supports branches and multiple inputs/outputs, which the Sequential API cannot. The model below is the functional equivalent of our sequential binary classifier.
"""))

# 22. Functional API code
cells.append(code(r"""
inputs = keras.Input(shape=(8,), name="features")
hidden = keras.layers.Dense(16, activation="relu", name="dense_1")(inputs)
hidden = keras.layers.Dense(8, activation="relu", name="dense_2")(hidden)
outputs = keras.layers.Dense(1, activation="sigmoid", name="probability")(hidden)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="functional_binary")

functional_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
functional_model.summary()
print("Functional and Sequential parameter counts match:",
      functional_model.count_params() == callbacks_model.count_params())
"""))

# 23. Classification types md
cells.append(md(r"""
### Binary, multiclass and multilabel classification, plus regression

| Problem | Targets | Output layer | Loss |
|---|---|---|---|
| Binary | one 0/1 per sample | `Dense(1, activation="sigmoid")` | `binary_crossentropy` |
| Multiclass (exclusive) | one integer class per sample | `Dense(K, activation="softmax")` | `sparse_categorical_crossentropy` |
| Multiclass (one-hot) | one-hot vector of length K | `Dense(K, activation="softmax")` | `categorical_crossentropy` |
| Multilabel | several 0/1 flags per sample | `Dense(L, activation="sigmoid")` | `binary_crossentropy` |
| Regression | real number(s) | `Dense(n)` (linear) | `mse` / `mae` |

Using the wrong pairing (for example softmax with binary crossentropy on integer labels) trains poorly or errors out, so match them deliberately.
"""))

# 24. Multiclass digits code
cells.append(code(r"""
# sklearn's digits dataset is bundled locally: 1797 images of 8x8 pixels, labels 0-9.
digits = load_digits()
X_digits = digits.data.astype("float32")   # shape (1797, 64)
y_digits = digits.target.astype("int64")   # integer class labels 0-9
print("digits shapes:", X_digits.shape, y_digits.shape, "| classes:", np.unique(y_digits))

X_tr, X_te, y_tr, y_te = train_test_split(X_digits, y_digits, test_size=0.25, random_state=SEED)

multiclass_model = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),  # 10 probabilities summing to 1
])
multiclass_model.compile(optimizer="adam",
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
multiclass_model.fit(X_tr, y_tr, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
multi_loss, multi_acc = multiclass_model.evaluate(X_te, y_te, verbose=0)
print(f"multiclass test loss {multi_loss:.3f}, accuracy {multi_acc:.3f}")
class_probs = multiclass_model.predict(X_te[:3], verbose=0)
print("predicted classes:", class_probs.argmax(axis=1), "| true:", y_te[:3])

# Note: to avoid downloads we do NOT use keras.datasets.mnist (it fetches from the internet).
# If a local CSV is available you can substitute it:
#   import pandas as pd
#   df = pd.read_csv(DATA / "digits.csv")
"""))

# 25. Multilabel code
cells.append(code(r"""
# Multilabel: each sample can carry several independent 0/1 tags.
X_ml = rng.normal(size=(300, 12)).astype("float32")
Y_ml = (X_ml[:, :3] > 0).astype("float32")  # three binary targets per sample

multilabel_model = keras.Sequential([
    keras.Input(shape=(12,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),  # one probability per label
])
multilabel_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
multilabel_model.fit(X_ml, Y_ml, epochs=5, batch_size=32, verbose=0)
label_probabilities = multilabel_model.predict(X_ml[:4], verbose=0)
print("multilabel probabilities:\n", label_probabilities.round(3))
print("thresholded labels:\n", (label_probabilities > 0.5).astype(int))
"""))

# 26. Regression (explanation lives in the required code comments)
cells.append(code(r"""
# Regression: real-valued targets use a LINEAR output (Dense(1), no activation)
# and the "mse" loss (or "mae", which is more robust to outliers). We also track
# "mae" as a metric because it is interpretable in the target's own units.
X_reg = np.linspace(-3, 3, 200).reshape(-1, 1).astype("float32")
y_reg = (2.5 * X_reg.ravel() + 1.0 + rng.normal(0, 0.4, 200)).astype("float32")

reg_model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1),  # linear output for regression
])
reg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05), loss="mse", metrics=["mae"]
)
reg_model.fit(X_reg, y_reg, epochs=60, batch_size=16, verbose=0)
reg_loss, reg_mae = reg_model.evaluate(X_reg, y_reg, verbose=0)
print(f"regression MSE {reg_loss:.3f}, MAE {reg_mae:.3f}")
print("predictions for x = -1, 0, 1:",
      reg_model.predict(np.array([[-1.0], [0.0], [1.0]], dtype="float32"), verbose=0).ravel().round(2))
"""))

# 28. CNN md
cells.append(md(r"""
## Convolutional neural networks, regularization and transfer learning

Images are rank-4 tensors `(batch, height, width, channels)`. A `Conv2D(filters, kernel_size, activation="relu")` learns small filters that detect local patterns; `MaxPooling2D(pool_size)` shrinks spatial size; `Flatten` converts feature maps to a vector; `Dense` layers finish the classifier. Regularization layers: `Dropout(rate)` randomly zeroes activations during training, and `BatchNormalization()` normalizes activations per batch (using running statistics at inference). We build a CNN on **synthetic** 28×28 grayscale images — no download needed.

**Transfer learning** reuses a network pretrained on a huge dataset (e.g. `keras.applications.VGG16`). Typical recipe: load a backbone with `weights="imagenet", include_top=False`, freeze it (`backbone.trainable = False`), add a new `Dense` head for your classes, train the head, then optionally fine-tune a few backbone layers with a small learning rate. We avoid instantiating/downloads of pretrained weights here; explore it in the official Keras Applications docs.
"""))

# 29. CNN code
cells.append(code(r"""
# Synthetic grayscale image batch: 120 images of 28x28 pixels, one channel.
X_img = rng.normal(size=(120, 28, 28, 1)).astype("float32")
# A learnable but trivial pattern: label 1 if the top-left quadrant average is positive.
y_img = (X_img[:, :14, :14, :].mean(axis=(1, 2, 3)) > 0).astype("float32")

cnn_model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(8, kernel_size=3, activation="relu"),
    keras.layers.BatchNormalization(),            # normalize activations each batch
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),                    # randomly drop 30% of units while training
    keras.layers.Dense(1, activation="sigmoid"),
], name="tiny_cnn")
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_model.summary()
cnn_model.fit(X_img, y_img, epochs=2, batch_size=16, verbose=0)
print("CNN evaluate:", [round(value, 3) for value in cnn_model.evaluate(X_img, y_img, verbose=0)])

# Transfer-learning skeleton (commented so it never downloads weights):
# backbone = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# backbone.trainable = False
# model = keras.Sequential([backbone, keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")])
"""))

# 30. Save/load Keras md
cells.append(md(r"""
## Saving and loading Keras models

Keras 3's native format is a single `.keras` file (a ZIP carrying config + weights). `model.save("path.keras")` writes it and `keras.models.load_model("path.keras")` restores the full model, including the compiled optimizer state. Use `model.export("folder")` for a TensorFlow SavedModel for serving, and `model.get_weights()` / `model.set_weights()` when you only need the weights as NumPy arrays.
"""))

# 31. Save/load Keras code
cells.append(code(r"""
keras_path = TMP / "multiclass_model.keras"
multiclass_model.save(keras_path)
reloaded_model = keras.models.load_model(keras_path)

reloaded_loss, reloaded_acc = reloaded_model.evaluate(X_te, y_te, verbose=0)
print("reloaded model accuracy:", round(reloaded_acc, 3), "| matches original:",
      np.isclose(reloaded_acc, multi_acc))
print("saved file size (bytes):", keras_path.stat().st_size)
"""))

# 32. PyTorch tensors md
cells.append(md(r"""
## PyTorch — tensors, dtype, device and NumPy interop

A `torch.Tensor` is like a NumPy array plus dtype, device and autograd support. Create with `torch.tensor(list)`, `torch.randn(...)`, `torch.zeros(...)`, `torch.arange(...)`. Inspect `.shape`, `.dtype`, `.device`. Change type with `.float()` / `.long()` or `.to(dtype=...)`. Move to hardware with `.to(device)` where `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Index with standard `x[row, col]` / slices, combine with `+`, `*`, `@` (or `torch.matmul`). On CPU, `.numpy()` shares memory with a NumPy array; `torch.from_numpy(array)` does the reverse (also shares memory).
"""))

# 33. PyTorch tensors code
cells.append(code(r"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
r = torch.randn(2, 3, device=device)
print("tensor:\n", t, "\nshape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device)
print("index [0,1]:", t[0, 1].item(), "| first row:", t[0].tolist())
print("t + 1:\n", (t + 1).cpu().numpy())
print("t @ t.T:\n", (t @ t.T).cpu().numpy())

array = np.arange(6, dtype=np.float32).reshape(2, 3)
from_numpy_tensor = torch.from_numpy(array)      # NumPy -> torch
back_to_numpy = from_numpy_tensor.numpy()        # torch (CPU) -> NumPy
print("round-trip equal:", np.array_equal(array, back_to_numpy))

# dtype conversion matters for losses: CrossEntropyLoss wants int64 labels
labels = torch.tensor([0, 1, 2], dtype=torch.int64)
print("labels dtype:", labels.dtype)
"""))

# 34. Autograd md
cells.append(md(r"""
## PyTorch autograd

Set `requires_grad=True` on a tensor (or use a layer's parameters, which require grad by default). PyTorch records a computation graph; calling `loss.backward()` populates each leaf tensor's `.grad`. Wrap inference or data manipulation in `with torch.no_grad():` to disable graph tracking (saves memory). Use `tensor.detach()` to obtain a tensor sharing data but detached from the graph. Gradients accumulate, so optimizer loops call `optimizer.zero_grad()` before each `backward()`.
"""))

# 35. Autograd code
cells.append(code(r"""
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()  # dy/dx = 2x + 2 = 8 at x = 3
print("y:", y.item(), "| x.grad:", x.grad.item())

# Vector outputs need a scalar to backward from; y.sum().backward() is the usual trick.
vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(vector ** 2).sum().backward()
print("gradients of squares:", vector.grad.tolist())

# Inference: no graph is built, so no gradients are stored.
with torch.no_grad():
    inference_result = x ** 2
print("no_grad result:", inference_result.item(), "| requires_grad:", inference_result.requires_grad)
print("detached:", x.detach().item())
"""))

# 36. nn.Module / losses / optimizers md
cells.append(md(r"""
## `nn.Module`, loss functions and optimizers

Subclass `torch.nn.Module`: define layers in `__init__` (calling `super().__init__()`), implement `forward(x)`. `nn.Linear(in_features, out_features)` is the dense layer; it holds `weight` and `bias` as parameters. Common losses: `nn.MSELoss()` for regression, `nn.CrossEntropyLoss()` for multiclass (takes **raw logits** and integer class labels; it applies log-softmax internally), `nn.BCELoss()`/`nn.BCEWithLogitsLoss()` for binary. Optimizers live in `torch.optim` (`SGD`, `Adam`) and receive `model.parameters()`.
"""))

# 37. nn.Module code
cells.append(code(r"""
class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

torch_model = TinyRegressor().to(device)
print(torch_model)
print("parameters:", [(name, tuple(p.shape)) for name, p in torch_model.named_parameters()])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
print("loss:", type(criterion).__name__, "| optimizer:", type(optimizer).__name__)
"""))

# 38. Explicit training loop md
cells.append(md(r"""
## An explicit PyTorch training loop

Unlike Keras `fit`, PyTorch training is written by hand. The canonical sequence per batch is:

1. `optimizer.zero_grad()` — clear stale gradients.
2. `prediction = model(xb)` — forward pass.
3. `loss = criterion(prediction, yb)` — compute loss.
4. `loss.backward()` — backpropagate.
5. `optimizer.step()` — update parameters.

Wrap the epoch loop in `model.train()` (enables dropout/BN training behaviour) and evaluation in `model.eval()` with `torch.no_grad()`.
"""))

# 39. Training loop code
cells.append(code(r"""
X_t = torch.tensor(X_reg, dtype=torch.float32, device=device)       # reuse regression data
y_t = torch.tensor(y_reg.reshape(-1, 1), dtype=torch.float32, device=device)

torch_model = TinyRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.05)

for epoch in range(80):
    torch_model.train()
    optimizer.zero_grad()
    prediction = torch_model(X_t)
    loss = criterion(prediction, y_t)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

torch_model.eval()
with torch.no_grad():
    final_prediction = torch_model(X_t)
print("final MSE:", round(nn.MSELoss()(final_prediction, y_t).item(), 4))
"""))

# 40. DataLoader md
cells.append(md(r"""
## `TensorDataset` and `DataLoader`

`TensorDataset(x, y)` pairs feature and label tensors. `DataLoader(dataset, batch_size=, shuffle=True)` yields mini-batches, handles shuffling and parallel loading, and is the standard way to feed data to a PyTorch model. Here `shuffle=True` applies to training data; keep validation/test loaders unshuffled.
"""))

# 41. DataLoader code
cells.append(code(r"""
dataset = TensorDataset(X_t, y_t)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("number of batches:", len(loader))
for batch_features, batch_targets in loader:
    print("one batch ->", tuple(batch_features.shape), tuple(batch_targets.shape))
    break  # show just the first batch
"""))

# 42. Digits classifier md
cells.append(md(r"""
## A PyTorch classifier on the digits dataset

We reuse the local `load_digits` arrays: 64 input features, 10 classes. The network ends in `nn.Linear(32, 10)` producing **logits** (no softmax) and uses `nn.CrossEntropyLoss()`, which expects raw logits plus `int64` class indices. Accuracy is the fraction of `argmax(logits)` matching the labels, computed under `torch.no_grad()`.
"""))

# 43. Digits classifier code
cells.append(code(r"""
Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
ytr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)
Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
yte_t = torch.tensor(y_te, dtype=torch.int64, device=device)

train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

class DigitsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)

digits_net = DigitsClassifier().to(device)
ce_loss = nn.CrossEntropyLoss()
digits_optimizer = torch.optim.Adam(digits_net.parameters(), lr=0.01)

for epoch in range(6):
    digits_net.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        digits_optimizer.zero_grad()
        logits = digits_net(batch_x)
        loss = ce_loss(logits, batch_y)
        loss.backward()
        digits_optimizer.step()
        epoch_loss += loss.item() * len(batch_y)
    print(f"epoch {epoch + 1} | loss {epoch_loss / len(Xtr_t):.4f}")

digits_net.eval()
with torch.no_grad():
    test_logits = digits_net(Xte_t)
    predicted = test_logits.argmax(dim=1)
    accuracy = (predicted == yte_t).float().mean().item()
print(f"PyTorch digits test accuracy: {accuracy:.3f} | sample predictions:", predicted[:8].tolist())
"""))

# 44. PyTorch CNN md
cells.append(md(r"""
## A small convolutional network in PyTorch (`nn.Conv2d`)

PyTorch image tensors use channel-first layout `(batch, channels, height, width)`, unlike Keras's channel-last `(batch, height, width, channels)`. The digits images are 8×8, so reshape each sample to `(1, 8, 8)`. `nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)` preserves spatial size with padding; `nn.MaxPool2d(2)` halves it. After two poolings an 8×8 map becomes 2×2, so `nn.Flatten()` yields `16 * 2 * 2 = 64` features before the final `nn.Linear(64, 10)`.
"""))

# 45. PyTorch CNN code
cells.append(code(r"""
X_img_t = torch.tensor(X_digits.reshape(-1, 1, 8, 8), dtype=torch.float32, device=device)
y_img_t = torch.tensor(y_digits, dtype=torch.int64, device=device)

class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1->16 channels, 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 8x8 -> 4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                               # 32 * 2 * 2 = 128
            nn.Linear(32 * 2 * 2, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

conv_net = TinyConvNet().to(device)
print(conv_net)
sample_logits = conv_net(X_img_t[:4])
print("logits for 4 images:", tuple(sample_logits.shape))  # (4, 10)
"""))

# 46. state_dict / seed / device md
cells.append(md(r"""
## Reproducibility, inference and `state_dict` persistence

`torch.manual_seed(42)` seeds the CPU generator (add `torch.cuda.manual_seed_all(42)` for GPU). Device portability uses `device = torch.device(...)` and `.to(device)` on both model and tensors. The recommended save format is the **`state_dict`** — a mapping of layer names to parameter tensors — via `torch.save(model.state_dict(), "file.pt")`; restore with `model.load_state_dict(torch.load("file.pt", weights_only=True))`. Saving the entire model with `torch.save(model, ...)` is discouraged because it pickles class definitions. For inference use `model.eval()` plus `torch.no_grad()` and `torch.softmax(logits, dim=1)` to get probabilities.
"""))

# 47. state_dict / seed / device code
cells.append(code(r"""
torch.manual_seed(SEED)
replica = DigitsClassifier().to(device)

# Save and load the learned parameters only
state_path = TMP / "digits_classifier.pt"
torch.save(digits_net.state_dict(), state_path)
replica.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
replica.eval()
print("state_dict restored:", state_path.exists())

# Deterministic inference: no graph is built, saturated softmax gives probabilities.
with torch.no_grad():
    logits = replica(Xte_t[:5])
    probabilities_torch = torch.softmax(logits, dim=1)
print("predicted classes:", logits.argmax(dim=1).tolist())
print("max class probability:", [round(float(value), 3) for value in probabilities_torch.max(dim=1).values])

# Verify identical predictions from the saved copy
with torch.no_grad():
    same = torch.equal(replica(Xte_t).argmax(1), digits_net(Xte_t).argmax(1))
print("replica matches original:", same)
"""))

# 48. Comparison table md
cells.append(md(r"""
## TensorFlow/Keras vs PyTorch — summary

| Aspect | TensorFlow / Keras 3 | PyTorch |
|---|---|---|
| Philosophy | High-level API first; Keras abstracts the training loop | Explicit, Pythonic control; you write the loop |
| Tensors | `tf.Tensor`, `tf.Variable`; static shapes emphasized | `torch.Tensor`; dynamic shapes, numpy-like |
| Autograd | `tf.GradientTape()` context manager | `loss.backward()` populates `.grad` |
| Model definition | `Sequential` / Functional `keras.Model` | Subclass `nn.Module`, implement `forward` |
| Layer | `keras.layers.Dense`, `Conv2D` | `nn.Linear`, `nn.Conv2d` (channel-first) |
| Compile / loss | `model.compile(loss=, optimizer=, metrics=)` | Pick `nn.*Loss` and `torch.optim.*` yourself |
| Training | `model.fit(..., callbacks=...)` | Manual loop: zero_grad → forward → loss → backward → step |
| Data pipeline | NumPy arrays, `tf.data.Dataset` | `TensorDataset` + `DataLoader` |
| Callbacks / logging | Rich callbacks, TensorBoard integration | Hand-written hooks; TensorBoard via `torch.utils.tensorboard` |
| Image layout | channel-last `(N, H, W, C)` | channel-first `(N, C, H, W)` |
| Deployment | SavedModel / `.keras`, TFLite, TF Serving | TorchScript, `torch.export`, state_dict |
| Best for | Fast prototyping, production serving | Research, custom training logic, debugging |
"""))

# 49. Exercise 1
cells.append(md(r"""
## Try it yourself — Exercise 1

Using **raw TensorFlow**, fit the line `y = 3x - 2` from `x = [0, 1, 2, 3, 4]` with `tf.Variable`s for slope and intercept. Use `tf.GradientTape` to compute MSE gradients and run 80 gradient-descent steps with learning rate `0.05`. Print the learned slope, intercept and final MSE. (Target slope ≈ 3, intercept ≈ −2.)
"""))

# 50. Solution 1
cells.append(code(r"""
# Solution 1 — fit y = 3x - 2 with raw TensorFlow and GradientTape
x_ex = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0])
y_ex = tf.constant([-2.0, 1.0, 4.0, 7.0, 10.0])  # 3x - 2

slope = tf.Variable(0.0)
intercept = tf.Variable(0.0)
for _ in range(80):
    with tf.GradientTape() as tape:
        mse = tf.reduce_mean(tf.square((slope * x_ex + intercept) - y_ex))
    d_slope, d_intercept = tape.gradient(mse, [slope, intercept])
    slope.assign_sub(0.05 * d_slope)
    intercept.assign_sub(0.05 * d_intercept)

print(f"slope {slope.numpy():.3f}, intercept {intercept.numpy():.3f}, "
      f"MSE {float(mse):.4f}")
"""))

# 51. Exercise 2
cells.append(md(r"""
## Try it yourself — Exercise 2

Build a **Keras** binary classifier on a fresh synthetic dataset with two informative features (`X`, shape `(300, 2)`, label 1 when `X[:, 0] + X[:, 1] > 0`). Use a `Sequential` model with one hidden `Dense(8, activation="relu")` and a sigmoid output, compile with `adam` / `binary_crossentropy` / `accuracy`, and fit for 10 epochs with `validation_split=0.2` and an `EarlyStopping(patience=2, restore_best_weights=True)` callback. Report `evaluate` accuracy.
"""))

# 52. Solution 2
cells.append(code(r"""
# Solution 2 — Keras binary classifier with EarlyStopping
rng2 = np.random.default_rng(SEED)
X_ex2 = rng2.normal(size=(300, 2)).astype("float32")
y_ex2 = (X_ex2[:, 0] + X_ex2[:, 1] > 0).astype("float32")

ex2_model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
ex2_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ex2_model.fit(
    X_ex2, y_ex2, epochs=10, batch_size=16, validation_split=0.2, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
)
print("Exercise 2 evaluate accuracy:", round(ex2_model.evaluate(X_ex2, y_ex2, verbose=0)[1], 3))
"""))

# 53. Exercise 3
cells.append(md(r"""
## Try it yourself — Exercise 3

In **PyTorch**, train a small regressor on `X = [[0],[1],[2],[3],[4]]`, `y = 5x + 1` using an `nn.Linear(1, 1)` model, `nn.MSELoss()` and `torch.optim.SGD(lr=0.02)` for 200 full-batch steps. Reset gradients, backpropagate and step each iteration, then print the learned `weight` and `bias` (target ≈ 5 and ≈ 1). Do the final prediction under `torch.no_grad()`.
"""))

# 54. Solution 3
cells.append(code(r"""
# Solution 3 — PyTorch linear regression with an explicit training loop
X_ex3 = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_ex3 = torch.tensor([[1.0], [6.0], [11.0], [16.0], [21.0]], dtype=torch.float32)  # 5x + 1

linear_model = nn.Linear(1, 1)
mse_loss = nn.MSELoss()
sgd = torch.optim.SGD(linear_model.parameters(), lr=0.02)
for _ in range(200):
    sgd.zero_grad()
    loss = mse_loss(linear_model(X_ex3), y_ex3)
    loss.backward()
    sgd.step()

weight_value = linear_model.weight.item()
bias_value = linear_model.bias.item()
with torch.no_grad():
    sample_prediction = linear_model(torch.tensor([[2.0]])).item()
print(f"weight {weight_value:.3f}, bias {bias_value:.3f}, "
      f"prediction at x=2: {sample_prediction:.3f} (target 11)")
"""))

# 55. Cheat sheet
cells.append(md(r"""
## Cheat sheet / key takeaways

| Task | TensorFlow / Keras | PyTorch |
|---|---|---|
| Version | `tf.__version__`, `keras.__version__` | `torch.__version__` |
| Backend | `keras.backend.backend()` → `"tensorflow"` | CPU build: `torch.cuda.is_available()` is `False` |
| Constant / variable | `tf.constant(x)`, `tf.Variable(x)` | `torch.tensor(x)`, `torch.randn(...)` |
| Shape / dtype | `tf.shape(x)`, `x.dtype`, `tf.cast` | `x.shape`, `x.dtype`, `.float()` / `.long()` |
| Random + seed | `tf.random.normal`, `tf.random.set_seed` | `torch.randn`, `torch.manual_seed` |
| Gradients | `with tf.GradientTape() as t: ...; t.gradient(loss, vars)` | `loss.backward()`; read `param.grad` |
| Model | `keras.Sequential([...])` / `keras.Model(inputs, outputs)` | subclass `nn.Module`, define `forward` |
| Dense layer | `keras.layers.Dense(units, activation=...)` | `nn.Linear(in, out)` + activation |
| Loss + optimizer | `model.compile(loss=, optimizer=, metrics=)` | `nn.MSELoss()`, `nn.CrossEntropyLoss()`, `torch.optim.Adam/SGD` |
| Train | `model.fit(x, y, epochs=, batch_size=, validation_split=, callbacks=)` | loop: `zero_grad → forward → loss → backward → step` |
| Evaluate / predict | `model.evaluate`, `model.predict` | `model.eval()` + `with torch.no_grad():` |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard | manual logic / hooks |
| Batches | NumPy arrays or `tf.data.Dataset` | `TensorDataset` + `DataLoader` |
| CNN layers | `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, `BatchNormalization` | `nn.Conv2d`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Dropout`, `nn.BatchNorm2d` |
| Image layout | `(N, H, W, C)` channel-last | `(N, C, H, W)` channel-first |
| Save / load | `model.save("m.keras")`, `keras.models.load_model("m.keras")` | `torch.save(model.state_dict(), "m.pt")`, `model.load_state_dict(torch.load("m.pt", weights_only=True))` |
| Classification pairings | binary→sigmoid+`binary_crossentropy`; multiclass→softmax+`sparse_categorical_crossentropy`; multilabel→sigmoid+`binary_crossentropy` | binary→`BCEWithLogitsLoss`; multiclass→`CrossEntropyLoss` on logits |
| Regression | linear output + `mse`/`mae` | `nn.Linear` output + `nn.MSELoss()` |

**Rules of thumb:** match output activation to loss; hold out validation data to catch overfitting; use dropout + batch normalization + early stopping as first-line regularizers; keep a fixed seed; save checkpoints; never download datasets in an offline notebook — use `load_digits` or synthetic arrays.
"""))

# 56. Further reading
cells.append(md(r"""
## Further reading

- [TensorFlow Keras classification tutorial](https://www.tensorflow.org/tutorials/keras/classification) — end-to-end classifier with the official API.
- [TensorFlow automatic differentiation guide](https://www.tensorflow.org/guide/autodiff) — deeper look at `GradientTape`.
- [Keras Functional API guide](https://keras.io/guides/functional_api/) — branches and multi-input models.
- [Keras callbacks guide](https://keras.io/guides/callbacks/) — writing custom callbacks.
- [Keras examples gallery](https://keras.io/examples/) — vetted reference models.
- [PyTorch learning basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — the official beginner series.
- [PyTorch `nn.Module` reference](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — the base class for all models.
- [PyTorch autograd reference](https://pytorch.org/docs/stable/autograd.html) — graph internals and `no_grad`.
- [PyTorch TensorBoard tutorial](https://pytorch.org/tutorials/beginner/former_torchies/tensorboard_tutorial.html) — logging with `SummaryWriter`.
- [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) — details of the offline dataset used throughout.
"""))

write_notebook(OUT, cells)
