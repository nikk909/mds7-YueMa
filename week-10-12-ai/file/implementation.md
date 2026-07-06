# Green AI Trade-Off — Implementation Guide
# 绿色 AI 权衡实验 — 实现指南

本文档严格遵循 [`task.md`](./task.md) 的结构，逐一说明各部分的实现思路、变量类型与代码示例。

**格式说明：** 每个大节均按 **English → 中文 → Code** 组织。

---

## 1. Objective | 目标

**English:** In this lab, you will act as a Lead MLOps Engineer tasked with optimizing a model for a production environment where cloud computing budgets and carbon emissions are strictly monitored. You must determine if utilizing the full dataset and training for extended epochs provides a justifiable Return on Investment (ROI) in terms of accuracy vs. carbon emitted.

**中文：** 在本实验中，你将扮演首席 MLOps 工程师，负责在云计算预算与碳排放均受严格监控的生产环境中优化模型。你需要判断：使用完整数据集并延长训练轮数，是否在「准确率 vs. 碳排放」方面具有合理的投资回报率（ROI）。

**Implementation goal | 实现目标：**

```python
# We build 4 MLP models from scratch, each with different data size & epochs,
# track CO2 via CodeCarbon, and compare accuracy vs. emissions.
```

---

## 2. Architectural Requirements | 架构要求

### 2.1 Dataset | 数据集

**English:** Choose any Computer Vision dataset (e.g., Fashion-MNIST, CIFAR-10, SVHN).  
**Hint:** `tf.keras.datasets` has many built-in options.

**中文：** 任选一种计算机视觉数据集（例如 Fashion-MNIST、CIFAR-10、SVHN）。  
**提示：** `tf.keras.datasets` 提供多种内置数据集。

**Recommended | 推荐：** Fashion-MNIST（28×28 灰度、10 类，训练快，符合 CV 要求）

```python
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

**Preprocessing | 预处理：**

```python
# Normalize to [0, 1]
X_train_full = X_train_full.astype("float32") / 255.0
X_test       = X_test.astype("float32") / 255.0

# Add channel dimension for MLP input shape (28, 28, 1)
X_train_full = np.expand_dims(X_train_full, axis=-1)
X_test       = np.expand_dims(X_test, axis=-1)

# Labels: keep as integer (N,); use loss="sparse_categorical_crossentropy"
num_classes  = 10
input_shape  = (28, 28, 1)
```

| Variable | Type | Shape / Notes |
|----------|------|---------------|
| `X_train_full` | `np.ndarray` | `(60000, 28, 28, 1)` |
| `y_train_full` | `np.ndarray` | `(60000,)`, `dtype=int` |
| `X_test` | `np.ndarray` | `(10000, 28, 28, 1)` |
| `y_test` | `np.ndarray` | `(10000,)` |
| `num_classes` | `int` | `10` |
| `input_shape` | `tuple` | `(28, 28, 1)` |

---

### 2.2 Model Architecture | 模型架构

**English:** You can use a CNN (`Conv2D` layers). You **must** build a Deep Multi-Layer Perceptron (MLP).

- **Requirement:** 1 `Flatten()` layer followed by at least 5 `Dense()` hidden layers, ending with your output classification layer.

**中文：** 可以使用 CNN（`Conv2D` 层），但**必须**构建一个深度多层感知机（MLP）。

- **要求：** 1 个 `Flatten()` 层，后接至少 5 个 `Dense()` 隐藏层，最后是输出分类层。

**Example architecture (Fashion-MNIST):**

```text
Input (28, 28, 1)
  → Flatten()                     # 784 features
  → Dense(512, activation="relu") # hidden 1
  → Dense(256, activation="relu") # hidden 2
  → Dense(128, activation="relu") # hidden 3
  → Dense(64,  activation="relu") # hidden 4
  → Dense(32,  activation="relu") # hidden 5
  → Dense(num_classes, activation="softmax")
```

**Encapsulated function | 封装函数：**

```python
def build_mlp(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Build a deep MLP with Flatten + ≥5 Dense hidden layers."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model
```

| Variable | Type | Notes |
|----------|------|-------|
| `model` | `tf.keras.Model` | new instance per experiment |
| `input_shape` | `tuple` | e.g. `(28, 28, 1)` |
| `num_classes` | `int` | e.g. `10` |

---

### 2.3 Tracking | 追踪

**English:** You must wrap your training loops using the **codecarbon** library's `EmissionsTracker`.

**中文：** 必须使用 **codecarbon** 库的 `EmissionsTracker` 包裹训练循环，以追踪碳排放。

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="green-ai-tradeoff",
    output_file="emissions.csv",
    log_level="error",          # suppress verbose logging
)
tracker.start()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=1)
emissions_g = tracker.stop()    # float, grams CO2
```

| Variable | Type | Notes |
|----------|------|-------|
| `tracker` | `EmissionsTracker` | CodeCarbon instance |
| `emissions_g` | `float` | grams CO₂ from `tracker.stop()` |
| `emissions.csv` | file | appended per run |

**Important | 重要：** `emissions.csv` accumulates rows across experiments — do not delete between runs.

---

## 3. The Experiment Matrix | 实验矩阵

**English:** You must write a Python script that trains **4 separate models from scratch** using the following configurations:

| Experiment | Training Data | Epochs |
|------------|---------------|--------|
| **A** | 50% of training data | 50 |
| **B** | 50% of training data | 100 |
| **C** | 100% of training data | 50 |
| **D** | 100% of training data | 100 |

**中文：** 你必须编写 Python 脚本，**从零开始**训练 **4 个独立模型**，配置如下：

| 实验 | 训练数据 | 训练轮数 |
|------|----------|----------|
| **A** | 50% 训练集 | 50 |
| **B** | 50% 训练集 | 100 |
| **C** | 100% 训练集 | 50 |
| **D** | 100% 训练集 | 100 |

### 3.1 Config Definition | 配置定义

```python
EXPERIMENTS: list[dict] = [
    {"name": "A", "data_fraction": 0.5, "epochs": 50},
    {"name": "B", "data_fraction": 0.5, "epochs": 100},
    {"name": "C", "data_fraction": 1.0, "epochs": 50},
    {"name": "D", "data_fraction": 1.0, "epochs": 100},
]
```

### 3.2 Data Slicing | 数据切片

**Hint | 提示：** To get 50% of the data, use slicing:

```python
n_half = len(X_train_full) // 2
X_train_half = X_train_full[:n_half]
y_train_half = y_train_full[:n_half]
```

**Encapsulated function | 封装函数：**

```python
def get_train_data(
    X_full: np.ndarray,
    y_full: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice training data by fraction."""
    n = int(len(X_full) * fraction)
    return X_full[:n], y_full[:n]
```

| Variable | Type | Notes |
|----------|------|-------|
| `n_half` | `int` | `len(X_train_full) // 2` |
| `X_train` | `np.ndarray` | current experiment training features |
| `y_train` | `np.ndarray` | current experiment training labels |
| `data_fraction` | `float` | `0.5` or `1.0` |
| `epochs` | `int` | `50` or `100` |

**Test set | 测试集：** Always use full `X_test`, `y_test` for all 4 experiments — **never change the test set**.

### 3.3 Experiment Loop | 实验循环

**English:** For each configuration: select data → build fresh model → compile → train under CodeCarbon → evaluate.

**中文：** 每组实验：选择数据 → 新建模型 → 编译 → CodeCarbon 追踪训练 → 评估。

```python
def run_experiment(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    input_shape: tuple,
    num_classes: int,
) -> dict[str, str | int | float]:
    """Run one experiment: build, train (tracked), evaluate, return metrics."""
    model = build_mlp(input_shape, num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    tracker = EmissionsTracker(
        project_name="green-ai-tradeoff",
        output_file="emissions.csv",
        log_level="error",
    )
    tracker.start()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=1)
    emissions_g = tracker.stop()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        "experiment": name,
        "data_fraction": len(X_train) / (len(X_train) + len(X_train) if data_fraction < 1 else 1),  # simplified
        "epochs": epochs,
        "train_samples": len(X_train),
        "test_accuracy": test_accuracy,
        "test_accuracy_pct": round(test_accuracy * 100, 2),
        "emissions_g": round(emissions_g, 4),
    }
```

| Variable | Type | Notes |
|----------|------|-------|
| `history` | `tf.keras.callbacks.History` | return value of `model.fit` |
| `test_loss` | `float` | first value from `model.evaluate` |
| `test_accuracy` | `float` | 0–1, second value from `evaluate` |
| `test_accuracy_pct` | `float` | `test_accuracy * 100` |

**Core principle | 核心原则：** Each experiment must **build and train a model from scratch** — never reuse trained weights across experiments.

---

## 4. Deliverables & Visualization | 交付物与可视化

**English:** Your final Python notebook/script must automatically generate a plot (or subplots) that clearly visualizes the trade-offs. Your plot should display:

- The 4 experiments on the X-axis (or grouped logically)
- **Test Accuracy (%)** for each experiment
- **CO₂ Emissions (grams)** for each experiment

**中文：** 最终的 Python Notebook/脚本须自动生成清晰展示权衡关系的图表（或子图），应包含：

- X 轴上的 4 组实验（或逻辑分组）
- 各实验的**测试准确率（%）**
- 各实验的**二氧化碳排放量（克）**

### 4.1 Aggregate Results | 汇总结果

```python
results: list[dict[str, str | int | float]] = []

def main() -> None:
    X_train_full, y_train_full, X_test, y_test, input_shape, num_classes = load_and_preprocess_data()
    results = []
    for cfg in EXPERIMENTS:
        X_train, y_train = get_train_data(X_train_full, y_train_full, cfg["data_fraction"])
        result = run_experiment(
            name=cfg["name"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=cfg["epochs"],
            input_shape=input_shape,
            num_classes=num_classes,
        )
        results.append(result)
    results_df = pd.DataFrame(results)
    plot_tradeoff(results_df)
```

**Single experiment result | 单次实验结果：**

```python
result: dict[str, str | int | float] = {
    "experiment": "A",
    "data_fraction": 0.5,
    "epochs": 50,
    "train_samples": 30000,
    "test_accuracy": 0.88,
    "test_accuracy_pct": 88.0,
    "emissions_g": 12.34,
}
```

| Variable | Type | Notes |
|----------|------|-------|
| `results` | `list[dict]` | list of 4 experiment result dicts |
| `results_df` | `pd.DataFrame` | columns: experiment, data_fraction, epochs, train_samples, test_accuracy, test_accuracy_pct, emissions_g |

### 4.2 Plotting | 绘图

**English:** Generate a chart with dual y-axes or two subplots comparing accuracy (%) and CO₂ (g).

**中文：** 生成双 y 轴图表或两个子图，对比准确率（%）和 CO₂（g）。

```python
import matplotlib.pyplot as plt

def plot_tradeoff(results_df: pd.DataFrame, save_path: str = "tradeoff_plot.png") -> None:
    """Plot accuracy (%) and CO₂ (g) side by side or on twin axes."""
    labels = results_df["experiment"].tolist()
    acc    = results_df["test_accuracy_pct"].tolist()
    co2    = results_df["emissions_g"].tolist()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar 1: Accuracy
    color1 = "tab:blue"
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Test Accuracy (%)", color=color1)
    bars1 = ax1.bar(labels, acc, color=color1, alpha=0.7, label="Accuracy (%)")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Bar 2: CO₂ on twin axis
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("CO₂ Emissions (g)", color=color2)
    bars2 = ax2.bar(labels, co2, color=color2, alpha=0.4, label="CO₂ (g)", width=0.4)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Value labels on bars
    for bar, val in zip(bars1, acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, co2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.title("Green AI Trade-Off: Accuracy vs. CO₂ Emissions")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
```

| Variable | Type | Notes |
|----------|------|-------|
| `experiment_labels` | `list[str]` | `["A", "B", "C", "D"]` |
| `accuracies_pct` | `list[float]` | length 4 |
| `emissions_list` | `list[float]` | length 4 |
| `fig`, `ax1`, `ax2` | `Figure`, `Axes` | matplotlib objects |

**Suggested layout | 建议布局：** Twin y‑axis bar chart, or two subplots sharing the same x‑labels. Save as `tradeoff_plot.png`.

---

## 5. The Conclusion (README.md) | 结论（README.md）

**English:** You must write a short conclusion identifying the **"Winning Configuration."** Answer the following questions:

1. Did doubling the dataset size double the accuracy? Did it double the emissions?
2. Did doubling the epochs from 50 to 100 provide a meaningful accuracy boost, or did it just waste electricity?
3. Which of the 4 models would you deploy to production and why?

**中文：** 你必须撰写简短结论，明确**「最优配置（Winning Configuration）」**，并回答以下问题：

1. 数据集规模翻倍后，准确率是否也翻倍？碳排放是否也翻倍？
2. 训练轮数从 50 增至 100，是否带来有意义的准确率提升，还是只是浪费电力？
3. 4 个模型中，你会将哪一个部署到生产环境？为什么？

### Analysis Logic | 分析逻辑

| Comparison | Pairs | What to check |
|------------|-------|---------------|
| Data doubled | A vs C, B vs D | Did `test_accuracy_pct` ~×2? Did `emissions_g` ~×2? |
| Epochs doubled | A vs B, C vs D | Accuracy gain vs extra CO₂ — worth it? |
| Winning config | all 4 rows | Best accuracy-per-gram or acceptable accuracy at lowest cost |

**Deploy criteria | 部署标准（示例）：**

```text
Pick the model with sufficient test accuracy and the lowest (or acceptable)
emissions_g — not necessarily the highest accuracy.
```

### README Template | 模板

```markdown
# Green AI Trade-Off — Results

## Winning Configuration: Experiment [X]

- **Accuracy:** XX.XX%
- **CO₂ Emissions:** X.XXXX g
- **Dataset:** XX%  |  **Epochs:** XX

### 1. Data Size: Doubling Effect
Doubling the dataset from 50% → 100% (A→C, B→D):
- Accuracy change: ...
- Emissions change: ...

### 2. Epochs: Doubling Effect
Doubling epochs from 50 → 100 (A→B, C→D):
- Accuracy gain: ...
- Energy cost: ...

### 3. Production Deployment Choice
I would deploy Experiment [X] because ...
```

---

## 6. GitHub Submission Protocol | GitHub 提交规范

**English:** You must submit this lab via the Open Source PR workflow established previously.

1. Open your Colab Notebook and execute your 4 experiments.
2. Save your visualization as an image (e.g., `tradeoff_plot.png`).
3. Create a new folder in your repository via the terminal:
4. Move your code, the `emissions.csv` file, your plot image, and a `README.md` (containing your conclusion) into this folder.
5. Add, commit, and push to your GitHub account.

**中文：** 须通过先前建立的开源 PR 工作流提交本实验。

1. 打开 Colab Notebook，运行 4 组实验。
2. 将可视化结果保存为图片（例如 `tradeoff_plot.png`）。
3. 在仓库中新建文件夹。
4. 将代码、`emissions.csv`、图表图片，以及包含结论的 `README.md` 移入该文件夹。
5. 添加、提交并推送到 GitHub。

### Commands | 命令

```bash
# Step 3: Create folder
mkdir -p week-10-12-async-lab

# Step 4: Move deliverables into folder
mv green_ai_tradeoff.ipynb week-10-12-async-lab/
mv emissions.csv week-10-12-async-lab/
mv tradeoff_plot.png week-10-12-async-lab/
mv README.md week-10-12-async-lab/

# Step 5: Commit & push
git add week-10-12-async-lab/
git commit -m "feat: completed Deep MLP carbon vs data size experiments"
git push origin main
```

### Deliverables Checklist | 交付清单

| File | Description |
|------|-------------|
| Notebook or `.py` script | 4 experiments, MLP, CodeCarbon |
| `emissions.csv` | CodeCarbon output |
| `tradeoff_plot.png` | accuracy vs CO₂ chart |
| `README.md` | Winning configuration + 3 questions |
| `week-10-12-async-lab/` | folder for GitHub submission |

---

## Appendix: Full Code Structure | 完整代码结构

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

# ── Config ──
EXPERIMENTS: list[dict] = [
    {"name": "A", "data_fraction": 0.5, "epochs": 50},
    {"name": "B", "data_fraction": 0.5, "epochs": 100},
    {"name": "C", "data_fraction": 1.0, "epochs": 50},
    {"name": "D", "data_fraction": 1.0, "epochs": 100},
]

# ── Step 1: Load & Preprocess ──
def load_and_preprocess_data() -> tuple:
    """Returns X_train_full, y_train_full, X_test, y_test, input_shape, num_classes."""
    (X_train_full, y_train_full), (X_test, y_test) = \
        tf.keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test       = X_test.astype("float32") / 255.0
    X_train_full = np.expand_dims(X_train_full, axis=-1)
    X_test       = np.expand_dims(X_test, axis=-1)
    return X_train_full, y_train_full, X_test, y_test, (28, 28, 1), 10

# ── Step 2: Data Slicing ──
def get_train_data(X_full, y_full, fraction):
    n = int(len(X_full) * fraction)
    return X_full[:n], y_full[:n]

# ── Step 3: Build MLP ──
def build_mlp(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model

# ── Step 4: Run Experiment ──
def run_experiment(name, X_train, y_train, X_test, y_test, epochs, input_shape, num_classes):
    model = build_mlp(input_shape, num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    tracker = EmissionsTracker(project_name="green-ai-tradeoff", output_file="emissions.csv", log_level="error")
    tracker.start()
    model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=1)
    emissions_g = tracker.stop()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        "experiment": name,
        "data_fraction": len(X_train) / (len(X_train) + len(X_train)),
        "epochs": epochs,
        "train_samples": len(X_train),
        "test_accuracy": test_accuracy,
        "test_accuracy_pct": round(test_accuracy * 100, 2),
        "emissions_g": round(emissions_g, 4),
    }

# ── Step 5: Plot ──
def plot_tradeoff(results_df, save_path="tradeoff_plot.png"):
    labels = results_df["experiment"].tolist()
    acc    = results_df["test_accuracy_pct"].tolist()
    co2    = results_df["emissions_g"].tolist()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Test Accuracy (%)", color="tab:blue")
    bars1 = ax1.bar(labels, acc, color="tab:blue", alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    for bar, val in zip(bars1, acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel("CO₂ Emissions (g)", color="tab:red")
    bars2 = ax2.bar(labels, co2, color="tab:red", alpha=0.4, width=0.4)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    for bar, val in zip(bars2, co2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.title("Green AI Trade-Off: Accuracy vs. CO₂ Emissions")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# ── Main ──
def main():
    X_train_full, y_train_full, X_test, y_test, input_shape, num_classes = load_and_preprocess_data()
    results = []
    for cfg in EXPERIMENTS:
        X_train, y_train = get_train_data(X_train_full, y_train_full, cfg["data_fraction"])
        result = run_experiment(
            name=cfg["name"],
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            epochs=cfg["epochs"],
            input_shape=input_shape,
            num_classes=num_classes,
        )
        results.append(result)
    results_df = pd.DataFrame(results)
    print(results_df)
    plot_tradeoff(results_df)

if __name__ == "__main__":
    main()
```

---

## Environment Notes | 环境说明

| Item | Recommendation |
|------|----------------|
| Python | **3.11 or 3.12** (TensorFlow may not support 3.13) |
| Packages | `tensorflow`, `codecarbon`, `matplotlib`, `pandas`, `numpy`, `ipykernel` |
| Kernel | Cursor: **Select Kernel → Python Environments → `venv\Scripts\python.exe`** |
| Minimal kernel setup | `pip install ipykernel` in venv; no full `jupyter` required for Cursor |

## Common Pitfalls | 常见错误

1. **Reusing the same trained `model`** across experiments — **must rebuild each time**
2. **Changing the test set** per experiment — **keep test set fixed**
3. **Using CNN only** without `Flatten + 5×Dense` — **violates task requirement**
4. **Forgetting to convert accuracy** to **percent (%)** on the plot
5. **Running on Python 3.13** without verifying TensorFlow installs
