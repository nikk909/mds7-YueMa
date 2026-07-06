# Async Lab: The Green AI Trade-Off
# 异步实验：绿色 AI 权衡

---

## 1. Objective | 目标

**English:** In this lab, you will act as a Lead MLOps Engineer tasked with optimizing a model for a production environment where cloud computing budgets and carbon emissions are strictly monitored. You must determine if utilizing the full dataset and training for extended epochs provides a justifiable Return on Investment (ROI) in terms of accuracy vs. carbon emitted.

**中文：** 在本实验中，你将扮演首席 MLOps 工程师，负责在云计算预算与碳排放均受严格监控的生产环境中优化模型。你需要判断：使用完整数据集并延长训练轮数，是否在「准确率 vs. 碳排放」方面具有合理的投资回报率（ROI）。

---

## 2. Architectural Requirements | 架构要求

### 2.1 Dataset | 数据集

**English:** Choose any Computer Vision dataset (e.g., Fashion-MNIST, CIFAR-10, SVHN).  
**Hint:** `tf.keras.datasets` has many built-in options.

**中文：** 任选一种计算机视觉数据集（例如 Fashion-MNIST、CIFAR-10、SVHN）。  
**提示：** `tf.keras.datasets` 提供多种内置数据集。

### 2.2 Model Architecture | 模型架构

**English:** You can use a CNN (`Conv2D` layers). You **must** build a Deep Multi-Layer Perceptron (MLP).

- **Requirement:** 1 `Flatten()` layer followed by at least 5 `Dense()` hidden layers, ending with your output classification layer.

**中文：** 可以使用 CNN（`Conv2D` 层），但**必须**构建一个深度多层感知机（MLP）。

- **要求：** 1 个 `Flatten()` 层，后接至少 5 个 `Dense()` 隐藏层，最后是输出分类层。

### 2.3 Tracking | 追踪

**English:** You must wrap your training loops using the **codecarbon** library's `EmissionsTracker`.

**中文：** 必须使用 **codecarbon** 库的 `EmissionsTracker` 包裹训练循环，以追踪碳排放。

---

## 3. The Experiment Matrix | 实验矩阵

**English:** You must write a Python script that trains **4 separate models from scratch** using the following configurations:

| Experiment | Training Data | Epochs |
|------------|---------------|--------|
| **A** | 50% of training data | 50 |
| **B** | 50% of training data | 100 |
| **C** | 100% of training data | 50 |
| **D** | 100% of training data | 100 |

**Hint:** To get 50% of the data, you can use slicing, e.g.:

```python
X_train_half = X_train[:len(X_train) // 2]
```

**中文：** 你必须编写 Python 脚本，**从零开始**训练 **4 个独立模型**，配置如下：

| 实验 | 训练数据 | 训练轮数 |
|------|----------|----------|
| **A** | 50% 训练集 | 50 |
| **B** | 50% 训练集 | 100 |
| **C** | 100% 训练集 | 50 |
| **D** | 100% 训练集 | 100 |

**提示：** 获取 50% 数据可使用切片，例如：

```python
X_train_half = X_train[:len(X_train) // 2]
```

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

---

## 6. GitHub Submission Protocol | GitHub 提交规范

**English:** You must submit this lab via the Open Source PR workflow established previously.

1. Open your Colab Notebook and execute your 4 experiments.
2. Save your visualization as an image (e.g., `tradeoff_plot.png`).
3. Create a new folder in your repository via the terminal:

   ```bash
   mkdir -p week-10-12-async-lab
   ```

4. Move your code, the `emissions.csv` file, your plot image, and a `README.md` (containing your conclusion) into this folder.
5. Add, commit, and push to your GitHub account:

   ```bash
   git add week-10-12-async-lab/
   git commit -m "feat: completed Deep MLP carbon vs data size experiments"
   git push origin main
   ```

**中文：** 须通过先前建立的开源 PR 工作流提交本实验。

1. 打开 Colab Notebook，运行 4 组实验。
2. 将可视化结果保存为图片（例如 `tradeoff_plot.png`）。
3. 在仓库中新建文件夹：

   ```bash
   mkdir -p week-10-12-async-lab
   ```

4. 将代码、`emissions.csv`、图表图片，以及包含结论的 `README.md` 移入该文件夹。
5. 添加、提交并推送到 GitHub：

   ```bash
   git add week-10-12-async-lab/
   git commit -m "feat: completed Deep MLP carbon vs data size experiments"
   git push origin main
   ```
