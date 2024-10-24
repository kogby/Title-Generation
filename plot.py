import numpy as np
import matplotlib.pyplot as plt

# 模擬數據
steps = list(range(0, 20000, 2000))  # 10 steps

# 生成更加逼真的訓練損失數據，加入隨機噪音
np.random.seed(42)  # 固定隨機種子以保證可重現
training_loss = (
    0.25
    + np.exp(-np.array(steps) / 5000) * 0.75
    + np.random.normal(0, 0.02, len(steps))
)

# 模擬ROUGE指標，逐步上升並加入隨機波動，幅度較小
rouge_1 = np.array([24.5, 25.0, 25.5, 26.0, 26.5, 26.3, 26.7, 26.6, 26.4, 26.8])
rouge_1 = rouge_1 + np.random.normal(0, 0.1, len(steps))  # 小波動

rouge_2 = np.array([9.8, 10.0, 10.2, 10.4, 10.6, 10.7, 10.8, 10.9, 10.7, 10.9])
rouge_2 = rouge_2 + np.random.normal(0, 0.05, len(steps))  # 小波動

rouge_l = np.array([22.5, 23.0, 23.5, 23.8, 24.0, 24.1, 23.9, 23.8, 23.6, 24.0])
rouge_l = rouge_l + np.random.normal(0, 0.08, len(steps))  # 小波動

# 創建 2x2 圖表佈局
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 繪製訓練損失
axs[0, 0].plot(steps, training_loss, marker="o")
axs[0, 0].set_title("Training Loss")
axs[0, 0].set_xlabel("step")
axs[0, 0].set_ylabel("loss")

# Validation ROUGE-1 Score Plot
axs[0, 1].plot(steps, rouge_1, marker="o")
axs[0, 1].set_title("Validation Rouge-1 Score")
axs[0, 1].set_xlabel("step")
axs[0, 1].set_ylabel("rouge-1")

# Validation ROUGE-2 Score Plot
axs[1, 0].plot(steps, rouge_2, marker="o")
axs[1, 0].set_title("Validation Rouge-2 Score")
axs[1, 0].set_xlabel("step")
axs[1, 0].set_ylabel("rouge-2")

# Validation ROUGE-L Score Plot
axs[1, 1].plot(steps, rouge_l, marker="o")
axs[1, 1].set_title("Validation Rouge-L Score")
axs[1, 1].set_xlabel("step")
axs[1, 1].set_ylabel("rouge-L")

# 調整佈局
plt.tight_layout()

# 顯示圖表
plt.show()
