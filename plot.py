import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# data = {
#     "Method": [
#         "DANN", "DANN_CORAL", "DANN_MMD",
#         "DANN_LMMD", "DANN_INFOMAX", "DANN_MMD_INFOMAX"
#     ],
#     "HC185": [0.0556, 0.08978, 0.21225, 0.19048, 0.34764, 0.44491],
#     "HC188": [0.1113, 0.09715, 0.27085, 0.31642, 0.4651, 0.46191],
#     "HC191": [0.0824, 0.03113, 0.18815, 0.12132, 0.36519, 0.39498],
#     "HC194": [0.0749, 0.04045, 0.24506, 0.17631, 0.29847, 0.28294],
#     "HC197": [0.12981, -0.00196, 0.2933, 0.14952, 0.2909, 0.41694],
# }
data = {
    "Method": [
        "Baseline","DANN", "DANN_CORAL", "DANN_MMD",
        "DANN_LMMD", "DANN_INFOMAX", "DANN_MMD_INFOMAX"
    ],
    "HC185": [0.408, 0.4636, 0.4978 , 0.6203 , 0.5985 , 0.7556 ,0.8529 ],
    "HC188": [0.38 , 0.4913, 0.4772 , 0.6509 , 0.6964 , 0.8451 ,0.8419 ],
    "HC191": [0.347, 0.4294, 0.3781 , 0.5352 , 0.4683 , 0.7122 ,0.7420 ],
    "HC194": [0.305, 0.3799, 0.3455 , 0.5501 , 0.4813 , 0.6035 ,0.5879 ],
    "HC197": [0.498, 0.6278, 0.4960 , 0.7913 , 0.6475 , 0.7889 ,0.9149 ]
}
df = pd.DataFrame(data)
df.set_index("Method", inplace=True)


colors = plt.cm.tab20.colors
ax = df.T.plot(kind="bar", figsize=(10,6), color=colors[:len(df)])


for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2, rotation=90)

# 美化样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
plt.axhline(0, color="black", linewidth=1)

# 标题和标签
# plt.title("Differenz der Methoden zum Baseline (pro HC-Gruppe)", fontsize=14, weight="bold")
# plt.ylabel("Differenz (vs. Baseline)", fontsize=12)
plt.title("Durchschnittliche Genauigkeit", fontsize=14, weight="bold")
plt.ylabel("ACC", fontsize=12)
plt.xlabel("HC-Gruppe", fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)

plt.legend(title="Methode", bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, fontsize=10)

plt.tight_layout()
plt.show()
