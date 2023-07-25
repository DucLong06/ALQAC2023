import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.task1.raw_data import df_create_data_training
import my_env


def visualize_data(df_train, df_test):
    # Số lượng trường None
    none_count_df1 = df_train.isnull().sum()
    none_count_df2 = df_test.isnull().sum()

    # Số lượng data trong các df
    data_count = pd.DataFrame({
        'DataFrame': ['df_train', 'df_test'],
        'Total Data': [len(df_train), len(df_test)]
    })

    # Gộp hai biểu đồ
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Số lượng trường None
    none_count_df1.plot(
        kind="bar", ax=axes[0, 0], title="None Count - DataFrame Train")
    none_count_df2.plot(
        kind="bar", ax=axes[0, 1], title="None Count - DataFrame Test")

    # Số lượng data trong các df
    axes[1, 0].axis('off')
    axes[1, 0].table(cellText=data_count.values,
                     colLabels=data_count.columns, cellLoc='center', loc='center')
    axes[1, 0].set_title('Total Data Count')

    # Biểu đồ thống tròn về tỷ lệ phân bố nhãn
    labels = ['Relevant == 1', 'Relevant == 0']
    sizes_df1 = [sum(df_train['relevant'] == 1),
                 sum(df_train['relevant'] == 0)]
    sizes_df2 = [sum(df_test['relevant'] == 1), sum(df_test['relevant'] == 0)]

    # Biểu đồ tròn - DataFrame Train
    ax1 = axes[1, 1].pie(sizes_df1, labels=labels,
                         autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Label Distribution - DataFrame Train')
    ax1_legend_labels = [f'{size:,} ({percent:.1f}%)' for size, percent in zip(
        sizes_df1, sizes_df1/np.sum(sizes_df1))]
    axes[1, 1].legend(ax1[0], ax1_legend_labels,
                      title='Data Count', loc='upper left')

    # Biểu đồ tròn - DataFrame Test
    ax2 = axes[1, 1].pie(sizes_df2, labels=labels,
                         autopct='%1.1f%%', startangle=90, radius=0.7)
    axes[1, 1].set_title('Label Distribution - DataFrame Test')
    ax2_legend_labels = [f'{size:,} ({percent:.1f}%)' for size, percent in zip(
        sizes_df2, sizes_df2/np.sum(sizes_df2))]
    axes[1, 1].legend(ax2[0], ax2_legend_labels,
                      title='Data Count', loc='upper right')

    plt.tight_layout()
    plt.savefig("data_visualization.png")
    plt.show()


# Gọi hàm visualize_data với hai DataFrame df_train và df_test
df_train, df_test = df_create_data_training(
    my_env.PATH_TO_PUBLIC_TRAIN, my_env.PATH_TO_CORPUS)
visualize_data(df_train, df_test)
