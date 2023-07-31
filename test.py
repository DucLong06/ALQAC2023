import matplotlib.pyplot as plt
import json


def get_unique_items_with_indices(data):
    result = {}
    seen_values = set()
    count = 1

    for key, value in data.items():
        if value not in seen_values:
            seen_values.add(value)
            result[f"{count}"] = value
            count += 1

    return result


def plot_accuracy_and_samples(data1, data2, category, ax):
    filtered_data1 = {key: value for key, value in data1.items()
                      if not key.startswith("prompt_Study") and key != "Essay"}
    filtered_data2 = {key: value for key, value in data2.items()
                      if not key.startswith("prompt_Study") and key != "Essay"}

    if category == "True/False":
        category_prefix = "prompt_True/False_"
        total_samples1 = data1["True/False"]
        total_samples2 = data2["True/False"]
    else:
        category_prefix = "prompt_Question_"
        total_samples1 = data1["Question"]
        total_samples2 = data2["Question"]

    prompt_labels1 = [key.replace(category_prefix, "")
                      for key in filtered_data1.keys() if category_prefix in key]
    prompt_counts1 = [value for key,
                      value in filtered_data1.items() if category_prefix in key]
    prompt_counts2 = [value for key,
                      value in filtered_data2.items() if category_prefix in key]

    accuracy_data1 = [count / total_samples1 for count in prompt_counts1]
    accuracy_data2 = [count / total_samples2 for count in prompt_counts2]

    ax.plot(prompt_labels1, accuracy_data1, marker='o',
            label='google translate', color='b')
    ax.plot(prompt_labels1, accuracy_data2,
            marker='o', label='vit5', color='r')
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    # Adjust the X-axis limit for better visualization
    ax.set_xlim(0, len(prompt_labels1) + 1)

    ax.set_title(f"Accuracy {category} Prompt ({total_samples1} samples)")
    ax.set_xlabel("Prompt Number")

    max_idx1 = accuracy_data1.index(max(accuracy_data1))
    min_idx1 = accuracy_data1.index(min(accuracy_data1))
    max_idx2 = accuracy_data2.index(max(accuracy_data2))
    min_idx2 = accuracy_data2.index(min(accuracy_data2))

    max_accuracy1 = max(accuracy_data1)
    min_accuracy1 = min(accuracy_data1)
    max_accuracy2 = max(accuracy_data2)
    min_accuracy2 = min(accuracy_data2)

    ax.plot(prompt_labels1[max_idx1], max_accuracy1, marker='o', markersize=10,
            color='g', label=f'Highest (google translate)\n{max_accuracy1*100:.2f}%')
    ax.plot(prompt_labels1[min_idx1], min_accuracy1, marker='o', markersize=10,
            color='orange', label=f'Lowest (google translate)\n{min_accuracy1*100:.2f}%')
    ax.plot(prompt_labels1[max_idx2], max_accuracy2, marker='o', markersize=10,
            color='c', linestyle='dashed', label=f'Highest (vit5)\n{max_accuracy2*100:.2f}%')
    ax.plot(prompt_labels1[min_idx2], min_accuracy2, marker='o', markersize=10,
            color='m', linestyle='dashed', label=f'Lowest (vit5)\n{min_accuracy2*100:.2f}%')

    ax.grid(axis='y')  # Add gridlines matching the Y-axis

    # Reduce the spacing between ticks on the Y-axis
    ax.set_yticks([i / 10 for i in range(11)])


if __name__ == "__main__":
    with open("data/result/googleflan-t5-xl_20230801001236.json", 'r') as file1:
        data1 = json.load(file1)

    with open("data/result/googleflan-t5-xl_20230801013425.json", 'r') as file2:
        data2 = json.load(file2)

    # Create the first plot for "True/False" category
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    plot_accuracy_and_samples(data1, data2, "True/False", ax1)

    # Move the legend outside of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the first plot
    plt.tight_layout()
    plt.savefig("data_visualization_true_false.png")
    plt.close()

    # Create the second plot for "Question" category
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_accuracy_and_samples(data1, data2, "Question", ax2)

    # Move the legend outside of the plot
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the second plot
    plt.tight_layout()
    plt.savefig("data_visualization_question.png")
    plt.close()
