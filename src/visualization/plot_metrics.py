import matplotlib.pyplot as plt
import json
import os
import numpy as np

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def plot_ncf_failure():
    """
    Generates a plot showing the training loss vs validation recall for the NCF model.
    This visualizes the "Cold-Start Paradox" where the model overfits to warm users.
    Saves the plot to `reports/figures/viz1_ncf_failure.png`.
    """
    ncf_path = os.path.join(REPORTS_DIR, "baseline", "training_history.json")
    if not os.path.exists(ncf_path):
        print("NCF history not found.")
        return

    with open(ncf_path) as f:
        history = json.load(f)

    epochs = history['epoch']
    loss = history['loss']
    recall = history['val_recall']

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, loss, color=color, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Recall@10', color=color, fontweight='bold')
    ax2.plot(epochs, recall, color=color, linewidth=2,
             linestyle='--', label='Recall@10')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('The Cold-Start Paradox: Overfitting to Warm Users', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "figures",
                "viz1_ncf_failure.png"), dpi=300)
    print("Saved Viz 1.")


def plot_model_comparison():
    """
    Generates a bar chart comparing the performance (Recall, NDCG, MAP) of NCF, CMF, and PTUPCDR models.
    Saves the plot to `reports/figures/viz2_comparison.png`.
    """
    # Load final results
    results = {}

    paths = {
        "Target-Only NCF": os.path.join(REPORTS_DIR, "baseline", "final_test_results.json"),
        "Linear CMF": os.path.join(REPORTS_DIR, "cmf_baseline", "final_test_results.json"),
        "Deep PTUPCDR": os.path.join(REPORTS_DIR, "ptupcdr_model", "final_results.json")
    }

    metrics = ['recall', 'ndcg', 'map']
    data = {m: [] for m in metrics}
    labels = []

    for name, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
                # Normalize keys (some saved as 'test_recall', some as 'recall')
                rec = res.get('test_recall', res.get('recall', 0))
                ndcg = res.get('test_ndcg', res.get('ndcg', 0))
                map_ = res.get('test_map', res.get('map', 0))

                data['recall'].append(rec)
                data['ndcg'].append(ndcg)
                data['map'].append(map_)
                labels.append(name)
        else:
            print(f"Warning: {name} results not found.")
            labels.append(name)
            for m in metrics:
                data[m].append(0)

    # Plotting
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, data['recall'],
                    width, label='Recall@10', color='#4e79a7')
    rects2 = ax.bar(x, data['ndcg'], width, label='NDCG@10', color='#f28e2b')
    rects3 = ax.bar(x + width, data['map'],
                    width, label='MAP', color='#e15759')

    ax.set_ylabel('Score')
    ax.set_title(
        'Cross-Domain Model Comparison (Cold-Start Test Set)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.legend()

    # Add labels on top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "figures",
                "viz2_comparison.png"), dpi=300)
    print("Saved Viz 2.")


if __name__ == "__main__":
    plot_ncf_failure()
    plot_model_comparison()
