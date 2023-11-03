from matplotlib import pyplot as plt
from metrics import MetricsCalculator


def report(metrics: MetricsCalculator):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    metrics.show_confusion_matrix(axs[0])
    metrics.show_roc_curve(axs[1])
    from matplotlib.patches import Rectangle

    # create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 1))

    # create the first rectangle with text
    score = round(metrics.get_accuracy(), 3)
    rect1 = Rectangle((0, 0), 1, 1, color=get_color(score))
    text1 = f'Accuracy : {score}'
    ax.add_patch(rect1)
    ax.text(0.5, 0.5, text1, ha='center', va='center', color='white')

    # create the second rectangle with text
    score = round(metrics.get_precision(), 3)
    rect2 = Rectangle((1.5, 0), 1, 1, color=get_color(score))
    text2 = f'Precision : {score}'
    ax.add_patch(rect2)
    ax.text(2, 0.5, text2, ha='center', va='center', color='white')

    # create the third rectangle with text
    score = round(metrics.get_recall(), 3)
    rect3 = Rectangle((3, 0), 1, 1, color=get_color(score))
    text3 = f'Recall : {score}'
    ax.add_patch(rect3)
    ax.text(3.5, 0.5, text3, ha='center', va='center', color='white')

    # create the fourth rectangle with text
    score = round(metrics.get_auc(), 3)
    rect4 = Rectangle((4.5, 0), 1, 1, color=get_color(score))
    text4 = f'AUC : {score}'
    ax.add_patch(rect4)
    ax.text(5, 0.5, text4, ha='center', va='center', color='white')

    # set the axis limits and remove the axis ticks
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # show the plot
    plt.show()


def get_color(score: float):
    if score < 0.60:
        return 'red'
    if score < 0.75:
        return 'orange'
    if score < 0.90:
        return 'yellow'
    return 'green'
