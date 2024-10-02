import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def make_contour_plot(X_plot, Y_plot, frame, text) -> None:
    fig, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})

    ax_text.axis('off')  # Turn off axis for the text subplot
    ax_text.text(0.5, 0.5, text, va='center', ha='center', fontsize=12, wrap=True)

    contour = ax_plot.contourf(X_plot, Y_plot, frame, levels=100, cmap='jet')
    # fig.colorbar(contour, ax=ax_plot)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def make_plot(x_data, y_data, text, x_label, y_label, title) -> None:
    _, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, text, va='center', ha='center', fontsize=12, wrap=True, usetex=False)

    # Right subplot for the plot
    ax_plot.plot(x_data, y_data)
    ax_plot.set_xlabel(x_label)
    ax_plot.set_ylabel(y_label)
    ax_plot.set_title(title)
    ax_plot.grid(True)
    plt.tight_layout()
    plt.show()
