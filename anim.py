import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

def create_gif_contour(name, X_plot, Y_plot, frames, text) -> None:
    fig, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})

    ax_text.axis('off')  # Turn off axis for the text subplot
    ax_text.text(0.5, 0.5, text, va='center', ha='center', fontsize=12, wrap=True)

    contour = ax_plot.contourf(X_plot, Y_plot, frames[0], levels=100, cmap='viridis')
    writer = PillowWriter(fps=15)
    plt.subplots_adjust(wspace=0.5)
    with writer.saving(fig, f'{name}.gif', 100):
        for i, frame in enumerate(frames):
            for c in contour.collections:
                c.remove()
            contour = ax_plot.contourf(X_plot, Y_plot, frame, levels=100, cmap='viridis')
            ax_plot.set_title(f'epoch={i*100}')
            writer.grab_frame()

def create_anim_gif_text(name: str, text: str, L, frames, x_range) -> None:
    # Create a figure with two subplots: text on the left, animation on the right
    fig, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 3]})

    # Left subplot for text
    ax_text.axis('off')  # Turn off axis for the text subplot
    ax_text.text(0.5, 0.5, text, va='center', ha='center', fontsize=12, wrap=True)

    # Right subplot for animation
    ax_plot.set_xlim(0, L)
    ax_plot.set_ylim(-1.5, 1.5)
    ax_plot.grid()
    line, = ax_plot.plot([], [], 'k-')

    # Create the writer for the GIF animation
    writer = PillowWriter(fps=5)

    plt.subplots_adjust(wspace=0.5)

    with writer.saving(fig, f'{name}.gif', 100):
        for i, frame in enumerate(frames):
            line.set_data(x_range, frame)
            ax_plot.set_title(f'epoch={i*100}')
            writer.grab_frame()


def create_anim_gif(name: str, L, frames, x_range) -> None:
    fig = plt.figure()
    plt.xlim(0, L)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    l, = plt.plot([], [], 'k-')
    
    writer = PillowWriter(fps=5)

    with writer.saving(fig, f'{name}.gif', 100):
        for frame in frames:
            l.set_data(x_range, frame)
            writer.grab_frame()


if __name__ == '__main__':
    NAME = 'heat_nn'
    create_anim_gif(NAME)