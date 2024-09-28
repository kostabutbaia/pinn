import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


def create_anim_gif(name: str, L, frames, x_range) -> None:
    fig = plt.figure()
    plt.xlim(0, L)
    plt.ylim(-1, 1)
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