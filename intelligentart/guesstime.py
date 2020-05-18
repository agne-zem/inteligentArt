import numpy as np
from scipy.interpolate import griddata
from intelligentart.models import Generated


def run_guess(image):

    selected_image = Generated.query.filter_by(file_name=image).first()
    epochs = selected_image.epochs
    steps_per_epoch = selected_image.steps


    max_steps = 50
    # [epoch, step]
    known_data = np.array([[1, 1], [1, 10], [1, 20], [1, max_steps],
                          [2, 1], [2, 10], [2, 20], [2, max_steps],
                          [3, 1], [3, 10], [2, 20], [3, max_steps],
                          [4, 1], [4, 10], [4, 20], [4, max_steps],
                          [5, 1], [5, 10], [5, 20], [5, max_steps]])
    # [time]
    known_values = [6, 41, 70, 176,
                    9, 65, 129, 349,
                    13, 108, 208, 558,
                    17, 143, 281, 782,
                    19, 184, 352, 900
                    ]

    # making mesh for all combinations
    x = np.linspace(1, 5, 5)
    y = np.linspace(1, max_steps, max_steps)
    grid_x, grid_y = np.meshgrid(x, y)

    # time interpolation
    grid_z0 = griddata(known_data, known_values, (grid_x, grid_y), method='linear')

    result = str(grid_z0[steps_per_epoch-1, epochs-1])

    return result
