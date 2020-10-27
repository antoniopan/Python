import numpy as np
import pandas as pd


def create_csv_file(file_name, data_len):
    data_xyz = np.random.randn(data_len, 3)

    # Formula
    # z = x^2 + y^2
    mask = data_xyz[:, 0] ** 2 + data_xyz[:, 1] ** 2 - data_xyz[:, 2] / 2 > -1
    b = np.zeros([data_len, 1])
    b[mask] = 1
    b[~mask] = 0
    data_xyz = np.c_[data_xyz, b]

    df = pd.DataFrame(data_xyz, columns=['X', 'Y', 'Z', 're'])
    df.to_csv(file_name)


if __name__ == '__main__':
    create_csv_file('data.csv', 1000)
