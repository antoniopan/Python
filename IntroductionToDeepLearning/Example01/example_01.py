import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


def train():
    TARGET_VARIABLE = 'user_action'
    TRAIN_TEST_SPLIT = 0.7
    HIDDEN_LAYER_SIZE = 30
    raw_data = pd.read_csv('data.csv')

    mask = np.random.rand(len(raw_data)) < TRAIN_TEST_SPLIT
    tr_dataset = raw_data[mask]
    te_dataset = raw_data[~mask]

    tr_data = np.array(tr_dataset.drop(TARGET_VARIABLE, axis=1))
    tr_labels = np.array(tr_dataset[[TARGET_VARIABLE]])
    te_data = np.array(te_dataset.drop(TARGET_VARIABLE, axis=1))
    te_labels = np.array(te_dataset[[TARGET_VARIABLE]])

    ffnn = Sequential()
    ffnn.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(3, ), activation="sigmoid"))
    ffnn.add(Dense(1, activation="sigmoid"))
    ffnn.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
    ffnn.fit(tr_data, tr_labels, epochs=150, batch_size=2, verbose=1)

    metrics = ffnn.evaluate(te_data, te_labels, verbose=1)
    print("%s: %.2f%%" % (ffnn.metrics_names[1], metrics[1]*100))
    

if __name__ == '__main__':
    train()
