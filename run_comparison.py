from utils.plot_utils import generate_video

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_utils import smooth

gru_train = np.load("data/gru_train_loss.npy")
gru_test = np.load("data/gru_test_loss.npy")
node_train = np.load("data/node_train_loss.npy")
node_test = np.load("data/node_test_loss.npy")

plt.figure()
plt.title("GRU Train/Test Curve: 50k Datapoints, 64 batch")
plt.plot(smooth(gru_train, 200), label="Train")
plt.plot(smooth(gru_test, 200), label="Test")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.savefig("plots/gru_loss.png")

plt.figure()
plt.title("NODE Train/Test Curve: 50k Datapoints, 64 batch")
plt.plot(smooth(node_train, 200), label="Train")
plt.plot(smooth(node_test, 200), label="Test")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.savefig("plots/node_loss.png")
