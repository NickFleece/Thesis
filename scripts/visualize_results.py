import pickle
import matplotlib.pyplot as plt

with open("../models/1_test_model_hist.pickle", 'rb') as f:
    data = pickle.load(f)

plt.figure(figsize=(15,15))
for k in data.keys():
    plt.plot(range(len(data[k])), data[k], label=k)
plt.legend()
plt.show()

# print(data.keys())