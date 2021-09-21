import pickle
import numpy as np
import matplotlib.pyplot as plt
with open('results.pkl', 'rb') as f:
    loss_D,acc_D, loss_GAN, acc_GAN, DICE = pickle.load(f)

with open('validation_values.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    mses, dices, mean_mses, mean_dices = pickle.load(f)

print(mean_mses, mean_dices)

#print(DICE)
print(len(DICE))
epoch = len(DICE) - 1
x = np.linspace(1, epoch + 1, epoch + 1)
#clear_output(wait=True)
plt.figure(1)
plt.subplot(321)
plt.title("loss_disc")
plt.plot(x, loss_D)
plt.subplot(322)
plt.title("loss_GAN")
plt.plot(x, loss_GAN)
plt.subplot(323)
plt.title("acc_disc")
plt.plot(x, acc_D)
plt.subplot(324)
plt.title("acc_GAN")
plt.plot(x, acc_GAN)

plt.subplot(313)
plt.title("DICE")
plt.plot(x, DICE)

plt.show()