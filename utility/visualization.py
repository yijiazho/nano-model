import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'results/best_model_combined_losses.csv'

loss_values = []

with open(path, 'r') as file:
    for line in file:
        if line == '\n':
            continue
        value = line.split(':')[-1].strip()
        loss_values.append(float(value))
        
loss_array = np.array(loss_values)

# Display the numpy array
print(loss_array)

l = len(loss_array)
steps = np.linspace(0, 300 * l, l)

plt.plot(steps, loss_array)
plt.xlabel('Steps')
plt.ylabel('Cross Entropy Loss')
plt.ylim(0, 12)


plt.savefig('plot/loss.png', format='png', dpi=300)

plt.show()