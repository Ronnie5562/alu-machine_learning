#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
owners = ['Farrah', 'Fred', 'Felicia']
fruit_names_and_colors = (
    ('apples', 'red'),
    ('bananas', 'yellow'),
    ('oranges', '#ff8000'),
    ('peaches', '#ffe5b4')
)

buttom_spacing = np.zeros(fruit.shape[1])
for i in range(len(fruit)):
    plt.bar(
        owners,
        fruit[i],
        bottom=buttom_spacing,
        width=0.5,
        label=fruit_names_and_colors[i][0],
        color=fruit_names_and_colors[i][1],
    )
    buttom_spacing += fruit[i, :]

plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
plt.xlabel('Fruit Owners')
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()