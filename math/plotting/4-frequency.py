#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.hist(
    student_grades,
    bins=10,
    edgecolor='black',
    range=(0, 100)
)
plt.axis([0, 100, 0, 30])
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()