import matplotlib.pyplot as plt
import pandas as pd

R_1_approach_1 = pd.read_csv("R_1_approach_1.csv")
R_2_approach_1 = pd.read_csv("R_2_approach_1.csv")
R_1_approach_2 = pd.read_csv("R_1_approach_2.csv")
R_2_approach_2 = pd.read_csv("R_2_approach_2.csv")

logRs_appraoch_1 = pd.read_csv("logRs_approach_1.csv")
logRs_appraoch_2 = pd.read_csv("logRs_approach_2.csv")

plt.plot(R_1_approach_1, label = 'R_1_approach_1')
plt.plot(R_1_approach_2, label = 'R_1_approach_2')
plt.plot(R_2_approach_1, label = 'R_2_approach_1')
plt.plot(R_2_approach_2, label = 'R_2_approach_2')
plt.legend(loc = 'upper right')
plt.xlabel('Steps')
plt.ylabel('R_1 and R_2')
plt.show()

plt.plot(logRs_appraoch_1, label = 'logRs_approach_1')
plt.plot(logRs_appraoch_2, label = 'logRs_approach_2')
plt.legend(loc = 'upper right')
plt.xlabel('Steps')
plt.ylabel('logR_1 + logR_2')
plt.show()