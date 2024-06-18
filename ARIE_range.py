import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import scipy.stats as st

## 已知beta,alpha，推alpha_x, alpha_y
def beta_to_xy(beta, alpha):
    beta = np.radians(beta)
    alpha = np.radians(alpha)
    alpha_y = math.asin(-math.cos(beta)*math.sin(alpha))
    alpha_x = math.pi/2 - math.asin(-math.sin(beta)*math.sin(alpha) / (math.sqrt(1-(math.cos(beta)*math.sin(beta))**2))) + math.pi/2
    Alpha_y = np.degrees(-alpha_y)
    Alpha_x = np.degrees(-alpha_x)

    return Alpha_x, Alpha_y

## 已知alpha_x, alpha_y, 推beta,alpha
def xy_to_beta(alpha_x, alpha_y):
    alpha_x = np.radians(alpha_x)
    alpha_y = np.radians(alpha_y)
    beta_estimate = math.atan(math.sin(alpha_x) / math.tan(alpha_y))
    alpha_estimate = math.asin(math.sin(alpha_y) / (-math.cos(beta_estimate)))
    Beta = np.degrees(beta_estimate)
    Alpha = np.degrees(alpha_estimate)

    return Beta, Alpha

# L1 = np.array([5.5, 4.3, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]) * 3
# Tx = np.array([0.524*4.365, 0.404*4.329, 0.374*4.319, 0.314*4.426, -0.165*4.157, -0.154*4.380, -0.299*4.255, -0.508*4.361, -0.748*4.410, -0.882*4.412, -1.249*4.303, -1.434*4.387])/4.365
# coeff = np.polyfit(L1, Tx,1)
# Tx_fit = coeff[0]*L1 + coeff[1]
# low_CI_bound, high_CI_bound = st.t.interval(0.95, 12 - 1,
#                                             loc=Tx_fit,
#                                             scale=st.sem(Tx))
#
# L2 = np.array([-5, -3, -1, 1, 2, 3, 4, 5])*3
# Ty = np.array([-0.0282*4.348, -0.0181*4.365, -0.0049*4.315, 0.0033*4.302, 0.0087*4.386, 0.0255*4.432, 0.0388*4.264, 0.0439*4.367])/4.302
# coeff_2 = np.polyfit(L2, Ty,1)
# Ty_fit = coeff_2[0]*L2 + coeff_2[1]
# low_CI_bound_2, high_CI_bound_2 = st.t.interval(0.98, 8 - 1,
#                                             loc=Ty_fit,
#                                             scale=st.sem(Ty))
# plt.subplot(2,1,1)
# plt.scatter(L1, Tx)
# plt.plot(L1, Tx_fit, linewidth=2)
# plt.xticks(fontsize=18, fontproperties='Times New Roman')
# plt.yticks(fontsize=18, fontproperties='Times New Roman')
# plt.fill_between(L1, low_CI_bound, high_CI_bound, alpha=0.5, facecolor='C0')
# plt.xlabel('Lx/mm', fontsize=18, fontdict={'family': 'Times New Roman'})
# plt.ylabel('Tx', fontsize=18, fontdict={'family': 'Times New Roman'})
# plt.legend()
# plt.subplot(2,1,2)
# plt.scatter(L2, Ty)
# plt.plot(L2, Ty_fit, linewidth=2)
# plt.xticks(fontsize=18, fontproperties='Times New Roman')
# plt.yticks(fontsize=18, fontproperties='Times New Roman')
# plt.fill_between(L2, low_CI_bound_2, high_CI_bound_2, alpha=0.5, facecolor='C0')
# plt.xlabel('Ly/mm', fontsize=18, fontdict={'family': 'Times New Roman'})
# plt.ylabel('Ty', fontsize=18, fontdict={'family': 'Times New Roman'})
# plt.legend()
# plt.show()