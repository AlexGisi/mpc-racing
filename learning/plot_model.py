import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from learning.vehicle import Vehicle
from learning.util import get_abs_fp

###
MODEL_FP = "models/linear-best-chill/model"
model = Vehicle('linear')

MAX = 10
###

model_fp = get_abs_fp(__file__, MODEL_FP)
model.load_state_dict(torch.load(model_fp))
model.eval()

with torch.no_grad():
    alpha = torch.arange(start=-MAX, end=MAX, step=0.001)
    fy_front = model.front_tire.forward(alpha)
    fy_back = model.back_tire.forward(alpha)

    alpha = alpha.numpy()
    fy_front = fy_front.numpy()
    fy_back = fy_back.numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].plot(alpha, fy_front)
axs[0].set_title('front tire slip curve')
axs[0].set_xlabel('side slip angle')
axs[0].set_ylabel('fy')
axs[0].grid(True)

axs[1].scatter(alpha, fy_back)
axs[1].set_title('back tire slip curve')
axs[1].set_xlabel('side slip angle')
axs[1].set_ylabel('fy')
axs[1].grid(True)

plt.show()
