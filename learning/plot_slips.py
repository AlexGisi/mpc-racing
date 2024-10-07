import pandas as pd
import matplotlib.pyplot as plt
from learning.util import get_abs_fp

###
PATH = "data/big/slip_validate.csv"
###

path = get_abs_fp(__file__, PATH)
df = pd.read_csv(path)


# plt.scatter(df['slip_back'], ay)
# plt.legend()
# plt.show()

# df['slip_front'] *= 10
# df['slip_back'] *= 10

# plt.plot(df.index, df.loc[:, ['slip_back', 'vy']], label=['slip', 'vy'])
# plt.legend()
# plt.show()

plt.scatter(df['slip_back'], df['ay'])  # Interesting circles
plt.legend()
plt.show()
