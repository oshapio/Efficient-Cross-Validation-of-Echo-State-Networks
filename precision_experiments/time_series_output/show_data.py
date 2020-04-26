from utils.data_reader import get_mitdb
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

rc('font', **font)

data = get_mitdb(read_names=["100"])

color1 = [0, 0, 0, 1]
color2 = [0, 0, 0, 1]
start = 546792 - 1000
range = 1500

fig, (ax0, ax1) = plt.subplots(2, figsize=(8, 3), sharex=True, gridspec_kw={'height_ratios':[3, 1]})


plt.margins(x=0)

ax0.plot(data[0][start:start+range, 0] / 360.0, data[0][start:start+range,1], "-" , color=color1, label="Input")

ax0.set_ylabel("Input")

ax1.plot(data[0][start:start+range, 0] / 360.0, data[0][start:start+range,3].astype(np.float16), "-" , color=color2, label="Output")
ax1.set_ylabel("Output")

for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.spines["left"].set_visible(False)

# frame1.axes.get_yaxis().set_visible(False)
# plt.ylim((0, 1))
plt.xlabel("Time (s)")

fig.align_ylabels((ax0, ax1))

ax0.grid(linestyle=":", alpha=0.7)
ax1.grid(linestyle=":", alpha=0.7)
plt.savefig("input_output_plot.pdf", bbox_inches="tight", pad_inches = 0)
plt.show()

