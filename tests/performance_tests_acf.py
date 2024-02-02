import timeit

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.tsa.stattools import acf

use_fft = [False, True]
signal_lengths = np.hstack((np.arange(15, 515, 15), np.arange(500, 5500, 500)))
results = [[], []]

for i, use_fft in enumerate([False, True]):
    for length in signal_lengths:
        signal = np.random.randn(length)

        def measure_acf(signal, use_fft):
            return acf(signal, adjusted=True, fft=use_fft, nlags=int(len(signal) / 4))[1:]

        cpu_time = timeit.timeit("measure_acf()", globals=globals(), number=10)
        mean_cpu_time = cpu_time / 10  # Average time per execution

        results[i].append(mean_cpu_time * 1000)
results = np.array(results)

fig, ax = plt.subplots(1, 1)
ax.plot(signal_lengths, results[0], "o-", label="full convolution")
ax.plot(signal_lengths, results[1], "o-", label="fft")
ax.set_ylabel("CPU time / ms")
ax.set_xlabel("Signal size / #")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()

ax_inset = inset_axes(ax, width="30%", height="30%", loc=1)  # loc=2 is for upper left
ax_inset.plot(signal_lengths, results[0], "o-", label="full convolution")
ax_inset.plot(signal_lengths, results[1], "o-", label="FFT")
ax_inset.set_xlim(0, 500)
ax_inset.set_ylim(0, 0.10)
ax_inset.yaxis.set_visible(False)
ax_inset.yaxis.set_ticks([])
ax_inset.axvline(
    x=signal_lengths[np.where(results[0] > results[1])[0][0]],
    color="r",
    linestyle="--",
)

plt.show(block=False)
