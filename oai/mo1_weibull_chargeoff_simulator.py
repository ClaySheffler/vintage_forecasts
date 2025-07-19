# filename: weibull_chargeoff_simulator.py

import marimo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    return np, plt, weibull_min


@app.cell
def __():
    app.title("Weibull Charge-Off Timing Simulator")
    return


@app.cell
def __(mo):
    # Sliders for parameters
    k = mo.slider(0.5, 5.0, value=1.5, step=0.1, label="Shape (k): Hazard Behavior")
    lam = mo.slider(3.0, 60.0, value=18.0, step=1.0, label="Scale (λ): Median Charge-off Month")
    n_samples = mo.slider(100, 10000, value=2000, step=100, label="Sample Size (synthetic loans)")
    return k, lam, n_samples
__dependencies__ = ["mo"]


@app.cell
def __(np):
    x = np.linspace(0.01, 60, 500)  # Avoid zero to prevent divide-by-zero in hazard
    return x,


@app.cell
def __(x, k, lam, weibull_min):
    # Theoretical Weibull functions
    pdf = weibull_min.pdf(x, c=k.value, scale=lam.value)
    cdf = weibull_min.cdf(x, c=k.value, scale=lam.value)
    sf = weibull_min.sf(x, c=k.value, scale=lam.value)
    hazard = np.where(sf > 0, pdf / sf, 0.0)
    return pdf, cdf, sf, hazard
  

@app.cell
def __(weibull_min, k, lam, n_samples):
    # Simulated charge-off times
    simulated_data = weibull_min.rvs(c=k.value, scale=lam.value, size=n_samples.value)
    return simulated_data,
  

@app.cell
def __(plt, x, pdf, cdf, sf, hazard, simulated_data, k, lam):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Histogram + PDF
    axs[0, 0].hist(simulated_data, bins=50, density=True, alpha=0.5, label="Simulated", color="gray")
    axs[0, 0].plot(x, pdf, label="Theoretical PDF", color="steelblue", linewidth=2)
    axs[0, 0].set_title("PDF & Simulated Charge-off Timing")
    axs[0, 0].set_xlabel("Month")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # CDF
    axs[0, 1].plot(x, cdf, color="darkgreen", linewidth=2)
    axs[0, 1].set_title("CDF: Cumulative Charge-off Rate")
    axs[0, 1].set_xlabel("Month")
    axs[0, 1].set_ylabel("Cumulative Probability")
    axs[0, 1].grid(True)

    # Survival Function
    axs[1, 0].plot(x, sf, color="darkred", linewidth=2)
    axs[1, 0].set_title("Survival Function: Accounts Still Open")
    axs[1, 0].set_xlabel("Month")
    axs[1, 0].set_ylabel("Survival Probability")
    axs[1, 0].grid(True)

    # Hazard Function
    axs[1, 1].plot(x, hazard, color="purple", linewidth=2)
    axs[1, 1].set_title("Hazard Function: Instantaneous Risk")
    axs[1, 1].set_xlabel("Month")
    axs[1, 1].set_ylabel("Hazard Rate")
    axs[1, 1].grid(True)

    fig.suptitle(f"Weibull Charge-Off Simulator (k = {k.value}, λ = {lam.value} months)", fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig, axs
