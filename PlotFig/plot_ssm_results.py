from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sciIO


PLOT_DIR = Path(__file__).resolve().parent
BASE_DIR = PLOT_DIR.parent
SSM_ROM_DIR = BASE_DIR / "dynamics" / "SSM" / "ROM8_Expe"


def load_ssm_results(plot_dir: str | Path = PLOT_DIR):
    plot_dir = Path(plot_dir)
    return {
        "t": np.load(plot_dir / "t_ssm.npy"),
        "y_exprm": np.load(plot_dir / "y_exprm_ssm.npy"),
        "ut_save": np.load(plot_dir / "ut_save_ssm.npy"),
        "zd": sciIO.loadmat(str(SSM_ROM_DIR / "zd_proj.mat"))["zd_proj"],
    }


def plot_ssm_results(results=None):
    if results is None:
        results = load_ssm_results()

    t = results["t"]
    y = results["y_exprm"]
    u = results["ut_save"]
    zd = results.get("zd")
    n = min(t.size, y.shape[0])

    fig1 = plt.figure()
    labels = ["end x", "end y", "end z", "mid x", "mid y", "mid z"]
    for idx, label in enumerate(labels):
        ax = fig1.add_subplot(3, 2, idx + 1)
        ax.plot(t[:n], y[:n, idx], label=label)
        if zd is not None and idx < 3:
            ref_n = min(n, zd.shape[1])
            ax.plot(t[:ref_n], zd[idx, :ref_n] * 1000.0, "k--", label=f"desired {label}")
        ax.legend()

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    for idx in range(u.shape[1]):
        ax.plot(t[: u.shape[0]], u[:, idx], label=f"u{idx + 1}")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor command (mm)")
    plt.show()


if __name__ == "__main__":
    plot_ssm_results()
