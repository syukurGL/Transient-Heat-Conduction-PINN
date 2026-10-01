# Environment and GPU setup

import os
import json
import time
import hashlib

import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator


# GPU / CuPy

GPU_DEVICE = cp.cuda.Device()
GPU_DEVICE_ID = GPU_DEVICE.id

print()
print("=" * 78)
print("GPU ACCELERATION — CuPy / CUDA")
print("=" * 78)
print(f"GPU device ID   : {GPU_DEVICE_ID}")
print(f"GPU device name : {cp.cuda.runtime.getDeviceProperties(GPU_DEVICE_ID)['name'].decode()}")
print(f"CuPy version    : {cp.__version__}")
print("=" * 78)


def gpu_to_cpu(array):
    """Convert a CuPy array to a NumPy array for disk I/O / plotting."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


from google.colab import drive

drive.mount(
    "/content/drive",
    force_remount=False
)


BASE_DIR = (
    "/content/drive/MyDrive/"
    "FTCS_Baseline"
)

os.makedirs(
    BASE_DIR,
    exist_ok=True
)


#

RUN_ID = "FTCS_baseline_v1"


X_MIN = 0.0
X_MAX = 35.0

Y_MIN = 0.0
Y_MAX = 17.0


T_INITIAL = 20.0


TIME_START = 0.0
TIME_END = 10.0


VELOCITIES = [
    1.0,
    2.0,
    3.0
]


#

GRID_LIST = [
    100,
    200,
    300,
    400,
    500
]

# Material properties and source configuration

RHO = 7.6e-06

CP = 658.0

K = 2.5e-02


#
# alpha = k / (rho * cp)

ALPHA = (
    K /
    (
        RHO *
        CP
    )
)


SOURCE_X0 = 34.0

SOURCE_Y0 = 8.5


SOURCE_Q0 = 5.0

SOURCE_RADIUS = 1.0


T_INLET = 20.0

GRADIENT_X_MAX = -0.001

GRADIENT_Y_MIN = +0.001

GRADIENT_Y_MAX = -0.001


CHECKPOINT_INTERVAL = 5000


OUTPUT_TIMES = np.arange(
    1.0,
    TIME_END + 1.0,
    1.0
)


CONTOUR_LEVELS = 60

COLORMAP = "coolwarm"


plt.rcParams.update({

    "font.family": "serif",

    "mathtext.fontset": "cm",

    "axes.labelsize": 16,

    "axes.titlesize": 19,

    "xtick.labelsize": 13,

    "ytick.labelsize": 13,

    "legend.fontsize": 12,

    "figure.dpi": 120,

    "savefig.dpi": 600
})

# Grid and configuration utilities

DX_GLOBAL = None

DY_GLOBAL = None


def velocity_folder(velocity):

    folder_name = (
        f"velocity_{velocity:.0f}mm_s"
    )

    path = os.path.join(
        BASE_DIR,
        folder_name
    )

    os.makedirs(
        path,
        exist_ok=True
    )

    return path


def grid_folder(velocity, N):

    base = velocity_folder(
        velocity
    )

    path = os.path.join(
        base,
        f"grid_{N}"
    )

    os.makedirs(
        path,
        exist_ok=True
    )

    return path


def case_name(velocity, N):

    return (
        f"FTCS_"
        f"v{velocity:.0f}mm_s_"
        f"grid{N}"
    )


def simulation_config(velocity, N):

    return {

        "method":
            "FTCS-GPU-CuPy",

        "run_id":
            RUN_ID,

        "velocity_mm_s":
            float(velocity),

        "grid_N":
            int(N),

        "x_min_mm":
            float(X_MIN),

        "x_max_mm":
            float(X_MAX),

        "y_min_mm":
            float(Y_MIN),

        "y_max_mm":
            float(Y_MAX),

        "time_start_s":
            float(TIME_START),

        "time_end_s":
            float(TIME_END),

        "initial_temperature_C":
            float(T_INITIAL),

        "rho":
            float(RHO),

        "cp":
            float(CP),

        "thermal_conductivity":
            float(K),

        "thermal_diffusivity":
            float(ALPHA),

        "source_x0_mm":
            float(SOURCE_X0),

        "source_y0_mm":
            float(SOURCE_Y0),

        "source_Q0":
            float(SOURCE_Q0),

        "source_radius_mm":
            float(SOURCE_RADIUS),

        "boundary_x_min":
            "T = 20 C",

        "boundary_x_max":
            "dT/dx = -0.001 C/mm",

        "boundary_y_min":
            "dT/dy = +0.001 C/mm",

        "boundary_y_max":
            "dT/dy = -0.001 C/mm",

        "checkpoint_interval":
            int(CHECKPOINT_INTERVAL),

        "output_times_s":
            [
                float(t)
                for t in OUTPUT_TIMES
            ]
    }


def configuration_hash(config):

    text = json.dumps(
        config,
        sort_keys=True
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def save_configuration(
    velocity,
    N,
    dx,
    dy,
    dt,
    dt_stable,
    Fo_x,
    Fo_y,
    number_steps
):

    folder = grid_folder(
        velocity,
        N
    )

    config = simulation_config(
        velocity,
        N
    )

    cfg_hash = configuration_hash(
        config
    )

    data = dict(config)

    data["config_hash"] = cfg_hash

    data["dx_mm"] = float(dx)

    data["dy_mm"] = float(dy)

    data["dt_stable_s"] = float(
        dt_stable
    )

    data["dt_used_s"] = float(
        dt
    )

    data["Fo_x"] = float(
        Fo_x
    )

    data["Fo_y"] = float(
        Fo_y
    )

    data["Fo_sum"] = float(
        Fo_x + Fo_y
    )

    data["number_steps"] = int(
        number_steps
    )

    path = os.path.join(
        folder,
        f"{case_name(velocity, N)}"
        "_configuration.json"
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    return (
        config,
        cfg_hash,
        path
    )

# FTCS numerical core


def calculate_time_step(dx, dy):

    denominator = (
        2.0 *
        ALPHA *
        (
            1.0 / dx**2
            +
            1.0 / dy**2
        )
    )

    dt_stable = (
        1.0 /
        denominator
    )

    safety_factor = 0.8

    dt_candidate = (
        safety_factor *
        dt_stable
    )

    return (
        dt_stable,
        dt_candidate
    )


def initial_field(nx, ny):

    return cp.full(
        (nx, ny),
        T_INITIAL,
        dtype=cp.float64
    )




def moving_heat_source(
    X,
    Y,
    time_s,
    velocity
):

    mu_x = (
        SOURCE_X0
        -
        velocity *
        time_s
    )

    mu_y = SOURCE_Y0

    r2 = (
        (X - mu_x)**2
        +
        (Y - mu_y)**2
    )

    Q = (
        SOURCE_Q0 *
        cp.exp(
            -r2 /
            SOURCE_RADIUS**2
        )
    )

    return Q


def apply_boundary_conditions(T):

    T[0, :] = T_INLET


    T[-1, :] = (
        T[-2, :]
        +
        GRADIENT_X_MAX *
        DX_GLOBAL
    )



    T[:, 0] = (
        T[:, 1]
        -
        GRADIENT_Y_MIN *
        DY_GLOBAL
    )




    T[:, -1] = (
        T[:, -2]
        +
        GRADIENT_Y_MAX *
        DY_GLOBAL
    )


    T[0, :] = T_INLET

    return T




def ftcs_step(
    T,
    dx,
    dy,
    dt,
    time_new,
    velocity,
    X,
    Y
):

    T_new = T.copy()


    # Fourier numbers

    Fo_x = (
        ALPHA *
        dt /
        dx**2
    )

    Fo_y = (
        ALPHA *
        dt /
        dy**2
    )


    # Stability protection

    if (
        Fo_x +
        Fo_y
        >
        0.5 + 1e-14
    ):

        raise RuntimeError(
            "FTCS instability detected: "
            f"Fo_x + Fo_y = "
            f"{Fo_x + Fo_y:.12f} > 0.5"
        )


    # Interior diffusion in x

    diffusion_x = (

        T[2:, 1:-1]

        -
        2.0 *
        T[1:-1, 1:-1]

        +
        T[:-2, 1:-1]

    )


    # Interior diffusion in y

    diffusion_y = (

        T[1:-1, 2:]

        -
        2.0 *
        T[1:-1, 1:-1]

        +
        T[1:-1, :-2]

    )


    # Moving Gaussian heat source

    Q = moving_heat_source(

        X[1:-1, 1:-1],

        Y[1:-1, 1:-1],

        time_new,

        velocity
    )


    # FTCS update

    T_new[1:-1, 1:-1] = (

        T[1:-1, 1:-1]

        +

        Fo_x *
        diffusion_x

        +

        Fo_y *
        diffusion_y

        +

        dt *
        Q /
        (
            RHO *
            CP
        )
    )


    # Boundary conditions

    T_new = apply_boundary_conditions(
        T_new
    )


    return T_new

# Checkpoint and field

def save_checkpoint(
    path,
    T,
    step,
    time_s,
    dx,
    dy,
    dt,
    cfg_hash
):

    np.savez_compressed(

        path,

        T=gpu_to_cpu(T),

        step=np.int64(
            step
        ),

        time_s=np.float64(
            time_s
        ),

        dx=np.float64(
            dx
        ),

        dy=np.float64(
            dy
        ),

        dt=np.float64(
            dt
        ),

        config_hash=np.array(
            cfg_hash
        )
    )


def load_checkpoint(
    path,
    expected_hash
):

    data = np.load(
        path,
        allow_pickle=False
    )

    stored_hash = str(
        data["config_hash"]
    )

    if stored_hash != expected_hash:

        raise RuntimeError(
            "\nCheckpoint found, "
            "but the configuration differs.\n"
            "Checkpoint will not be used."
        )


    T = data["T"]

    step = int(
        data["step"]
    )

    time_s = float(
        data["time_s"]
    )

    dx = float(
        data["dx"]
    )

    dy = float(
        data["dy"]
    )

    dt = float(
        data["dt"]
    )

    return (
        T,
        step,
        time_s,
        dx,
        dy,
        dt
    )


def snapshot_path(
    velocity,
    N,
    time_value
):

    folder = grid_folder(
        velocity,
        N
    )

    return os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        f"_snapshot_t{time_value:.0f}s.npz"
    )


def save_snapshot(
    velocity,
    N,
    time_value,
    x,
    y,
    T
):

    path = snapshot_path(
        velocity,
        N,
        time_value
    )

    np.savez_compressed(

        path,

        x=gpu_to_cpu(x),

        y=gpu_to_cpu(y),

        T=gpu_to_cpu(T),

        time_s=np.float64(
            time_value
        ),

        velocity=np.float64(
            velocity
        ),

        grid_N=np.int64(
            N
        )
    )

    return path


def load_snapshot(
    velocity,
    N,
    time_value
):

    path = snapshot_path(
        velocity,
        N,
        time_value
    )

    if not os.path.exists(path):

        return None


    data = np.load(
        path,
        allow_pickle=False
    )

    return {

        "x":
            data["x"],

        "y":
            data["y"],

        "T":
            data["T"],

        "time_s":
            float(
                data["time_s"]
            )
    }


def final_field_path(
    velocity,
    N
):

    folder = grid_folder(
        velocity,
        N
    )

    return os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        "_final_field.npz"
    )


def save_final_field(
    velocity,
    N,
    x,
    y,
    T,
    time_s
):

    path = final_field_path(
        velocity,
        N
    )

    np.savez_compressed(

        path,

        x=gpu_to_cpu(x),

        y=gpu_to_cpu(y),

        T=gpu_to_cpu(T),

        time_s=np.float64(
            time_s
        ),

        velocity=np.float64(
            velocity
        ),

        grid_N=np.int64(
            N
        )
    )

    return path


def load_final_field(
    velocity,
    N
):

    path = final_field_path(
        velocity,
        N
    )

    if not os.path.exists(path):

        return None


    return np.load(
        path,
        allow_pickle=False
    )


# Profiles, histories, and plots

#
# y = 8.5 mm
#

def centerline_profile(
    x,
    y,
    T,
    y_target=SOURCE_Y0
):

    profile = np.empty(
        len(x),
        dtype=np.float64
    )

    for i in range(len(x)):

        profile[i] = np.interp(

            y_target,

            y,

            T[i, :]
        )

    return profile


def save_centerline_csv(
    velocity,
    N,
    time_value,
    x,
    y,
    T
):

    folder = grid_folder(
        velocity,
        N
    )

    profile = centerline_profile(

        x,
        y,
        T,
        SOURCE_Y0
    )

    df = pd.DataFrame({

        "x_mm":
            x,

        "T_C":
            profile
    })

    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        f"_centerline_t{time_value:.0f}s.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return path


def save_combined_centerline_csv(
    velocity,
    N,
    output_times
):

    folder = grid_folder(
        velocity,
        N
    )

    first_snapshot = None

    for time_value in output_times:

        snap = load_snapshot(
            velocity,
            N,
            time_value
        )

        if snap is not None:

            first_snapshot = snap
            break


    if first_snapshot is None:

        return None


    data = {
        "x_mm":
            first_snapshot["x"]
    }


    for time_value in output_times:

        snap = load_snapshot(

            velocity,
            N,
            time_value
        )

        if snap is None:

            continue


        profile = centerline_profile(

            snap["x"],

            snap["y"],

            snap["T"],

            SOURCE_Y0
        )


        data[
            f"T_t{time_value:.0f}s_C"
        ] = profile


    if len(data) <= 1:

        return None


    df = pd.DataFrame(
        data
    )


    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        "_centerline_all_times.csv"
    )


    df.to_csv(
        path,
        index=False
    )

    return path

tg

def save_time_history(
    velocity,
    N,
    records
):

    folder = grid_folder(
        velocity,
        N
    )

    df = pd.DataFrame(
        records
    )


    if not df.empty:

        df = (

            df

            .drop_duplicates(
                subset="time_s",
                keep="last"
            )

            .sort_values(
                "time_s"
            )

            .reset_index(
                drop=True
            )
        )


    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        "_time_history.csv"
    )


    df.to_csv(
        path,
        index=False
    )

    return path


def save_contour_plot(
    velocity,
    N,
    time_value,
    x,
    y,
    T
):

    folder = grid_folder(
        velocity,
        N
    )

    X, Y = np.meshgrid(

        x,
        y,
        indexing="ij"
    )


    fig, ax = plt.subplots(

        figsize=(10.0, 7.2)
    )


    contour = ax.contourf(

        X,

        Y,

        T,

        levels=CONTOUR_LEVELS,

        cmap=COLORMAP
    )


    ax.set_xlim(
        X_MIN,
        X_MAX
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX
    )

    ax.set_aspect(
        "auto"
    )


    ax.set_xlabel(
        r"$x$ (mm)",
        fontsize=16
    )

    ax.set_ylabel(
        r"$y$ (mm)",
        fontsize=16
    )


    ax.set_title(

        rf"Temperature Field "
        rf"at $t={time_value:.1f}$ s "
        rf"($v={velocity:.1f}$ mm/s, "
        rf"$N={N}$)",

        fontsize=19,

        pad=10
    )


    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

        length=6,

        width=1.1
    )


    ax.tick_params(

        which="minor",

        length=3
    )


    ax.xaxis.set_minor_locator(
        AutoMinorLocator(5)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(5)
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            1.2
        )


    cbar = fig.colorbar(

        contour,

        ax=ax,

        orientation="vertical",

        fraction=0.045,

        pad=0.025
    )


    cbar.set_label(

        r"$T$ (°C)",

        rotation=270,

        labelpad=20,

        fontsize=16,

        va="center"
    )


    cbar.ax.tick_params(

        which="both",

        direction="in",

        length=5,

        width=1.0,

        labelsize=12
    )


    cbar.outline.set_linewidth(
        1.1
    )


    fig.subplots_adjust(

        left=0.095,

        right=0.875,

        bottom=0.105,

        top=0.900
    )


    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        f"_temperature_field_t{time_value:.0f}s.png"
    )


    fig.savefig(

        path,

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)

    return path


def save_centerline_plot(
    velocity,
    N,
    time_value,
    x,
    y,
    T
):

    folder = grid_folder(
        velocity,
        N
    )

    profile = centerline_profile(

        x,
        y,
        T,
        SOURCE_Y0
    )


    fig, ax = plt.subplots(

        figsize=(10.0, 6.5)
    )


    ax.plot(

        x,

        profile,

        linewidth=2.4
    )


    ax.set_xlabel(
        r"$x$ (mm)",
        fontsize=16
    )

    ax.set_ylabel(
        r"$T$ (°C)",
        fontsize=16
    )


    ax.set_title(

        rf"Centerline Temperature "
        rf"($t={time_value:.1f}$ s, "
        rf"$v={velocity:.1f}$ mm/s, "
        rf"$N={N}$)",

        fontsize=19,

        pad=10
    )


    ax.set_xlim(
        X_MIN,
        X_MAX
    )


    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

        length=6,

        width=1.1
    )


    ax.tick_params(

        which="minor",

        length=3
    )


    ax.xaxis.set_minor_locator(
        AutoMinorLocator(5)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(5)
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            1.2
        )


    fig.subplots_adjust(

        left=0.095,

        right=0.970,

        bottom=0.105,

        top=0.900
    )


    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        f"_centerline_t{time_value:.0f}s.png"
    )


    fig.savefig(

        path,

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)

    return path


#
# t = 1 - 10 s

def save_combined_centerline_plot(
    velocity,
    N,
    output_times
):

    folder = grid_folder(
        velocity,
        N
    )


    fig, ax = plt.subplots(

        figsize=(10.0, 6.5)
    )


    number_curves = 0


    for time_value in output_times:

        snap = load_snapshot(

            velocity,
            N,
            time_value
        )


        if snap is None:

            continue


        profile = centerline_profile(

            snap["x"],

            snap["y"],

            snap["T"],

            SOURCE_Y0
        )


        ax.plot(

            snap["x"],

            profile,

            linewidth=2.0,

            label=rf"$t={time_value:.0f}$ s"
        )


        number_curves += 1


    if number_curves == 0:

        plt.close(fig)

        return None


    ax.set_xlabel(
        r"$x$ (mm)",
        fontsize=16
    )

    ax.set_ylabel(
        r"$T$ (°C)",
        fontsize=16
    )


    ax.set_title(

        rf"Centerline Temperature Profiles "
        rf"($v={velocity:.1f}$ mm/s, "
        rf"$N={N}$)",

        fontsize=19,

        pad=10
    )


    ax.set_xlim(
        X_MIN,
        X_MAX
    )


    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

        length=6,

        width=1.1
    )


    ax.tick_params(

        which="minor",

        length=3
    )


    ax.xaxis.set_minor_locator(
        AutoMinorLocator(5)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(5)
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            1.2
        )


    ax.legend(

        loc="best",

        frameon=False,

        ncol=2
    )


    fig.subplots_adjust(

        left=0.095,

        right=0.970,

        bottom=0.105,

        top=0.900
    )


    path = os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        "_centerline_all_times.png"
    )


    fig.savefig(

        path,

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)

    return path

# Completion tracking

def complete_marker_path(
    velocity,
    N
):

    folder = grid_folder(
        velocity,
        N
    )

    return os.path.join(

        folder,

        f"{case_name(velocity, N)}"
        "_COMPLETE.txt"
    )


def case_is_complete(
    velocity,
    N
):

    marker = complete_marker_path(
        velocity,
        N
    )

    final = final_field_path(
        velocity,
        N
    )

    return (

        os.path.exists(marker)

        and

        os.path.exists(final)
    )


def write_complete_marker(
    velocity,
    N,
    current_time,
    Tmax,
    Tmin,
    Tmean,
    cfg_hash
):

    path = complete_marker_path(
        velocity,
        N
    )


    with open(
        path,
        "w"
    ) as f:

        f.write(
            "FTCS simulation completed.\n"
        )

        f.write(
            f"run_id = {RUN_ID}\n"
        )

        f.write(
            f"velocity_mm_s = "
            f"{velocity:.8f}\n"
        )

        f.write(
            f"grid_N = {N}\n"
        )

        f.write(
            f"time_s = "
            f"{current_time:.8f}\n"
        )

        f.write(
            f"Tmax_C = "
            f"{Tmax:.10f}\n"
        )

        f.write(
            f"Tmin_C = "
            f"{Tmin:.10f}\n"
        )

        f.write(
            f"Tmean_C = "
            f"{Tmean:.10f}\n"
        )

        f.write(
            f"config_hash = "
            f"{cfg_hash}\n"
        )

# Main FTCS simulation

global_start = time.time()

all_summary = []


print()
print("=" * 78)
print("FTCS BASELINE — GRID CONVERGENCE STUDY")
print("=" * 78)

print(
    f"Output directory : {BASE_DIR}"
)

print(
    f"Run ID           : {RUN_ID}"
)

print(
    f"Domain X         : "
    f"{X_MIN} - {X_MAX} mm"
)

print(
    f"Domain Y         : "
    f"{Y_MIN} - {Y_MAX} mm"
)

print(
    f"Time             : "
    f"{TIME_START} - {TIME_END} s"
)

print(
    f"Velocities       : "
    f"{VELOCITIES} mm/s"
)

print(
    f"Grid             : "
    f"{GRID_LIST}"
)

print(
    f"Initial T        : "
    f"{T_INITIAL} °C"
)

print(
    f"rho              : "
    f"{RHO:.6e}"
)

print(
    f"cp               : "
    f"{CP:.6f}"
)

print(
    f"k                : "
    f"{K:.6e}"
)

print(
    f"alpha            : "
    f"{ALPHA:.8f} mm²/s"
)

print(
    f"Source Q0        : "
    f"{SOURCE_Q0:.6f}"
)

print(
    f"Source radius    : "
    f"{SOURCE_RADIUS:.6f} mm"
)

print(
    f"Source center y  : "
    f"{SOURCE_Y0:.6f} mm"
)

print(
    f"Checkpoint       : "
    f"every {CHECKPOINT_INTERVAL:,} steps"
)

print("=" * 78)


for N in GRID_LIST:

    print()
    print()
    print("#" * 78)

    print(
        f"GRID = {N} x {N}"
    )

    print("#" * 78)


    # Number of nodes

    nx = int(N)

    ny = int(N)


    # Coordinates

    x = np.linspace(

        X_MIN,

        X_MAX,

        nx
    )

    y = np.linspace(

        Y_MIN,

        Y_MAX,

        ny
    )


    # Spatial increments

    dx = (
        x[1] -
        x[0]
    )

    dy = (
        y[1] -
        y[0]
    )


    # Meshgrid

    X, Y = cp.meshgrid(

        cp.asarray(x),

        cp.asarray(y),

        indexing="ij"
    )


    # Update global grid spacing

    DX_GLOBAL = dx

    DY_GLOBAL = dy


    # Stable time step

    (
        dt_stable,
        dt_candidate
    ) = calculate_time_step(
        dx,
        dy
    )


    # Number of steps

    number_steps = int(

        np.ceil(

            (
                TIME_END -
                TIME_START
            )
            /
            dt_candidate
        )
    )


    # Adjust dt to reach exactly 10 s

    dt = (

        TIME_END -
        TIME_START
    ) / number_steps


    # Fourier numbers

    Fo_x = (

        ALPHA *
        dt /
        dx**2
    )

    Fo_y = (

        ALPHA *
        dt /
        dy**2
    )

    Fo_sum = (
        Fo_x +
        Fo_y
    )


    # Print grid information

    print()

    print(
        f"dx             = "
        f"{dx:.10e} mm"
    )

    print(
        f"dy             = "
        f"{dy:.10e} mm"
    )

    print(
        f"dt_stable      = "
        f"{dt_stable:.10e} s"
    )

    print(
        f"dt_used        = "
        f"{dt:.10e} s"
    )

    print(
        f"Fo_x           = "
        f"{Fo_x:.10f}"
    )

    print(
        f"Fo_y           = "
        f"{Fo_y:.10f}"
    )

    print(
        f"Fo_x + Fo_y    = "
        f"{Fo_sum:.10f}"
    )

    print(
        f"Number of steps= "
        f"{number_steps:,}"
    )


    # Stability check

    if Fo_sum > 0.5:

        raise RuntimeError(

            "\nFTCS stability criterion violated.\n"

            f"Grid = {N} x {N}\n"

            f"Fo_x = {Fo_x}\n"

            f"Fo_y = {Fo_y}\n"

            f"Fo_x + Fo_y = {Fo_sum}\n\n"

            "FTCS criterion: "
            "Fo_x + Fo_y <= 0.5"
        )


    # VELOCITY LOOP

    for velocity in VELOCITIES:

        print()
        print()
        print("-" * 78)

        print(
            f"RUNNING: "
            f"v = {velocity:.1f} mm/s | "
            f"Grid = {N} x {N}"
        )

        print("-" * 78)


        # Folder

        folder = grid_folder(
            velocity,
            N
        )


        # Configuration

        (
            config,
            cfg_hash,
            config_path
        ) = save_configuration(

            velocity,

            N,

            dx,

            dy,

            dt,

            dt_stable,

            Fo_x,

            Fo_y,

            number_steps
        )


        # Case name

        tag = case_name(
            velocity,
            N
        )


        # Checkpoint path

        checkpoint_path = os.path.join(

            folder,

            f"{tag}_checkpoint.npz"
        )


        # CHECK IF ALREADY COMPLETE

        if case_is_complete(
            velocity,
            N
        ):

            print()
            print(
                "STATUS: COMPLETED"
            )

            print(
                "Case already completed."
            )

            print(
                "Case skipped."
            )


            final_data = load_final_field(

                velocity,

                N
            )


            if final_data is not None:

                T_final = final_data["T"]


                all_summary.append({

                    "velocity_mm_s":
                        velocity,

                    "grid_N":
                        N,

                    "dx_mm":
                        dx,

                    "dy_mm":
                        dy,

                    "dt_s":
                        dt,

                    "Fo_x":
                        Fo_x,

                    "Fo_y":
                        Fo_y,

                    "Fo_sum":
                        Fo_sum,

                    "Tmax_C":
                        float(
                            cp.max(
                                cp.asarray(T_final)
                            ).get()
                        ),

                    "Tmin_C":
                        float(
                            cp.min(
                                cp.asarray(T_final)
                            ).get()
                        ),

                    "Tmean_C":
                        float(
                            cp.mean(
                                cp.asarray(T_final)
                            ).get()
                        ),

                    "status":
                        "completed_existing"
                })


            continue


        # INITIALIZE / RESUME

        if os.path.exists(
            checkpoint_path
        ):

            print()
            print(
                "CHECKPOINT FOUND"
            )


            (
                T_cpu,
                step,
                current_time,
                dx_cp,
                dy_cp,
                dt_cp
            ) = load_checkpoint(

                checkpoint_path,

                cfg_hash
            )


            T = cp.asarray(T_cpu)

            # Verify checkpoint dimensions

            if not np.isclose(
                dx_cp,
                dx
            ):

                raise RuntimeError(
                    "Checkpoint dx differs."
                )


            if not np.isclose(
                dy_cp,
                dy
            ):

                raise RuntimeError(
                    "Checkpoint dy differs."
                )


            if not np.isclose(
                dt_cp,
                dt
            ):

                raise RuntimeError(
                    "Checkpoint dt differs."
                )


            if T.shape != (
                nx,
                ny
            ):

                raise RuntimeError(
                    "Checkpoint field shape "
                    "does not match the grid."
                )


            print(
                f"Resume step = "
                f"{step:,}"
            )

            print(
                f"Resume time = "
                f"{current_time:.8f} s"
            )


        else:

            print()
            print(
                "NO CHECKPOINT"
            )

            print(
                "Starting from "
                f"t = {TIME_START:.1f} s"
            )


            T = initial_field(
                nx,
                ny
            )


            T = apply_boundary_conditions(
                T
            )


            step = 0

            current_time = TIME_START


        # RECONSTRUCT HISTORY

        history_records = []


        for output_time in OUTPUT_TIMES:

            snap = load_snapshot(

                velocity,

                N,

                output_time
            )


            if snap is None:

                continue


            field = snap["T"]


            history_records.append({

                "time_s":
                    float(output_time),

                "Tmax_C":
                    float(
                        np.max(
                            field
                        )
                    ),

                "Tmin_C":
                    float(
                        np.min(
                            field
                        )
                    ),

                "Tmean_C":
                    float(
                        np.mean(
                            field
                        )
                    )
            })


        # COMPLETED SNAPSHOTS

        completed_output_times = set()


        for output_time in OUTPUT_TIMES:

            snap = load_snapshot(

                velocity,

                N,

                output_time
            )


            if snap is not None:

                completed_output_times.add(

                    round(
                        float(
                            output_time
                        ),
                        8
                    )
                )


        # TIMER

        simulation_start = time.time()


        # MAIN FTCS TIME LOOP

        while (

            current_time
            <
            TIME_END -
            1e-14

        ):

            step += 1


            # Next time

            next_time = min(

                TIME_END,

                current_time + dt
            )


            dt_step = (

                next_time -
                current_time
            )


            # FTCS update

            T = ftcs_step(

                T,

                dx,

                dy,

                dt_step,

                next_time,

                velocity,

                X,

                Y
            )


            current_time = next_time


            # SAVE OUTPUT SNAPSHOTS

            current_key = round(

                float(
                    current_time
                ),

                8
            )


            for output_time in OUTPUT_TIMES:

                output_key = round(

                    float(
                        output_time
                    ),

                    8
                )


                if (

                    current_key
                    >=
                    output_key

                    and

                    output_key
                    not in
                    completed_output_times

                ):

                    save_snapshot(

                        velocity,

                        N,

                        output_time,

                        x,

                        y,

                        T
                    )


                    completed_output_times.add(
                        output_key
                    )


                    Tmax_value = float(

                        np.max(T)
                    )


                    Tmin_value = float(

                        np.min(T)
                    )


                    Tmean_value = float(

                        np.mean(T)
                    )


                    history_records.append({

                        "time_s":
                            float(
                                output_time
                            ),

                        "Tmax_C":
                            Tmax_value,

                        "Tmin_C":
                            Tmin_value,

                        "Tmean_C":
                            Tmean_value
                    })


                    print(

                        f"  "
                        f"step={step:>9,} | "
                        f"t={output_time:>5.1f} s | "
                        f"Tmax={Tmax_value:>11.4f} °C | "
                        f"Tmean={Tmean_value:>11.4f} °C"
                    )


            # CHECKPOINT

            if (

                step %
                CHECKPOINT_INTERVAL
                ==
                0

                or

                current_time
                >=
                TIME_END -
                1e-14

            ):

                cp.cuda.Stream.null.synchronize()

                save_checkpoint(

                    checkpoint_path,

                    T,

                    step,

                    current_time,

                    dx,

                    dy,

                    dt,

                    cfg_hash
                )


        # FINAL STATISTICS

        cp.cuda.Stream.null.synchronize()

        Tmax_final = float(cp.max(T).get())

        Tmin_final = float(cp.min(T).get())

        Tmean_final = float(cp.mean(T).get())


        # RUNTIME

        runtime_seconds = (

            time.time()
            -
            simulation_start
        )

        runtime_minutes = (

            runtime_seconds /
            60.0
        )


        # SAVE FINAL FIELD

        save_final_field(

            velocity,

            N,

            x,

            y,

            T,

            current_time
        )


        # SAVE TIME HISTORY

        save_time_history(

            velocity,

            N,

            history_records
        )


        # SAVE CENTERLINE CSV

        for output_time in OUTPUT_TIMES:

            snap = load_snapshot(

                velocity,

                N,

                output_time
            )


            if snap is None:

                continue


            save_centerline_csv(

                velocity,

                N,

                output_time,

                snap["x"],

                snap["y"],

                snap["T"]
            )


        # SAVE COMBINED CENTERLINE CSV

        save_combined_centerline_csv(

            velocity,

            N,

            OUTPUT_TIMES
        )


        # SAVE PLOTS

        print()
        print(
            "Saving plots..."
        )


        for output_time in OUTPUT_TIMES:

            snap = load_snapshot(

                velocity,

                N,

                output_time
            )


            if snap is None:

                continue


            save_contour_plot(

                velocity,

                N,

                output_time,

                snap["x"],

                snap["y"],

                snap["T"]
            )


            save_centerline_plot(

                velocity,

                N,

                output_time,

                snap["x"],

                snap["y"],

                snap["T"]
            )


        # COMBINED CENTERLINE FIGURE

        save_combined_centerline_plot(

            velocity,

            N,

            OUTPUT_TIMES
        )


        # RUNTIME CSV

        runtime_path = os.path.join(

            folder,

            f"{tag}_runtime.csv"
        )


        pd.DataFrame({

            "velocity_mm_s":
                [velocity],

            "grid_N":
                [N],

            "runtime_seconds":
                [runtime_seconds],

            "runtime_minutes":
                [runtime_minutes],

            "number_steps":
                [step]

        }).to_csv(

            runtime_path,

            index=False
        )


        # CASE SUMMARY

        case_summary_path = os.path.join(

            folder,

            f"{tag}_summary.csv"
        )


        pd.DataFrame({

            "velocity_mm_s":
                [velocity],

            "grid_N":
                [N],

            "dx_mm":
                [dx],

            "dy_mm":
                [dy],

            "dt_stable_s":
                [dt_stable],

            "dt_s":
                [dt],

            "Fo_x":
                [Fo_x],

            "Fo_y":
                [Fo_y],

            "Fo_sum":
                [Fo_sum],

            "Tmax_C":
                [Tmax_final],

            "Tmin_C":
                [Tmin_final],

            "Tmean_C":
                [Tmean_final],

            "runtime_seconds":
                [runtime_seconds],

            "runtime_minutes":
                [runtime_minutes],

            "number_steps":
                [step],

            "centerline_y_mm":
                [SOURCE_Y0],

            "source_Q0":
                [SOURCE_Q0],

            "source_radius_mm":
                [SOURCE_RADIUS]

        }).to_csv(

            case_summary_path,

            index=False
        )


        # COMPLETE MARKER

        write_complete_marker(

            velocity,

            N,

            current_time,

            Tmax_final,

            Tmin_final,

            Tmean_final,

            cfg_hash
        )


        # REMOVE CHECKPOINT AFTER COMPLETION

        if os.path.exists(
            checkpoint_path
        ):

            os.remove(
                checkpoint_path
            )


        # ADD GLOBAL SUMMARY

        all_summary.append({

            "velocity_mm_s":
                velocity,

            "grid_N":
                N,

            "dx_mm":
                dx,

            "dy_mm":
                dy,

            "dt_stable_s":
                dt_stable,

            "dt_s":
                dt,

            "Fo_x":
                Fo_x,

            "Fo_y":
                Fo_y,

            "Fo_sum":
                Fo_sum,

            "Tmax_C":
                Tmax_final,

            "Tmin_C":
                Tmin_final,

            "Tmean_C":
                Tmean_final,

            "runtime_seconds":
                runtime_seconds,

            "runtime_minutes":
                runtime_minutes,

            "number_steps":
                step,

            "status":
                "completed"
        })


        # PRINT CASE RESULT

        print()
        print(
            "=" * 78
        )

        print(
            f"COMPLETED: "
            f"v={velocity:.1f} mm/s | "
            f"Grid={N} x {N}"
        )

        print(
            f"Tmax      = "
            f"{Tmax_final:.6f} °C"
        )

        print(
            f"Tmin      = "
            f"{Tmin_final:.6f} °C"
        )

        print(
            f"Tmean     = "
            f"{Tmean_final:.6f} °C"
        )

        print(
            f"Runtime    = "
            f"{runtime_minutes:.3f} min"
        )

        print(
            f"Steps      = "
            f"{step:,}"
        )

        print(
            f"Output     = "
            f"{folder}"
        )

        print(
            "=" * 78
        )

# Global summary and final output

global_runtime_seconds = (

    time.time()
    -
    global_start
)


global_runtime_minutes = (

    global_runtime_seconds /
    60.0
)


summary_df = pd.DataFrame(
    all_summary
)


if not summary_df.empty:

    summary_df = (

        summary_df

        .sort_values(

            [
                "velocity_mm_s",

                "grid_N"
            ]
        )

        .reset_index(
            drop=True
        )
    )


summary_path = os.path.join(

    BASE_DIR,

    "FTCS_Baseline_summary.csv"
)


summary_df.to_csv(

    summary_path,

    index=False
)


global_parameters = {

    "method":
        "FTCS",

    "run_id":
        RUN_ID,

    "domain_x_mm":
        [
            X_MIN,
            X_MAX
        ],

    "domain_y_mm":
        [
            Y_MIN,
            Y_MAX
        ],

    "time_range_s":
        [
            TIME_START,
            TIME_END
        ],

    "velocities_mm_s":
        VELOCITIES,

    "grid_list":
        GRID_LIST,

    "initial_temperature_C":
        T_INITIAL,

    "rho":
        RHO,

    "cp":
        CP,

    "thermal_conductivity":
        K,

    "thermal_diffusivity":
        ALPHA,

    "source_x0_mm":
        SOURCE_X0,

    "source_y0_mm":
        SOURCE_Y0,

    "source_Q0":
        SOURCE_Q0,

    "source_radius_mm":
        SOURCE_RADIUS,

    "boundary_conditions":
        {

            "x_min":
                "T = 20 C",

            "x_max":
                "dT/dx = -0.001 C/mm",

            "y_min":
                "dT/dy = +0.001 C/mm",

            "y_max":
                "dT/dy = -0.001 C/mm"
        },

    "centerline_y_mm":
        SOURCE_Y0,

    "checkpoint_interval":
        CHECKPOINT_INTERVAL,

    "output_times_s":
        [
            float(t)
            for t in OUTPUT_TIMES
        ],

    "contour_levels":
        CONTOUR_LEVELS,

    "colormap":
        COLORMAP,

    "grid_convergence_study":
        True,

    "global_runtime_seconds":
        global_runtime_seconds,

    "global_runtime_minutes":
        global_runtime_minutes
}


parameters_path = os.path.join(

    BASE_DIR,

    "FTCS_Baseline_parameters.json"
)


with open(

    parameters_path,

    "w"
) as f:

    json.dump(

        global_parameters,

        f,

        indent=4
    )


print()
print()
print("=" * 78)
print(
    "FTCS BASELINE — "
    "ALL SIMULATIONS COMPLETED"
)
print("=" * 78)

print()

print(
    f"Grid      : "
    f"{GRID_LIST}"
)

print(
    f"Velocity  : "
    f"{VELOCITIES} mm/s"
)

print(
    f"Time      : "
    f"{TIME_START} - {TIME_END} s"
)

print(
    f"Total runtime = "
    f"{global_runtime_minutes:.3f} minutes"
)

print()

print(
    "GLOBAL SUMMARY:"
)

print(
    summary_path
)

print()

print(
    "GLOBAL PARAMETERS:"
)

print(
    parameters_path
)

print()

print(
    "OUTPUT STRUCTURE:"
)

for velocity in VELOCITIES:

    print()

    print(
        f"velocity_{velocity:.0f}mm_s/"
    )

    for N in GRID_LIST:

        print(
            f"    grid_{N}/"
        )


print()
print("=" * 78)