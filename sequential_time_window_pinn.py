
# Environment and configuration
import os
import time
import pickle
from pathlib import Path
from functools import partial
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import jvp, vjp, value_and_grad
from tqdm import trange

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    }
)



try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception:
    pass


base_dir = Path(
    "/content/drive/MyDrive/09_09_PINN_Sequential_Time_Window"
)

base_dir.mkdir(
    parents=True,
    exist_ok=True
)


xmin = 0.0
xmax = 35.0

ymin = 0.0
ymax = 17.0

tmin = 0.0
tmax = 10.0


velocities = [1.0, 2.0, 3.0]


rho = 7.6e-6
c = 658.0
k = 25e-3


NC = 128**2
NI = 96**2
NB = 96**2

NC_TEST = 500

SEED = 444

LR = 1e-3

EPOCHS = 8000

N_LAYERS = 9

FEATURES = 128

delt = 1.0

CHECKPOINT_EVERY = 500

centerline_y = 8.5

# Sampling and run configuration
# Sampling configuration

SOURCE_FRAC = 0.3

SIGMA_FACTOR = 3.0


runs = int(
    round(
        (tmax - tmin) / delt
    )
)


comparison_times = np.arange(
    tmin + delt,
    tmax + 1e-12,
    delt
)


feat_sizes = tuple(
    [FEATURES] * (N_LAYERS - 1)
    +
    [1]
)


if runs != len(
    comparison_times
):

    raise ValueError(
        "The time-window configuration is inconsistent."
    )


if not np.isclose(
    delt * runs,
    tmax - tmin
):

    raise ValueError(
        "The time-window size does not divide the simulation time."
    )


print(
    "Sequential Time-Window PINN"
)

print(
    f"Output directory : {base_dir}"
)

print(
    f"Domain           : "
    f"{xmin:.1f}-{xmax:.1f} x "
    f"{ymin:.1f}-{ymax:.1f} mm"
)

print(
    f"Time range       : "
    f"{tmin:.1f}-{tmax:.1f} s"
)

print(
    f"Velocities       : "
    f"{velocities} mm/s"
)

print(
    f"Collocation      : "
    f"{NC:,}"
)

print(
    f"Initial points   : "
    f"{NI:,}"
)

print(
    f"Boundary points  : "
    f"{NB:,}"
)

print(
    f"Architecture     : "
    f"{N_LAYERS} layers x "
    f"{FEATURES} neurons"
)

print(
    f"Epochs/window    : "
    f"{EPOCHS:,}"
)

print(
    f"Number of windows: "
    f"{runs}"
)

print(
    f"Window size      : "
    f"{delt:.1f} s"
)

print(
    f"Checkpoint       : "
    f"every {CHECKPOINT_EVERY:,} epochs"
)

print(
    f"Source-focused   : "
    f"{SOURCE_FRAC * 100:.0f}% of collocation points "
    f"(sigma={SIGMA_FACTOR:.1f} r0)"
)

print(
    f"JAX version      : "
    f"{jax.__version__}"
)

print(
    f"JAX backend      : "
    f"{jax.default_backend()}"
)

print(
    f"JAX devices      : "
    f"{jax.devices()}"
)

# PINN architecture and derivatives
class PINN(nn.Module):

    features: Sequence[int]

    @nn.compact
    def __call__(
        self,
        x,
        y,
        t
    ):

        inputs = jnp.concatenate(
            [
                x,
                y,
                t
            ],
            axis=1
        )

        init = nn.initializers.glorot_normal()

        for features in self.features[:-1]:

            inputs = nn.Dense(
                features,
                kernel_init=init
            )(inputs)

            inputs = nn.activation.tanh(
                inputs
            )

        return nn.Dense(
            self.features[-1],
            kernel_init=init
        )(inputs)


def hvp_fwdfwd(
    f,
    primals,
    tangents
):

    g = lambda p: jvp(
        f,
        (p,),
        tangents
    )[1]

    return jvp(
        g,
        primals,
        tangents
    )[1]

# Moving heat source
def q(
    pos,
    t,
    velocity
):

    r0 = 1.0

    q0 = 5.0

    muy = (
        8.5
        *
        jnp.ones(
            (t.shape[0], 1)
        )
    )

    mux = (
        34.0
        *
        jnp.ones(
            (t.shape[0], 1)
        )
        -
        velocity * t
    )

    x_source = (
        pos[:, 0]
        .reshape(
            -1,
            1
        )
    )

    y_source = (
        pos[:, 1]
        .reshape(
            -1,
            1
        )
    )

    r2 = (
        jnp.square(
            x_source - mux
        )
        +
        jnp.square(
            y_source - muy
        )
    )

    return (
        q0
        *
        jnp.exp(
            -r2 / r0**2
        )
    )

# Source-focused collocation sampling
# Source-focused collocation sampling

def generate_collocation_points(
    nc,
    key,
    velocity,
    t_start,
    t_end,
    source_frac=SOURCE_FRAC,
    sigma_factor=SIGMA_FACTOR
):

    keys = jax.random.split(
        key,
        6
    )

    r0 = 1.0

    sigma = sigma_factor * r0

    nc_source = int(
        nc * source_frac
    )

    nc_uniform = (
        nc
        -
        nc_source
    )


    # Uniform samples cover the full domain.

    tc_uniform = jax.random.uniform(
        keys[0],
        (nc_uniform, 1),
        minval=t_start,
        maxval=t_end
    )

    xc_uniform = jax.random.uniform(
        keys[1],
        (nc_uniform, 1),
        minval=xmin,
        maxval=xmax
    )

    yc_uniform = jax.random.uniform(
        keys[2],
        (nc_uniform, 1),
        minval=ymin,
        maxval=ymax
    )


    # Source-focused samples follow the moving heat source.

    tc_source = jax.random.uniform(
        keys[3],
        (nc_source, 1),
        minval=t_start,
        maxval=t_end
    )

    mux_source = (
        34.0
        *
        jnp.ones(
            (nc_source, 1)
        )
        -
        velocity * tc_source
    )

    muy_source = (
        8.5
        *
        jnp.ones(
            (nc_source, 1)
        )
    )

    xc_source = (
        mux_source
        +
        sigma * jax.random.normal(
            keys[4],
            (nc_source, 1)
        )
    )

    yc_source = (
        muy_source
        +
        sigma * jax.random.normal(
            keys[5],
            (nc_source, 1)
        )
    )

    # Keep sampled points inside the domain.
    xc_source = jnp.clip(
        xc_source,
        xmin,
        xmax
    )

    yc_source = jnp.clip(
        yc_source,
        ymin,
        ymax
    )


    # Combine the two sampling sets.

    tc = jnp.concatenate(
        [tc_uniform, tc_source],
        axis=0
    )

    xc = jnp.concatenate(
        [xc_uniform, xc_source],
        axis=0
    )

    yc = jnp.concatenate(
        [yc_uniform, yc_source],
        axis=0
    )

    uc = q(
        jnp.concatenate(
            [xc, yc],
            axis=1
        ),
        tc,
        velocity
    )

    return tc, xc, yc, uc

# Physics-informed loss and optimization step
def pinn_loss(
    apply_fn,
    *train_data
):

    (
        tc,
        xc,
        yc,
        uc,
        ti,
        xi,
        yi,
        ui,
        tb,
        xb,
        yb,
        ub
    ) = train_data


    dirichlet_mask = (
        xb == xmin
    )


    dirichlet_indices = jnp.where(
        dirichlet_mask
    )[0]


    neumann_indices = jnp.where(
        ~dirichlet_mask
    )[0]


    xdb = xb[
        dirichlet_indices
    ]

    ydb = yb[
        dirichlet_indices
    ]

    tdb = tb[
        dirichlet_indices
    ]

    udb = ub[
        dirichlet_indices
    ]


    xnb = xb[
        neumann_indices
    ]

    ynb = yb[
        neumann_indices
    ]

    tnb = tb[
        neumann_indices
    ]

    unb = ub[
        neumann_indices
    ]


    def residual_loss(
        params
    ):

        u = apply_fn(
            params,
            xc,
            yc,
            tc
        )

        ones = jnp.ones(
            (
                u.shape[0],
                1
            )
        )


        ut = vjp(
            lambda t_var:
                apply_fn(
                    params,
                    xc,
                    yc,
                    t_var
                ),
            tc
        )[1](
            ones
        )[0]


        uxx = hvp_fwdfwd(
            lambda x_var:
                apply_fn(
                    params,
                    x_var,
                    yc,
                    tc
                ),
            (xc,),
            (ones,)
        )


        uyy = hvp_fwdfwd(
            lambda y_var:
                apply_fn(
                    params,
                    xc,
                    y_var,
                    tc
                ),
            (yc,),
            (ones,)
        )


        residual = (
            rho * c * ut / k
            -
            uxx
            -
            uyy
            -
            uc / k
        )


        return jnp.mean(
            jnp.square(
                residual
            )
        )


    def initial_loss(
        params
    ):

        prediction = apply_fn(
            params,
            xi,
            yi,
            ti
        )

        return jnp.mean(
            jnp.square(
                prediction - ui
            )
        )


    def boundary_loss_neumann(
        params
    ):

        u = apply_fn(
            params,
            xnb,
            ynb,
            tnb
        )

        ones = jnp.ones(
            (
                u.shape[0],
                1
            )
        )


        ux = vjp(
            lambda x_var:
                apply_fn(
                    params,
                    x_var,
                    ynb,
                    tnb
                ),
            xnb
        )[1](
            ones
        )[0]


        uy = vjp(
            lambda y_var:
                apply_fn(
                    params,
                    xnb,
                    y_var,
                    tnb
                ),
            ynb
        )[1](
            ones
        )[0]


        result = jnp.where(
            xnb == xmax,
            ux - unb,
            uy - unb
        )


        return jnp.mean(
            jnp.square(
                result
            )
        )


    def boundary_loss_dirichlet(
        params
    ):

        prediction = apply_fn(
            params,
            xdb,
            ydb,
            tdb
        )

        return jnp.mean(
            jnp.square(
                prediction - udb
            )
        )


    def total_loss(
        params
    ):

        return (
            1e3
            *
            residual_loss(
                params
            )
            +
            250.0
            *
            initial_loss(
                params
            )
            +
            250.0
            *
            boundary_loss_neumann(
                params
            )
            +
            250.0
            *
            boundary_loss_dirichlet(
                params
            )
        )


    return total_loss


@partial(
    jax.jit,
    static_argnums=(0,)
)
def update_model(
    optim,
    gradient,
    params,
    state
):

    updates, state = optim.update(
        gradient,
        state
    )

    params = optax.apply_updates(
        params,
        updates
    )

    return (
        params,
        state
    )

# Training data generation and time-window transfer
def pinn_training_data_generator(
    nc,
    ni,
    nb,
    key,
    velocity,
    t_start,
    t_end
):

    keys = jax.random.split(
        key,
        12
    )


    # Generate PDE collocation points.

    tc, xc, yc, uc = generate_collocation_points(
        nc,
        keys[0],
        velocity,
        t_start,
        t_end
    )


    ti = jnp.full(
        (
            ni,
            1
        ),
        t_start
    )


    xi = jax.random.uniform(
        keys[1],
        (
            ni,
            1
        ),
        minval=xmin,
        maxval=xmax
    )


    yi = jax.random.uniform(
        keys[2],
        (
            ni,
            1
        ),
        minval=ymin,
        maxval=ymax
    )


    ui = (
        20.0
        *
        jnp.ones(
            (
                ni,
                1
            )
        )
    )


    tb = jnp.concatenate(
        [

            jax.random.uniform(
                keys[3],
                (
                    nb,
                    1
                ),
                minval=t_start,
                maxval=t_end
            ),

            jax.random.uniform(
                keys[4],
                (
                    nb,
                    1
                ),
                minval=t_start,
                maxval=t_end
            ),

            jax.random.uniform(
                keys[5],
                (
                    nb,
                    1
                ),
                minval=t_start,
                maxval=t_end
            ),

            jax.random.uniform(
                keys[6],
                (
                    nb,
                    1
                ),
                minval=t_start,
                maxval=t_end
            )

        ]
    )


    xb = jnp.concatenate(
        [

            jnp.full(
                (
                    nb,
                    1
                ),
                xmin
            ),

            jnp.full(
                (
                    nb,
                    1
                ),
                xmax
            ),

            jax.random.uniform(
                keys[7],
                (
                    nb,
                    1
                ),
                minval=xmin,
                maxval=xmax
            ),

            jax.random.uniform(
                keys[8],
                (
                    nb,
                    1
                ),
                minval=xmin,
                maxval=xmax
            )

        ]
    )


    yb = jnp.concatenate(
        [

            jax.random.uniform(
                keys[9],
                (
                    nb,
                    1
                ),
                minval=ymin,
                maxval=ymax
            ),

            jax.random.uniform(
                keys[10],
                (
                    nb,
                    1
                ),
                minval=ymin,
                maxval=ymax
            ),

            jnp.full(
                (
                    nb,
                    1
                ),
                ymin
            ),

            jnp.full(
                (
                    nb,
                    1
                ),
                ymax
            )

        ]
    )


    ub = jnp.concatenate(
        [

            20.0
            *
            jnp.ones(
                (
                    nb,
                    1
                )
            ),

            -0.001
            *
            jnp.ones(
                (
                    nb,
                    1
                )
            ),

            0.001
            *
            jnp.ones(
                (
                    nb,
                    1
                )
            ),

            -0.001
            *
            jnp.ones(
                (
                    nb,
                    1
                )
            )

        ]
    )


    return (
        tc,
        xc,
        yc,
        uc,
        ti,
        xi,
        yi,
        ui,
        tb,
        xb,
        yb,
        ub
    )


def shift_training_data(
    train_data,
    params,
    apply_fn,
    velocity,
    delt,
    key,
    t_start_new,
    t_end_new
):

    (
        tc,
        xc,
        yc,
        uc,
        ti,
        xi,
        yi,
        ui,
        tb,
        xb,
        yb,
        ub
    ) = train_data


    keys = jax.random.split(
        key,
        1
    )


    # Regenerate collocation points for the new window.

    tc_new, xc_new, yc_new, uc_new = generate_collocation_points(
        tc.shape[0],
        keys[0],
        velocity,
        t_start_new,
        t_end_new
    )


    # Reuse spatial initial-condition points and update their targets.

    ti_new = (
        ti
        +
        delt
    )


    ui_new = apply_fn(
        params,
        xi,
        yi,
        ti_new
    )


    # Shift boundary times to the new window.

    tb_new = (
        tb
        +
        delt
    )


    return (
        tc_new,
        xc_new,
        yc_new,
        uc_new,
        ti_new,
        xi,
        yi,
        ui_new,
        tb_new,
        xb,
        yb,
        ub
    )

# Dataset and checkpoint utilities
def save_dataset(
    path,
    train_data
):

    arrays = [
        np.asarray(
            jax.device_get(
                a
            )
        )
        for a in train_data
    ]


    np.savez_compressed(
        path,

        tc=arrays[0],
        xc=arrays[1],
        yc=arrays[2],
        uc=arrays[3],

        ti=arrays[4],
        xi=arrays[5],
        yi=arrays[6],
        ui=arrays[7],

        tb=arrays[8],
        xb=arrays[9],
        yb=arrays[10],
        ub=arrays[11]
    )


def load_dataset(
    path
):

    with np.load(
        path
    ) as data:

        return tuple(
            jnp.asarray(
                data[name]
            )
            for name in [

                "tc",
                "xc",
                "yc",
                "uc",

                "ti",
                "xi",
                "yi",
                "ui",

                "tb",
                "xb",
                "yb",
                "ub"

            ]
        )


def configuration_signature():
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "tmin": tmin,
        "tmax": tmax,
        "velocities": tuple(velocities),
        "rho": rho,
        "c": c,
        "k": k,
        "r0": 1.0,
        "q0": 5.0,
        "muy": 8.5,
        "source_x0": 34.0,
        "NC": NC,
        "NI": NI,
        "NB": NB,
        "SEED": SEED,
        "LR": LR,
        "N_LAYERS": N_LAYERS,
        "FEATURES": FEATURES,
        "delt": delt,
        "SOURCE_FRAC": SOURCE_FRAC,
        "SIGMA_FACTOR": SIGMA_FACTOR,
    }


def checkpoint_matches_configuration(checkpoint):
    saved = checkpoint.get("configuration")
    if saved is None:
        return False
    current = configuration_signature()
    for key, value in current.items():
        if key not in saved:
            return False
        if isinstance(value, tuple):
            if tuple(saved[key]) != value:
                return False
        elif isinstance(value, list):
            if list(saved[key]) != value:
                return False
        elif not np.isclose(float(saved[key]), float(value), rtol=0.0, atol=1e-12):
            return False
    return True


def save_checkpoint(
    path,
    payload
):

    tmp_path = Path(
        str(path)
        +
        ".tmp"
    )


    with open(
        tmp_path,
        "wb"
    ) as f:

        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    os.replace(
        tmp_path,
        path
    )


def load_checkpoint(
    path
):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(
            f
        )


def save_csv(
    path,
    array,
    header
):

    np.savetxt(
        path,
        np.asarray(array),
        delimiter=",",
        header=header,
        comments=""
    )

# Evaluation utilities
def pinn_test_data_generator(
    nc_test,
    ts
):

    t_axis = jax.lax.stop_gradient(
        jnp.array(
            [
                ts
            ]
        )
    )


    x_axis = jax.lax.stop_gradient(
        jnp.linspace(
            xmin,
            xmax,
            nc_test
        )
    )


    y_axis = jax.lax.stop_gradient(
        jnp.linspace(
            ymin,
            ymax,
            nc_test
        )
    )


    tm, xm, ym = jnp.meshgrid(
        t_axis,
        x_axis,
        y_axis,
        indexing="ij"
    )


    return (
        tm.reshape(
            -1,
            1
        ),
        xm.reshape(
            -1,
            1
        ),
        ym.reshape(
            -1,
            1
        ),
        tm,
        xm,
        ym
    )


def pinn_centerline_data_generator(
    nc_test,
    ts,
    y_position
):

    x_eval = jnp.linspace(
        xmin,
        xmax,
        nc_test
    ).reshape(
        -1,
        1
    )


    t_eval = (
        ts
        *
        jnp.ones(
            (
                nc_test,
                1
            )
        )
    )


    y_eval = (
        y_position
        *
        jnp.ones(
            (
                nc_test,
                1
            )
        )
    )


    return (
        jax.lax.stop_gradient(
            t_eval
        ),
        jax.lax.stop_gradient(
            x_eval
        ),
        jax.lax.stop_gradient(
            y_eval
        )
    )


def evaluate_window(
    params,
    apply_fn,
    t_eval
):

    (
        t_eval_data,
        x_eval,
        y_eval,
        tm,
        xm,
        ym
    ) = pinn_test_data_generator(
        NC_TEST,
        t_eval
    )


    u_eval = apply_fn(
        params,
        x_eval,
        y_eval,
        t_eval_data
    )


    temperature_field = np.asarray(
        jax.device_get(
            u_eval
        )
    ).reshape(
        NC_TEST,
        NC_TEST
    )


    return (
        temperature_field,
        np.asarray(
            xm[0]
        ),
        np.asarray(
            ym[0]
        ),
        np.asarray(
            xm
        ),
        np.asarray(
            ym
        )
    )

# Window-level plots
def save_window_plots(
    velocity_dir,
    velocity,
    t_eval,
    temperature_field,
    x_grid,
    y_grid,
    centerline,
    loss_history_array
):

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05
        }
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    cf = ax.contourf(
        x_grid,
        y_grid,
        temperature_field,
        levels=100,
        cmap="coolwarm"
    )


    cbar = fig.colorbar(
        cf,
        ax=ax,
        pad=0.02
    )


    cbar.set_label(
        r"$T$ (°C)"
    )


    ax.set_xlabel(
        "$x$ (mm)"
    )


    ax.set_ylabel(
        "$y$ (mm)"
    )


    ax.set_title(
        rf"$t={t_eval:.1f}$ s "
        rf"($v={velocity:.1f}$ mm/s)"
    )


    ax.tick_params(
        direction="in",
        width=0.8,
        length=4
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_contour_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.png",
        facecolor="white"
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_contour_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.pdf",
        facecolor="white"
    )


    plt.close(
        fig
    )


    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )


    ax.plot(
        np.arange(
            1,
            len(loss_history_array) + 1
        ),
        loss_history_array,
        linewidth=1.2,
        color="#1f4e79"
    )


    ax.set_xlabel(
        "Epoch"
    )


    ax.set_ylabel(
        r"$\ln(\mathcal{L})$"
    )


    ax.set_title(
        rf"Training Convergence "
        rf"($t={t_eval:.1f}$ s, "
        rf"$v={velocity:.1f}$ mm/s)"
    )


    ax.tick_params(
        direction="in",
        width=0.8,
        length=4
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_loss_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.png",
        facecolor="white"
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_loss_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.pdf",
        facecolor="white"
    )


    plt.close(
        fig
    )


    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )


    ax.plot(
        np.linspace(
            xmin,
            xmax,
            NC_TEST
        ),
        centerline,
        linewidth=2.0
    )


    ax.set_xlabel(
        "$x$ (mm)"
    )


    ax.set_ylabel(
        r"$T$ (°C)"
    )


    ax.set_title(
        rf"Centerline Temperature "
        rf"($t={t_eval:.1f}$ s, "
        rf"$v={velocity:.1f}$ mm/s)"
    )


    ax.tick_params(
        direction="in",
        width=0.8,
        length=4
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_centerline_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.png",
        facecolor="white"
    )


    fig.savefig(
        velocity_dir
        /
        f"SequentialTimeWindow_centerline_"
        f"v{velocity:.0f}mm_s_"
        f"t{t_eval:.0f}s.pdf",
        facecolor="white"
    )


    plt.close(
        fig
    )

# Progress tables and parameter export
def save_progress_tables(
    velocity_dir,
    velocity,
    time_history,
    runtime_history,
    window_loss_history,
    centerline_history,
    all_loss_histories
):

    time_history_array = np.asarray(
        time_history,
        dtype=float
    )

    runtime_history_array = np.asarray(
        runtime_history,
        dtype=float
    )

    window_loss_array = np.asarray(
        window_loss_history,
        dtype=float
    )

    if centerline_history:
        centerline_array = np.asarray(
            centerline_history,
            dtype=float
        )
    else:
        centerline_array = np.empty(
            (0, NC_TEST),
            dtype=float
        )

    if len(time_history_array) > 0:
        save_csv(
            velocity_dir /
            f"time_history_v{velocity:.0f}mm_s.csv",
            time_history_array,
            "time_s,Tmax_C,Tmin_C,Tmean_C"
        )

    if len(runtime_history_array) > 0:
        save_csv(
            velocity_dir /
            "SequentialTimeWindow_runtime_history.csv",
            runtime_history_array,
            "window,time_s,runtime_s,runtime_per_epoch_s"
        )

    if len(window_loss_array) > 0:
        save_csv(
            velocity_dir /
            "SequentialTimeWindow_window_loss.csv",
            window_loss_array,
            "window,time_s,final_ln_loss"
        )

    if centerline_array.size > 0:
        n_profiles = centerline_array.shape[0]
        valid_times = comparison_times[:n_profiles]

        centerline_output = np.column_stack(
            [
                np.linspace(
                    xmin,
                    xmax,
                    NC_TEST
                ),
                centerline_array.T
            ]
        )

        header = (
            "x_mm,"
            +
            ",".join(
                f"T_{int(t)}s_C"
                for t in valid_times
            )
        )

        save_csv(
            velocity_dir /
            "SequentialTimeWindow_centerline_all.csv",
            centerline_output,
            header
        )

    loss_rows = []

    for window_index, history in enumerate(
        all_loss_histories
    ):

        history_array = np.asarray(
            history,
            dtype=float
        ).reshape(-1)

        if history_array.size == 0:
            continue

        if window_index < len(comparison_times):
            time_value = comparison_times[window_index]
        else:
            time_value = np.nan

        for epoch_index, loss_value in enumerate(
            history_array,
            start=1
        ):
            loss_rows.append(
                [
                    window_index + 1,
                    time_value,
                    epoch_index,
                    loss_value
                ]
            )

    if loss_rows:
        save_csv(
            velocity_dir /
            "SequentialTimeWindow_loss_all.csv",
            np.asarray(
                loss_rows,
                dtype=float
            ),
            "window,time_s,epoch,ln_loss"
        )


def save_parameter_file(
    velocity_dir,
    velocity
):

    parameters = np.array(
        [[
            velocity,
            rho,
            c,
            k,

            34.0,
            8.5,
            1.0,
            5.0,

            xmin,
            xmax,
            ymin,
            ymax,

            tmin,
            tmax,

            NC,
            NI,
            NB,
            NC_TEST,

            EPOCHS,
            N_LAYERS,
            FEATURES,

            LR,
            delt,
            runs,

            CHECKPOINT_EVERY,

            centerline_y,

            SOURCE_FRAC,

            SIGMA_FACTOR
        ]]
    )


    header = (
        "velocity_mm_s,"
        "rho,"
        "specific_heat,"
        "thermal_conductivity,"
        "source_x0_mm,"
        "source_y_mm,"
        "source_radius_mm,"
        "source_strength,"
        "xmin_mm,"
        "xmax_mm,"
        "ymin_mm,"
        "ymax_mm,"
        "tmin_s,"
        "tmax_s,"
        "NC,"
        "NI,"
        "NB,"
        "NC_TEST,"
        "epochs_per_window,"
        "layers,"
        "features,"
        "learning_rate,"
        "window_size_s,"
        "number_of_windows,"
        "checkpoint_every,"
        "centerline_y_mm,"
        "source_frac,"
        "sigma_factor"
    )


    save_csv(
        velocity_dir /
        "simulation_parameters.csv",

        parameters,

        header
    )

# Final plots
def save_final_plots(
    velocity_dir,
    velocity,
    time_history,
    centerline_history,
    all_loss_histories
):

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05
        }
    )

    time_history_array = np.asarray(
        time_history,
        dtype=float
    )

    if centerline_history:
        centerline_array = np.asarray(
            centerline_history,
            dtype=float
        )
    else:
        centerline_array = np.empty(
            (0, NC_TEST),
            dtype=float
        )

    if len(time_history_array) > 0:

        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )

        ax.plot(
            time_history_array[:, 0] * 0 + np.arange(1, len(time_history_array) + 1) * EPOCHS,
            time_history_array[:, 1],
            marker="o",
            linewidth=1.5,
            color="#c0392b",
            label=r"$T_{\mathrm{max}}$"
        )

        ax.plot(
            time_history_array[:, 0] * 0 + np.arange(1, len(time_history_array) + 1) * EPOCHS,
            time_history_array[:, 3],
            marker="s",
            linewidth=1.5,
            color="#1f4e79",
            label=r"$T_{\mathrm{mean}}$"
        )

        ax.set_xlabel(
            "Cumulative Epoch"
        )

        ax.set_ylabel(
            "T (°C)"
        )

        ax.set_title(
            rf"Temperature Convergence "
            rf"($v={velocity:.1f}$ mm/s)"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_convergence_v{velocity:.0f}mm_s.png",
            facecolor="white"
        )

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_convergence_v{velocity:.0f}mm_s.pdf",
            facecolor="white"
        )

        plt.close(
            fig
        )


    if centerline_array.size > 0:

        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )

        for i, ts in enumerate(
            comparison_times[
                :centerline_array.shape[0]
            ]
        ):

            ax.plot(
                np.linspace(
                    xmin,
                    xmax,
                    NC_TEST
                ),
                centerline_array[i],
                linewidth=1.6,
                label=rf"$t={ts:.0f}$ s"
            )

        ax.set_xlabel(
            "$x$ (mm)"
        )

        ax.set_ylabel(
            "T (°C)"
        )

        ax.set_title(
            rf"Centerline Temperature Profiles "
            rf"($v={velocity:.1f}$ mm/s)"
        )

        ax.legend(
            frameon=False,
            ncol=2
        )

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_centerline_v{velocity:.0f}mm_s.png",
            facecolor="white"
        )

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_centerline_v{velocity:.0f}mm_s.pdf",
            facecolor="white"
        )

        plt.close(
            fig
        )


    final_window_index = len(centerline_array)

    if final_window_index > 0:

        final_time = comparison_times[
            final_window_index - 1
        ]

        dataset_temperature_path = (
            velocity_dir
            /
            f"temperature_field_"
            f"v{velocity:.0f}mm_s_"
            f"t{final_time:.0f}s.npy"
        )

        if dataset_temperature_path.exists():

            final_temperature_field = np.load(
                dataset_temperature_path
            )

            x_final = np.linspace(
                xmin,
                xmax,
                NC_TEST
            )

            y_final = np.linspace(
                ymin,
                ymax,
                NC_TEST
            )

            X_final, Y_final = np.meshgrid(
                x_final,
                y_final,
                indexing="ij"
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            cf = ax.contourf(
                X_final,
                Y_final,
                final_temperature_field,
                levels=100,
                cmap="coolwarm"
            )

            cbar = fig.colorbar(
                cf,
                ax=ax,
                pad=0.02
            )

            cbar.set_label(
                "Temperature (°C)"
            )

            ax.set_xlabel(
                "$x$ (mm)"
            )

            ax.set_ylabel(
                "$y$ (mm)"
            )

            ax.set_aspect(
                "equal"
            )

            ax.set_title(
                rf"$t={final_time:.0f}$ s "
                rf"($v={velocity:.1f}$ mm/s)"
            )

            fig.tight_layout()

            fig.savefig(
                velocity_dir /
                f"SequentialTimeWindow_contour_final_"
                f"v{velocity:.0f}mm_s.png",
                facecolor="white"
            )

            fig.savefig(
                velocity_dir /
                f"SequentialTimeWindow_contour_final_"
                f"v{velocity:.0f}mm_s.pdf",
                facecolor="white"
            )

            plt.close(
                fig
            )


    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    cumulative_offset = 0
    plotted = False

    for i, history in enumerate(
        all_loss_histories
    ):

        history_array = np.asarray(
            history,
            dtype=float
        ).reshape(-1)

        if history_array.size == 0:
            continue

        epoch_axis = (
            np.arange(
                1,
                history_array.size + 1
            )
            +
            cumulative_offset
        )

        ax.plot(
            epoch_axis,
            history_array,
            linewidth=1.2,
            color="#1f4e79",
            label=None
        )

        cumulative_offset += history_array.size
        plotted = True

    if plotted:

        ax.set_xlabel(
            "Cumulative Epoch"
        )

        ax.set_ylabel(
            r"$\ln(\mathcal{L})$"
        )

        ax.set_title(
            rf"Training Convergence "
            rf"($v={velocity:.1f}$ mm/s)"
        )

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_loss_v{velocity:.0f}mm_s.png",
            facecolor="white"
        )

        fig.savefig(
            velocity_dir /
            f"SequentialTimeWindow_loss_v{velocity:.0f}mm_s.pdf",
            facecolor="white"
        )

    plt.close(
        fig
    )

# Model initialization
def initialize_model():

    key = jax.random.PRNGKey(
        SEED
    )


    key, subkey = jax.random.split(
        key
    )


    model = PINN(
        feat_sizes
    )


    params = model.init(
        subkey,

        jnp.ones(
            (
                NC,
                1
            )
        ),

        jnp.ones(
            (
                NC,
                1
            )
        ),

        jnp.ones(
            (
                NC,
                1
            )
        )
    )


    optim = optax.adam(
        LR
    )


    state = optim.init(
        params
    )


    apply_fn = jax.jit(
        model.apply
    )


    return (
        key,
        model,
        params,
        optim,
        state,
        apply_fn
    )

# Sequential training loop
def train_velocity(
    velocity
):

    velocity_dir = (
        base_dir
        /
        f"velocity_{velocity:.0f}mm_s"
    )


    velocity_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    save_parameter_file(
        velocity_dir,
        velocity
    )


    complete_marker = (
        velocity_dir
        /
        "velocity_complete.pkl"
    )


    if complete_marker.exists():

        print(
            f"Velocity {velocity:.1f} mm/s "
            f"is already complete."
        )

        return


    checkpoint_path = (
        velocity_dir
        /
        "checkpoint_latest.pkl"
    )


    (
        key,
        model,
        params,
        optim,
        state,
        apply_fn
    ) = initialize_model()


    time_history = []

    runtime_history = []

    centerline_history = []

    window_loss_history = []

    all_loss_histories = []


    start_window = 0

    resume_epoch = 0

    train_data = None

    loss_history = []

    completed_runtime = 0.0

    window_runtime_base = 0.0


    if checkpoint_path.exists():

        checkpoint = load_checkpoint(
            checkpoint_path
        )

        if checkpoint_matches_configuration(
            checkpoint
        ):

            params = checkpoint["params"]

            state = checkpoint["state"]

            key = checkpoint["key"]

            start_window = int(
                checkpoint["window_index"]
            )

            resume_epoch = int(
                checkpoint["epoch"]
            )

            completed_runtime = float(
                checkpoint.get(
                    "completed_runtime_s",
                    0.0
                )
            )

            time_history = checkpoint.get(
                "time_history",
                []
            )

            runtime_history = checkpoint.get(
                "runtime_history",
                []
            )

            centerline_history = checkpoint.get(
                "centerline_history",
                []
            )

            window_loss_history = checkpoint.get(
                "window_loss_history",
                []
            )

            all_loss_histories = checkpoint.get(
                "all_loss_histories",
                []
            )

            dataset_path = checkpoint.get(
                "dataset_path"
            )

            if (
                dataset_path
                and
                Path(dataset_path).exists()
            ):

                train_data = load_dataset(
                    dataset_path
                )

            loss_history = checkpoint.get(
                "loss_history",
                []
            )

            completed_history_runtime = sum(
                float(row[2])
                for row in runtime_history
            )

            window_runtime_base = max(
                completed_runtime
                - completed_history_runtime,
                0.0
            )

            print(
                f"Resuming v={velocity:.1f} mm/s "
                f"from window "
                f"{start_window + 1}/{runs}, "
                f"epoch "
                f"{resume_epoch:,}."
            )

        else:

            backup_path = (
                velocity_dir
                /
                "checkpoint_incompatible_previous.pkl"
            )

            if not backup_path.exists():

                os.replace(
                    checkpoint_path,
                    backup_path
                )

            print(
                f"Checkpoint v={velocity:.1f} mm/s "
                f"does not match the current configuration."
            )

            print(
                "Starting from the beginning."
            )

    else:

        window_runtime_base = 0.0

        print(
            f"Starting v={velocity:.1f} mm/s "
            f"from the beginning."
        )


    for r in range(
        start_window,
        runs
    ):

        t_start_w = (
            tmin
            +
            r * delt
        )


        t_end_w = (
            tmin
            +
            (r + 1) * delt
        )


        dataset_path = (
            velocity_dir
            /
            f"dataset_"
            f"v{velocity:.0f}mm_s_"
            f"window{r + 1:02d}.npz"
        )


        if (
            r == start_window
            and
            resume_epoch > 0
        ):

            if train_data is None:

                if not dataset_path.exists():

                    raise FileNotFoundError(
                        f"Training dataset not found: "
                        f"{dataset_path}"
                    )

                train_data = load_dataset(
                    dataset_path
                )


        else:

            if r == 0:

                key, subkey = jax.random.split(
                    key
                )


                train_data = (
                    pinn_training_data_generator(
                        NC,
                        NI,
                        NB,
                        subkey,
                        velocity,
                        t_start_w,
                        t_end_w
                    )
                )


            else:

                # Generate a new collocation set for the current window.

                key, subkey = jax.random.split(
                    key
                )

                train_data = shift_training_data(
                    train_data,
                    params,
                    apply_fn,
                    velocity,
                    delt,
                    subkey,
                    t_start_w,
                    t_end_w
                )


            save_dataset(
                dataset_path,
                train_data
            )


            resume_epoch = 0

            loss_history = []

            window_runtime_base = 0.0


        window_session_start = time.time()

        if resume_epoch >= EPOCHS:

            print(
                f"Window {r + 1}/{runs} "
                f"already reached "
                f"{EPOCHS:,} epochs."
            )


        else:

            loss_fn = pinn_loss(
                apply_fn,
                *train_data
            )


            @jax.jit
            def train_one_step(
                params,
                state
            ):

                loss, gradient = (
                    value_and_grad(
                        loss_fn
                    )(params)
                )


                params, state = (
                    update_model(
                        optim,
                        gradient,
                        params,
                        state
                    )
                )


                return (
                    loss,
                    params,
                    state
                )


            if resume_epoch == 0:

                print(
                    f"Training window "
                    f"{r + 1}/{runs}: "
                    f"t=[{t_start_w:.1f}, "
                    f"{t_end_w:.1f}] s"
                )


            else:

                print(
                    f"Resuming window "
                    f"{r + 1}/{runs} "
                    f"at epoch "
                    f"{resume_epoch + 1:,}."
                )


            for epoch in trange(
                resume_epoch + 1,
                EPOCHS + 1,
                desc=(
                    f"v={velocity:.1f} mm/s, "
                    f"window {r + 1}/{runs}"
                )
            ):

                loss, params, state = (
                    train_one_step(
                        params,
                        state
                    )
                )


                loss_value = float(
                    jax.device_get(
                        loss
                    )
                )


                loss_history.append(
                    loss_value
                )


                if (
                    epoch % CHECKPOINT_EVERY == 0
                    or
                    epoch == EPOCHS
                ):

                    loss_array = np.asarray(
                        loss_history,
                        dtype=float
                    )


                    loss_array_logged = np.log(
                        np.maximum(
                            loss_array,
                            1e-30
                        )
                    )


                    elapsed_window = (
                        time.time()
                        -
                        window_session_start
                    )


                    total_runtime = (
                        completed_runtime
                        +
                        elapsed_window
                    )


                    checkpoint_payload = {

                        "velocity":
                            velocity,

                        "window_index":
                            r,

                        "epoch":
                            epoch,

                        "params":
                            params,

                        "state":
                            state,

                        "key":
                            key,

                        "dataset_path":
                            str(dataset_path),

                        "loss_history":
                            loss_history,

                        "time_history":
                            time_history,

                        "runtime_history":
                            runtime_history,

                        "centerline_history":
                            centerline_history,

                        "window_loss_history":
                            window_loss_history,

                        "all_loss_histories":
                            all_loss_histories,

                        "completed_runtime_s":
                            total_runtime,

                        "timestamp":
                            time.time(),

                        "configuration":
                            configuration_signature()
                    }


                    save_checkpoint(
                        checkpoint_path,
                        checkpoint_payload
                    )


                    save_csv(
                        velocity_dir
                        /
                        f"loss_progress_"
                        f"v{velocity:.0f}mm_s_"
                        f"window{r + 1:02d}.csv",

                        np.column_stack(
                            [
                                np.arange(
                                    1,
                                    len(
                                        loss_array_logged
                                    ) + 1
                                ),
                                loss_array_logged
                            ]
                        ),

                        "epoch,ln_loss"
                    )


        if not loss_history:

            raise RuntimeError(
                f"No loss history is available "
                f"for window {r + 1}."
            )


        loss_array = np.asarray(
            loss_history,
            dtype=float
        )


        loss_array = np.log(
            np.maximum(
                loss_array,
                1e-30
            )
        )


        t_eval = t_end_w


        (
            temperature_field,
            x_grid,
            y_grid,
            xm,
            ym
        ) = evaluate_window(
            params,
            apply_fn,
            t_eval
        )


        (
            t_cl,
            x_cl,
            y_cl
        ) = pinn_centerline_data_generator(
            NC_TEST,
            t_eval,
            centerline_y
        )


        centerline = np.asarray(
            jax.device_get(
                apply_fn(
                    params,
                    x_cl,
                    y_cl,
                    t_cl
                )
            )
        ).reshape(
            -1
        )


        if resume_epoch >= EPOCHS:

            runtime_window = (
                window_runtime_base
            )

        else:

            runtime_window = (
                window_runtime_base
                +
                time.time()
                -
                window_session_start
            )


        result = {

            "window":
                r + 1,

            "time_s":
                t_eval,

            "Tmax_C":
                float(
                    np.max(
                        temperature_field
                    )
                ),

            "Tmin_C":
                float(
                    np.min(
                        temperature_field
                    )
                ),

            "Tmean_C":
                float(
                    np.mean(
                        temperature_field
                    )
                ),

            "runtime_s":
                float(
                    runtime_window
                ),

            "runtime_per_epoch_s":
                float(
                    runtime_window
                    /
                    max(
                        EPOCHS,
                        1
                    )
                ),

            "centerline":
                centerline
        }


        if len(
            time_history
        ) <= r:

            time_history.append(
                [
                    result[
                        "time_s"
                    ],

                    result[
                        "Tmax_C"
                    ],

                    result[
                        "Tmin_C"
                    ],

                    result[
                        "Tmean_C"
                    ]
                ]
            )


            runtime_history.append(
                [
                    result[
                        "window"
                    ],

                    result[
                        "time_s"
                    ],

                    result[
                        "runtime_s"
                    ],

                    result[
                        "runtime_per_epoch_s"
                    ]
                ]
            )


            window_loss_history.append(
                [
                    result[
                        "window"
                    ],

                    result[
                        "time_s"
                    ],

                    float(
                        loss_array[-1]
                    )
                ]
            )


            centerline_history.append(
                centerline.copy()
            )


            all_loss_histories.append(
                loss_array.copy()
            )


        else:

            time_history[r] = [
                result[
                    "time_s"
                ],

                result[
                    "Tmax_C"
                ],

                result[
                    "Tmin_C"
                ],

                result[
                    "Tmean_C"
                ]
            ]


            runtime_history[r] = [
                result[
                    "window"
                ],

                result[
                    "time_s"
                ],

                result[
                    "runtime_s"
                ],

                result[
                    "runtime_per_epoch_s"
                ]
            ]


            window_loss_history[r] = [
                result[
                    "window"
                ],

                result[
                    "time_s"
                ],

                float(
                    loss_array[-1]
                )
            ]


            centerline_history[r] = (
                centerline.copy()
            )


            if len(all_loss_histories) <= r:

                all_loss_histories.append(
                    loss_array.copy()
                )

            else:

                all_loss_histories[r] = (
                    loss_array.copy()
                )


        completed_runtime = sum(
            float(row[2])
            for row in runtime_history
        )


        contour_output = (
            np.column_stack(
                [
                    xm.reshape(-1),
                    ym.reshape(-1),
                    temperature_field.reshape(-1)
                ]
            )
        )


        save_csv(
            velocity_dir
            /
            f"contour_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.csv",

            contour_output,

            "x_mm,y_mm,T_C"
        )


        save_csv(
            velocity_dir
            /
            f"centerline_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.csv",

            np.column_stack(
                [
                    np.linspace(
                        xmin,
                        xmax,
                        NC_TEST
                    ),

                    centerline
                ]
            ),

            "x_mm,T_C"
        )


        save_csv(
            velocity_dir
            /
            f"loss_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.csv",

            np.column_stack(
                [
                    np.arange(
                        1,
                        len(loss_array) + 1
                    ),

                    loss_array
                ]
            ),

            "epoch,ln_loss"
        )


        np.save(
            velocity_dir
            /
            f"SequentialTimeWindow_contour_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.npy",

            temperature_field
        )


        np.save(
            velocity_dir
            /
            f"centerline_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.npy",

            centerline
        )


        np.save(
            velocity_dir
            /
            f"loss_"
            f"v{velocity:.0f}mm_s_"
            f"t{t_eval:.0f}s.npy",

            loss_array
        )


        save_window_plots(
            velocity_dir,
            velocity,
            t_eval,
            temperature_field,
            x_grid,
            y_grid,
            centerline,
            loss_array
        )


        save_progress_tables(
            velocity_dir,
            velocity,
            time_history,
            runtime_history,
            window_loss_history,
            centerline_history,
            all_loss_histories
        )


        completed_checkpoint = {

            "velocity":
                velocity,

            "window_index":
                r + 1,

            "epoch":
                0,

            "params":
                params,

            "state":
                state,

            "key":
                key,

            "dataset_path":
                str(dataset_path),

            "loss_history":
                [],

            "time_history":
                time_history,

            "runtime_history":
                runtime_history,

            "centerline_history":
                centerline_history,

            "window_loss_history":
                window_loss_history,

            "all_loss_histories":
                all_loss_histories,

            "completed_runtime_s":
                completed_runtime,

            "timestamp":
                time.time(),

            "configuration":
                configuration_signature()
        }


        save_checkpoint(
            checkpoint_path,
            completed_checkpoint
        )


        resume_epoch = 0

        loss_history = []

        window_runtime_base = 0.0


        print(
            f"Window {r + 1}/{runs} completed: "
            f"Tmax="
            f"{result['Tmax_C']:.3f} °C, "
            f"Tmean="
            f"{result['Tmean_C']:.3f} °C"
        )


    save_final_plots(
        velocity_dir,
        velocity,
        time_history,
        centerline_history,
        all_loss_histories
    )


    final_data = {

        "velocity":
            velocity,

        "time_history":
            np.asarray(
                time_history,
                dtype=float
            ),

        "runtime_history":
            np.asarray(
                runtime_history,
                dtype=float
            ),

        "centerline_history":
            np.asarray(
                centerline_history,
                dtype=float
            ),

        "window_loss_history":
            np.asarray(
                window_loss_history,
                dtype=float
            ),

        "all_loss_histories":
            [
                np.asarray(
                    history,
                    dtype=float
                )
                for history in all_loss_histories
            ],

        "total_runtime_s":
            float(
                sum(
                    float(row[2])
                    for row in runtime_history
                )
            )
    }


    with open(
        velocity_dir
        /
        "final_results.pkl",
        "wb"
    ) as f:

        pickle.dump(
            final_data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    save_csv(
        velocity_dir
        /
        "final_time_history.csv",

        final_data[
            "time_history"
        ],

        "time_s,Tmax_C,Tmin_C,Tmean_C"
    )


    save_csv(
        velocity_dir
        /
        "final_runtime_history.csv",

        final_data[
            "runtime_history"
        ],

        "window,time_s,runtime_s,runtime_per_epoch_s"
    )


    save_csv(
        velocity_dir
        /
        "final_window_loss.csv",

        final_data[
            "window_loss_history"
        ],

        "window,time_s,final_ln_loss"
    )


    complete_payload = {

        "velocity":
            velocity,

        "completed":
            True,

        "total_runtime_s":
            final_data[
                "total_runtime_s"
            ],

        "timestamp":
            time.time()
    }


    save_checkpoint(
        complete_marker,
        complete_payload
    )


    print(
        f"Velocity {velocity:.1f} mm/s "
        f"completed. "
        f"Total runtime="
        f"{final_data['total_runtime_s'] / 60.0:.2f} min."
    )

# Run all cases
for velocity in velocities:

    train_velocity(
        velocity
    )


print(
    "All Sequential Time-Window PINN cases are complete."
)