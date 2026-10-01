#Environment and Configuration
import os
import time
import pickle
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from tqdm import tqdm
from flax import linen as nn
from jax import jvp, vjp, value_and_grad


try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    BASE_DIR = "/content/drive/MyDrive/PINN_Baseline_Standard"
except ImportError:
    BASE_DIR = "./PINN_Baseline_Standard"

os.makedirs(BASE_DIR, exist_ok=True)

print(f"Output directory : {BASE_DIR}")

#Problem Setup
xmin, xmax = 0.0, 35.0
ymin, ymax = 0.0, 17.0

tmin, tmax = 0.0, 10.0

VELOCITIES = [1.0, 2.0, 3.0]

rho = 7.6e-6
c = 658.0
k = 25e-3

alpha = k / (rho * c)

#Moving Heat Source
r0 = 1.0
q0 = 5.0
muy = 8.5

#Training Configuration
NC_PER_WINDOW = 128**2
NB_PER_WINDOW = 96**2
NI_FIXED = 96**2

N_WINDOWS_SEQUENTIAL = 10

NC = NC_PER_WINDOW * N_WINDOWS_SEQUENTIAL
NB = NB_PER_WINDOW * N_WINDOWS_SEQUENTIAL
NI = NI_FIXED

#Network and Training Budget
SEED = 444
LR = 1e-3
N_LAYERS = 9
FEATURES = 128

EPOCHS_PER_WINDOW = 8000
EPOCHS_TOTAL = EPOCHS_PER_WINDOW * N_WINDOWS_SEQUENTIAL

CHECKPOINT_EVERY = EPOCHS_PER_WINDOW
SAVE_EVERY = 500

N_CHECKPOINTS = EPOCHS_TOTAL // CHECKPOINT_EVERY

#Evaluation Configuration
NC_TEST = 500
CENTERLINE_Y = 8.5

comparison_times = np.arange(
    tmin + 1.0,
    tmax + 1.0,
    1.0
)

centerline_x = np.linspace(
    xmin,
    xmax,
    NC_TEST
)

#Runtime Information
print("=" * 68)
print("Standard PINN with automatic checkpointing")
print("=" * 68)

print(f"JAX version         : {jax.__version__}")
print(f"JAX backend         : {jax.default_backend()}")
print(f"JAX devices         : {jax.devices()}")

print(
    f"Domain              : "
    f"{xmin:.1f}-{xmax:.1f} x {ymin:.1f}-{ymax:.1f} mm"
)

print(f"Time range          : {tmin:.1f}-{tmax:.1f} s")
print(f"Velocity            : {VELOCITIES} mm/s")
print(f"Collocation points  : {NC:,}")
print(f"Boundary points     : {NB:,}")
print(f"Initial points      : {NI:,}")

print(
    f"Network architecture: "
    f"{N_LAYERS} layers x {FEATURES} neurons"
)

print(f"Total epochs        : {EPOCHS_TOTAL:,}")

print(
    f"Tmax/Tmean evaluation: "
    f"every {CHECKPOINT_EVERY} epochs"
)

print(
    f"Drive checkpoint    : "
    f"every {SAVE_EVERY} epochs"
)

print("Windowing           : not used")
print("Transfer learning   : not used")
print("=" * 68)

#Neural Network and Derivatives
class PINN(nn.Module):

    features: Sequence[int]

    @nn.compact
    def __call__(self, x, y, t):

        inputs = jnp.concatenate(
            [x, y, t],
            axis=1
        )

        init = nn.initializers.glorot_normal()

        for features in self.features[:-1]:

            inputs = nn.Dense(
                features,
                kernel_init=init
            )(inputs)

            inputs = nn.activation.tanh(inputs)

        inputs = nn.Dense(
            self.features[-1],
            kernel_init=init
        )(inputs)

        return inputs


def hvp_fwdfwd(
    f,
    primals,
    tangents,
    return_primals=False
):

    g = lambda p: jvp(
        f,
        (p,),
        tangents
    )[1]

    primals_out, tangents_out = jvp(
        g,
        primals,
        tangents
    )

    if return_primals:
        return primals_out, tangents_out

    return tangents_out

#Moving Heat Source
def heat_source(pos, t, velocity):

    mux = 34.0 * jnp.ones(
        (t.shape[0], 1)
    ) - velocity * t

    muy_array = 8.5 * jnp.ones(
        (t.shape[0], 1)
    )

    x_source = pos[:, 0].reshape(-1, 1)
    y_source = pos[:, 1].reshape(-1, 1)

    r2 = (
        jnp.square(x_source - mux)
        + jnp.square(y_source - muy_array)
    )

    return q0 * jnp.exp(
        -r2 / r0**2
    )

#Physics-Informed Loss
def pinn_loss(
    apply_fn,
    train_data,
    velocity
):

    (
        tc, xc, yc, source_term,
        ti, xi, yi, ui,
        tb, xb, yb, ub
    ) = train_data

    dirichlet_mask = xb == xmin

    dirichlet_indices = jnp.where(
        dirichlet_mask
    )[0]

    neumann_indices = jnp.where(
        ~dirichlet_mask
    )[0]

    xdb = xb[dirichlet_indices]
    ydb = yb[dirichlet_indices]
    tdb = tb[dirichlet_indices]
    udb = ub[dirichlet_indices]

    xnb = xb[neumann_indices]
    ynb = yb[neumann_indices]
    tnb = tb[neumann_indices]
    unb = ub[neumann_indices]

    def residual_loss(params):

        u = apply_fn(
            params,
            xc,
            yc,
            tc
        )

        ones = jnp.ones(
            [u.shape[0], 1]
        )

        ut_f = vjp(
            lambda t_var:
            apply_fn(
                params,
                xc,
                yc,
                t_var
            ),
            tc
        )[1]

        ut = ut_f(ones)[0]

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
            - uxx
            - uyy
            - source_term / k
        )

        return jnp.mean(
            jnp.square(residual)
        )

    def initial_loss(params):

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

    def neumann_loss(params):

        u = apply_fn(
            params,
            xnb,
            ynb,
            tnb
        )

        ones = jnp.ones(
            [u.shape[0], 1]
        )

        ux_f = vjp(
            lambda x_var:
            apply_fn(
                params,
                x_var,
                ynb,
                tnb
            ),
            xnb
        )[1]

        ux = ux_f(ones)[0]

        uy_f = vjp(
            lambda y_var:
            apply_fn(
                params,
                xnb,
                y_var,
                tnb
            ),
            ynb
        )[1]

        uy = uy_f(ones)[0]

        residual = jnp.where(
            xnb == xmax,
            ux - unb,
            jnp.where(
                (ynb == ymin) |
                (ynb == ymax),
                uy - unb,
                unb
            )
        )

        return jnp.mean(
            jnp.square(residual)
        )

    def dirichlet_loss(params):

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

    def total_loss(params):

        return (
            1e3 * residual_loss(params)
            + 250.0 * initial_loss(params)
            + 250.0 * neumann_loss(params)
            + 250.0 * dirichlet_loss(params)
        )

    return total_loss

#Optimizer Update and Training Data

@partial(
    jax.jit,
    static_argnums=(0,)
)
def update_model(
    optimizer,
    gradient,
    params,
    state
):

    updates, state = optimizer.update(
        gradient,
        state
    )

    params = optax.apply_updates(
        params,
        updates
    )

    return params, state


def generate_training_data(
    nc,
    ni,
    nb,
    key,
    velocity
):

    keys = jax.random.split(
        key,
        13
    )

    tc = jax.random.uniform(
        keys[0],
        (nc, 1),
        minval=tmin,
        maxval=tmax
    )

    xc = jax.random.uniform(
        keys[1],
        (nc, 1),
        minval=xmin,
        maxval=xmax
    )

    yc = jax.random.uniform(
        keys[2],
        (nc, 1),
        minval=ymin,
        maxval=ymax
    )

    source_term = heat_source(
        jnp.concatenate(
            [xc, yc],
            axis=1
        ),
        tc,
        velocity
    )

    ti = jnp.zeros(
        (ni, 1)
    )

    xi = jax.random.uniform(
        keys[3],
        (ni, 1),
        minval=xmin,
        maxval=xmax
    )

    yi = jax.random.uniform(
        keys[4],
        (ni, 1),
        minval=ymin,
        maxval=ymax
    )

    ui = 20.0 * jnp.ones(
        (ni, 1)
    )

    tb = [
        jax.random.uniform(
            keys[5],
            (nb, 1),
            minval=tmin,
            maxval=tmax
        ),
        jax.random.uniform(
            keys[6],
            (nb, 1),
            minval=tmin,
            maxval=tmax
        ),
        jax.random.uniform(
            keys[7],
            (nb, 1),
            minval=tmin,
            maxval=tmax
        ),
        jax.random.uniform(
            keys[8],
            (nb, 1),
            minval=tmin,
            maxval=tmax
        )
    ]

    xb = [
        jnp.full(
            (nb, 1),
            xmin
        ),
        jnp.full(
            (nb, 1),
            xmax
        ),
        jax.random.uniform(
            keys[9],
            (nb, 1),
            minval=xmin,
            maxval=xmax
        ),
        jax.random.uniform(
            keys[10],
            (nb, 1),
            minval=xmin,
            maxval=xmax
        )
    ]

    yb = [
        jax.random.uniform(
            keys[11],
            (nb, 1),
            minval=ymin,
            maxval=ymax
        ),
        jax.random.uniform(
            keys[12],
            (nb, 1),
            minval=ymin,
            maxval=ymax
        ),
        jnp.full(
            (nb, 1),
            ymin
        ),
        jnp.full(
            (nb, 1),
            ymax
        )
    ]

    ub = [
        20.0 * jnp.ones(
            (nb, 1)
        ),
        -0.001 * jnp.ones(
            (nb, 1)
        ),
        0.001 * jnp.ones(
            (nb, 1)
        ),
        -0.001 * jnp.ones(
            (nb, 1)
        )
    ]

    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)

    return (
        tc, xc, yc, source_term,
        ti, xi, yi, ui,
        tb, xb, yb, ub
    )

#Evaluation and Checkpoint Utilities

def test_data(
    n,
    t_value
):

    x = jnp.linspace(
        xmin,
        xmax,
        n
    )

    y = jnp.linspace(
        ymin,
        ymax,
        n
    )

    t = jnp.full(
        (n * n, 1),
        t_value
    )

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij"
    )

    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    return (
        t,
        X_flat,
        Y_flat,
        X,
        Y
    )


def centerline_data(
    n,
    t_value
):

    x = jnp.linspace(
        xmin,
        xmax,
        n
    ).reshape(-1, 1)

    y = CENTERLINE_Y * jnp.ones(
        (n, 1)
    )

    t = t_value * jnp.ones(
        (n, 1)
    )

    return t, x, y


def checkpoint_path(velocity):

    return os.path.join(
        BASE_DIR,
        f"checkpoint_v{velocity:.0f}mm_s.pkl"
    )


def final_output_path(velocity):

    return os.path.join(
        BASE_DIR,
        f"BaselineStandard_time_history_v{velocity:.0f}mm_s.csv"
    )


def save_checkpoint(
    velocity,
    params,
    state,
    epoch,
    loss_history,
    checkpoint_history,
    runtime_history,
    cumulative_runtime
):

    payload = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(state),
        "epoch": epoch,
        "loss_history": loss_history,
        "checkpoint_history": checkpoint_history,
        "runtime_history": runtime_history,
        "cumulative_runtime": cumulative_runtime,
    }

    final_path = checkpoint_path(
        velocity
    )

    tmp_path = final_path + ".tmp"

    with open(
        tmp_path,
        "wb"
    ) as f:

        pickle.dump(
            payload,
            f
        )

    os.replace(
        tmp_path,
        final_path
    )


def load_checkpoint(velocity):

    path = checkpoint_path(
        velocity
    )

    if not os.path.exists(path):
        return None

    with open(
        path,
        "rb"
    ) as f:

        payload = pickle.load(f)

    payload["params"] = jax.tree_util.tree_map(
        jnp.asarray,
        payload["params"]
    )

    payload["opt_state"] = jax.tree_util.tree_map(
        jnp.asarray,
        payload["opt_state"]
    )

    return payload

#Visualization Settings
plt.rcParams.update({

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
})

#Standard PINN Training
feat_sizes = tuple(
    [FEATURES] * (N_LAYERS - 1)
    + [1]
)

all_summary = []

for velocity_index, velocity in enumerate(
    VELOCITIES
):

    print()
    print("=" * 68)

    print(
        f"Heat-source velocity : "
        f"v = {velocity:.1f} mm/s"
    )

    print("=" * 68)

    if os.path.exists(
        final_output_path(velocity)
    ):

        print(
            "Results for this velocity already exist "
            "from a previous session."
        )

        print(
            "Skipping retraining."
        )

        existing_time = np.loadtxt(
            final_output_path(velocity),
            delimiter=",",
            skiprows=1
        )

        runtime_file = os.path.join(
            BASE_DIR,
            f"BaselineStandard_runtime_v{velocity:.0f}mm_s.csv"
        )

        loss_file = os.path.join(
            BASE_DIR,
            f"BaselineStandard_loss_v{velocity:.0f}mm_s.csv"
        )

        runtime_existing = np.loadtxt(
            runtime_file,
            delimiter=",",
            skiprows=1
        )

        loss_existing = np.loadtxt(
            loss_file,
            delimiter=",",
            skiprows=1
        )

        all_summary.append([

            velocity,

            existing_time[-1, 1],

            existing_time[-1, 2],

            existing_time[-1, 3],

            runtime_existing[-1, 3],

            loss_existing[-1, 1],

        ])

        continue

    key = jax.random.PRNGKey(
        SEED + velocity_index
    )

    key, init_key, data_key = jax.random.split(
        key,
        3
    )

    model = PINN(
        features=feat_sizes
    )

    optimizer = optax.adam(
        LR
    )

    apply_fn = jax.jit(
        model.apply
    )

    train_data = generate_training_data(
        NC,
        NI,
        NB,
        data_key,
        velocity
    )

    loss_fn = pinn_loss(
        apply_fn,
        train_data,
        velocity
    )

    @jax.jit
    def train_step(
        params,
        state
    ):

        loss, gradient = value_and_grad(
            loss_fn
        )(params)

        params, state = update_model(
            optimizer,
            gradient,
            params,
            state
        )

        return loss, params, state

    resume_state = load_checkpoint(
        velocity
    )

    if resume_state is not None:

        params = resume_state["params"]

        state = resume_state["opt_state"]

        start_epoch = (
            resume_state["epoch"] + 1
        )

        loss_history = (
            resume_state["loss_history"]
        )

        checkpoint_history = (
            resume_state["checkpoint_history"]
        )

        runtime_history = (
            resume_state["runtime_history"]
        )

        cumulative_runtime_offset = (
            resume_state["cumulative_runtime"]
        )

        print(
            f"Checkpoint found. "
            f"Resuming from epoch {start_epoch}"
        )

        print(
            f"Stored runtime from previous session: "
            f"{cumulative_runtime_offset / 60:.2f} min"
        )

    else:

        params = model.init(

            init_key,

            jnp.ones(
                (1, 1)
            ),

            jnp.ones(
                (1, 1)
            ),

            jnp.ones(
                (1, 1)
            )
        )

        state = optimizer.init(
            params
        )

        start_epoch = 1

        loss_history = []

        checkpoint_history = []

        runtime_history = []

        cumulative_runtime_offset = 0.0

        print(
            "No checkpoint found. "
            "Training starts from scratch."
        )

    if start_epoch > EPOCHS_TOTAL:

        print(
            "Training for this velocity is already complete."
        )

        print(
            "Proceeding directly to final evaluation."
        )

    else:

        print(
            "Compiling training step..."
        )

        compile_start = time.time()

        _warmup_loss, _warmup_params, _warmup_state = (
            train_step(
                params,
                state
            )
        )

        _warmup_loss.block_until_ready()

        compile_time = (
            time.time() - compile_start
        )

        print(
            f"Compilation time : "
            f"{compile_time:.2f} s"
        )

        training_session_start = time.time()

        progress_bar = tqdm(

            range(
                start_epoch,
                EPOCHS_TOTAL + 1
            ),

            initial=start_epoch - 1,

            total=EPOCHS_TOTAL,

            desc=f"v={velocity:.1f} mm/s"
        )

        try:

            for epoch in progress_bar:

                loss, params, state = train_step(
                    params,
                    state
                )

                loss.block_until_ready()

                loss_history.append(
                    float(loss)
                )

                if epoch % CHECKPOINT_EVERY == 0:

                    (
                        t_eval,
                        x_eval,
                        y_eval,
                        X_eval,
                        Y_eval
                    ) = test_data(
                        NC_TEST,
                        tmax
                    )

                    prediction = apply_fn(
                        params,
                        x_eval,
                        y_eval,
                        t_eval
                    )

                    prediction.block_until_ready()

                    field = np.asarray(
                        prediction
                    ).reshape(
                        NC_TEST,
                        NC_TEST
                    )

                    Tmax_ckpt = float(
                        np.max(field)
                    )

                    Tmean_ckpt = float(
                        np.mean(field)
                    )

                    elapsed_session = (
                        time.time()
                        - training_session_start
                    )

                    cumulative_runtime = (
                        cumulative_runtime_offset
                        + elapsed_session
                    )

                    window_index = (
                        epoch
                        //
                        CHECKPOINT_EVERY
                    )

                    checkpoint_history.append([

                        window_index,

                        epoch,

                        Tmax_ckpt,

                        Tmean_ckpt

                    ])

                    runtime_history.append([

                        window_index,

                        epoch,

                        elapsed_session,

                        cumulative_runtime

                    ])

                    progress_bar.write(

                        f"  Checkpoint "
                        f"{window_index}/"
                        f"{N_CHECKPOINTS} completed | "

                        f"Tmax = "
                        f"{Tmax_ckpt:.3f} C | "

                        f"Tmean = "
                        f"{Tmean_ckpt:.3f} C"

                    )

                if epoch % SAVE_EVERY == 0:

                    elapsed_session = (
                        time.time()
                        - training_session_start
                    )

                    cumulative_runtime = (
                        cumulative_runtime_offset
                        + elapsed_session
                    )

                    save_checkpoint(

                        velocity,

                        params,

                        state,

                        epoch,

                        loss_history,

                        checkpoint_history,

                        runtime_history,

                        cumulative_runtime

                    )

        except KeyboardInterrupt:

            elapsed_session = (
                time.time()
                - training_session_start
            )

            cumulative_runtime = (
                cumulative_runtime_offset
                + elapsed_session
            )

            save_checkpoint(

                velocity,

                params,

                state,

                epoch - 1,

                loss_history,

                checkpoint_history,

                runtime_history,

                cumulative_runtime

            )

            print(
                "Training interrupted manually. "
                "The latest checkpoint has been saved."
            )

            raise

        elapsed_session = (
            time.time()
            - training_session_start
        )

        cumulative_runtime_offset = (
            cumulative_runtime_offset
            + elapsed_session
        )

        save_checkpoint(

            velocity,

            params,

            state,

            EPOCHS_TOTAL,

            loss_history,

            checkpoint_history,

            runtime_history,

            cumulative_runtime_offset

        )

#Post-Training Evaluation and Results

total_runtime = (
    cumulative_runtime_offset
)

loss_array = np.asarray(
    loss_history,
    dtype=float
)

log_loss = np.log(
    np.maximum(
        loss_array,
        1e-30
    )
)

checkpoint_history_arr = np.asarray(
    checkpoint_history,
    dtype=float
)

runtime_history_arr = np.asarray(
    runtime_history,
    dtype=float
)

time_history = []
centerline_history = []
contour_history = []

last_X = None
last_Y = None

for ts in comparison_times:

    (
        t_eval,
        x_eval,
        y_eval,
        X_eval,
        Y_eval
    ) = test_data(
        NC_TEST,
        ts
    )

    prediction = apply_fn(
        params,
        x_eval,
        y_eval,
        t_eval
    )

    prediction.block_until_ready()

    field = np.asarray(
        prediction
    ).reshape(
        NC_TEST,
        NC_TEST
    )

    contour_history.append(
        field.copy()
    )

    last_X = np.asarray(
        X_eval
    )

    last_Y = np.asarray(
        Y_eval
    )

    Tmax = float(
        np.max(field)
    )

    Tmin = float(
        np.min(field)
    )

    Tmean = float(
        np.mean(field)
    )

    time_history.append([
        ts,
        Tmax,
        Tmin,
        Tmean
    ])

    contour_output = np.column_stack((
        last_X.reshape(-1),
        last_Y.reshape(-1),
        field.reshape(-1)
    ))

    np.savetxt(
        os.path.join(
            BASE_DIR,
            f"BaselineStandard_contour_"
            f"v{velocity:.0f}mm_s_t{ts:.0f}s.csv"
        ),
        contour_output,
        delimiter=",",
        header="x_mm,y_mm,T_C",
        comments=""
    )

    t_cl, x_cl, y_cl = centerline_data(
        NC_TEST,
        ts
    )

    prediction_cl = apply_fn(
        params,
        x_cl,
        y_cl,
        t_cl
    )

    prediction_cl.block_until_ready()

    x_cl_np = np.asarray(
        x_cl
    ).reshape(-1)

    T_cl_np = np.asarray(
        prediction_cl
    ).reshape(-1)

    centerline_history.append(
        T_cl_np.copy()
    )

    centerline_output = np.column_stack((
        x_cl_np,
        T_cl_np
    ))

    np.savetxt(
        os.path.join(
            BASE_DIR,
            f"BaselineStandard_centerline_"
            f"v{velocity:.0f}mm_s_t{ts:.0f}s.csv"
        ),
        centerline_output,
        delimiter=",",
        header="x_mm,T_C",
        comments=""
    )

time_history = np.asarray(
    time_history,
    dtype=float
)

centerline_history = np.asarray(
    centerline_history,
    dtype=float
)

contour_history = np.asarray(
    contour_history,
    dtype=float
)

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_time_history_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    time_history,
    delimiter=",",
    header="time_s,Tmax_C,Tmin_C,Tmean_C",
    comments=""
)

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_checkpoint_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    checkpoint_history_arr,
    delimiter=",",
    header=(
        "checkpoint,"
        "cumulative_epoch,"
        "Tmax_at_t10s_C,"
        "Tmean_at_t10s_C"
    ),
    comments=""
)

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_runtime_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    runtime_history_arr,
    delimiter=",",
    header=(
        "checkpoint,"
        "cumulative_epoch,"
        "runtime_checkpoint_s,"
        "cumulative_runtime_s"
    ),
    comments=""
)

loss_output = np.column_stack((
    np.arange(
        1,
        len(log_loss) + 1
    ),
    log_loss
))

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_loss_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    loss_output,
    delimiter=",",
    header="epoch,ln_loss",
    comments=""
)

centerline_output_all = np.column_stack((
    centerline_x,
    centerline_history.T
))

centerline_header = (
    "x_mm,"
    +
    ",".join(
        [
            f"T_{int(ts)}s_C"
            for ts in comparison_times
        ]
    )
)

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_centerline_all_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    centerline_output_all,
    delimiter=",",
    header=centerline_header,
    comments=""
)

config_output = np.array([[
    velocity,
    NC,
    NB,
    NI,
    EPOCHS_TOTAL,
    xmin,
    xmax,
    ymin,
    ymax,
    CENTERLINE_Y,
    muy,
    r0,
    q0,
    rho,
    c,
    k
]])

np.savetxt(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_config_"
        f"v{velocity:.0f}mm_s.csv"
    ),
    config_output,
    delimiter=",",
    header=(
        "velocity_mm_s,"
        "NC,"
        "NB,"
        "NI,"
        "EPOCHS,"
        "xmin_mm,"
        "xmax_mm,"
        "ymin_mm,"
        "ymax_mm,"
        "centerline_y_mm,"
        "muy_mm,"
        "r0_mm,"
        "q0,"
        "rho,"
        "c,"
        "k"
    ),
    comments=""
)

#Final Figures and Summary
fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

ax.plot(
    np.arange(
        1,
        len(log_loss) + 1
    ),
    log_loss,
    linewidth=1.2,
    color="#1f4e79"
)

ax.set_xlabel(
    "Cumulative Epoch"
)

ax.set_ylabel(
    r"$\ln(\mathcal{L})$"
)

ax.set_title(
    f"Training Convergence "
    f"($v={velocity:.1f}$ mm/s)"
)

fig.savefig(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_loss_"
        f"v{velocity:.0f}mm_s.png"
    ),
    facecolor="white"
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

ax.plot(
    checkpoint_history_arr[:, 1],
    checkpoint_history_arr[:, 2],
    marker="o",
    linewidth=1.5,
    color="#c0392b",
    label=r"$T_{\mathrm{max}}$"
)

ax.plot(
    checkpoint_history_arr[:, 1],
    checkpoint_history_arr[:, 3],
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
    f"Temperature Convergence "
    f"($v={velocity:.1f}$ mm/s)"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_convergence_"
        f"v{velocity:.0f}mm_s.png"
    ),
    facecolor="white"
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(7, 4)
)

cf = ax.contourf(
    last_X,
    last_Y,
    contour_history[-1],
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
    f"$t=10$ s "
    f"($v={velocity:.1f}$ mm/s)"
)

fig.savefig(
    os.path.join(
        BASE_DIR,
        f"BaselineStandard_contour_final_"
        f"v{velocity:.0f}mm_s.png"
    ),
    facecolor="white"
)

plt.close(fig)


final_Tmax = time_history[-1, 1]
final_Tmin = time_history[-1, 2]
final_Tmean = time_history[-1, 3]

all_summary.append([
    velocity,
    final_Tmax,
    final_Tmin,
    final_Tmean,
    total_runtime,
    log_loss[-1]
])

print(
    f"Tmax = "
    f"{final_Tmax:.3f} C | "

    f"Tmean = "
    f"{final_Tmean:.3f} C | "

    f"runtime = "
    f"{total_runtime / 60:.2f} min"
)

#Combined Summary
all_summary = np.asarray(
    all_summary,
    dtype=float
)

np.savetxt(
    os.path.join(
        BASE_DIR,
        "BaselineStandard_summary.csv"
    ),
    all_summary,
    delimiter=",",
    header=(
        "velocity_mm_s,"
        "Tmax_C,"
        "Tmin_C,"
        "Tmean_C,"
        "runtime_s,"
        "final_ln_loss"
    ),
    comments=""
)

print()
print("=" * 68)

print(
    "Standard PINN completed "
    "for all velocities"
)

print("=" * 68)

print(
    f"{'Velocity':>10} "
    f"{'Tmax (C)':>14} "
    f"{'Tmean (C)':>14} "
    f"{'Runtime (min)':>18}"
)

for row in all_summary:

    print(

        f"{row[0]:>10.1f} "

        f"{row[1]:>14.3f} "

        f"{row[3]:>14.3f} "

        f"{row[4] / 60:>18.2f}"

    )

print("=" * 68)

print(
    f"All results saved to : "
    f"{BASE_DIR}"
)

print("=" * 68)