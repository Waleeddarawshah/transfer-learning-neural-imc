from pathlib import Path
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

PLANT_CHECKPOINT = BASE_DIR / "cstr_gru_deltaISS.pth"

X_TRAIN_PATH = BASE_DIR / "X_train.npy"
Y_TRAIN_PATH = BASE_DIR / "Y_train.npy"
X_VAL_PATH = BASE_DIR / "X_val.npy"
Y_VAL_PATH = BASE_DIR / "Y_val.npy"
X_TEST_PATH = BASE_DIR / "X_test.npy"
Y_TEST_PATH = BASE_DIR / "Y_test.npy"

NORMALIZATION_STATS_PATH = BASE_DIR / "normalization_stats.csv"
if not NORMALIZATION_STATS_PATH.exists():
    NORMALIZATION_STATS_PATH = BASE_DIR / "Normalization_Stats_Multi_EoverR.csv"

CONTROLLER_SAVE_PATH = BASE_DIR / "cstr_gru_controller_deltaISS_transfer.pth"

PLANT_INPUT_SIZE = 2
PLANT_HIDDEN_SIZE = 64
PLANT_NUM_LAYERS = 3
PLANT_OUTPUT_SIZE = 2

CONTROLLER_INPUT_SIZE = 3
CONTROLLER_HIDDEN_SIZE = 64
CONTROLLER_NUM_LAYERS = 3
CONTROLLER_OUTPUT_SIZE = 1

BATCH_SIZE = 64
EPOCHS = 800
LR = 3e-4
WEIGHT_DECAY = 1e-6
WASHOUT = 100

EARLY_STOP_PATIENCE = 100
MIN_DELTA = 1e-7

N_REF_TOTAL_PER_E = 430
N_REF_TRAIN_PER_E = 380
N_REF_VAL_PER_E = 40
N_REF_TEST_PER_E = 10

REF_SEQ_LEN = 700
MIN_HOLD = 80
MAX_HOLD = 180
TAU_REF = 16.0
DT = 1.0

N_FEASIBLE_GRID = 500
N_SETTLING_STEPS = 1200
N_AVG_TAIL = 100

USE_OBSERVED_SUPPORT_FILTER = True
OBSERVED_SUPPORT_STRIDE = 80
SUPPORT_DISTANCE_MAX = 0.20
MIN_FEASIBLE_POINTS = 40

BRANCH_SUPPORT_LOWER_PCT = 1.0
BRANCH_SUPPORT_UPPER_PCT = 99.0
CA_BRANCH_MARGIN = 0.015
T_BRANCH_MARGIN = 2.0

MAX_REF_INDEX_JUMP = 20
MIN_REF_MOVE_CA = 0.005
MIN_REF_MOVE_T = 2.0
MIN_SEQ_RANGE_CA = 0.010
MIN_SEQ_RANGE_T = 4.0
MAX_REF_RESAMPLE_TRIES = 80

CONTROL_SMOOTH_WEIGHT = 2e-3
CONTROL_MAG_WEIGHT = 1e-6
CONTROL_ISS_WEIGHT = 1e-5
OVERSHOOT_WEIGHT = 2e-3
ISS_MARGIN = 1e-3

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

def polish_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_figure(fig, filename_base):
    fig.savefig(BASE_DIR / f"{filename_base}.png", bbox_inches="tight")
    fig.savefig(BASE_DIR / f"{filename_base}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

norm_stats = pd.read_csv(NORMALIZATION_STATS_PATH)

def get_stat(var, col):
    return float(norm_stats.loc[norm_stats["Variable"] == var, col].iloc[0])

Tc_mean = get_stat("Tc", "Mean")
Tc_std = get_stat("Tc", "Std")
Ca_mean = get_stat("Ca", "Mean")
Ca_std = get_stat("Ca", "Std")
T_mean = get_stat("T", "Mean")
T_std = get_stat("T", "Std")
E_mean = get_stat("EoverR", "Mean")
E_std = get_stat("EoverR", "Std")

def denorm_tc(u_norm):
    return np.asarray(u_norm) * Tc_std + Tc_mean

def denorm_y(y_norm):
    y_norm = np.asarray(y_norm)
    y_real = np.zeros_like(y_norm, dtype=float)
    y_real[..., 0] = y_norm[..., 0] * Ca_std + Ca_mean
    y_real[..., 1] = y_norm[..., 1] * T_std + T_mean
    return y_real

def denorm_e(e_norm):
    return np.asarray(e_norm) * E_std + E_mean

class CSTR_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                self.num_layers,
                x.size(0),
                self.hidden_size,
                device=x.device
            )

        out, h_n = self.gru(x, h)
        out = self.fc(out)

        return out, h_n


class ControllerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, u_low, u_high):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

        u_low = torch.tensor(u_low, dtype=torch.float32).view(1, 1, output_size)
        u_high = torch.tensor(u_high, dtype=torch.float32).view(1, 1, output_size)

        self.register_buffer("u_mid", 0.5 * (u_low + u_high))
        self.register_buffer("u_half_range", 0.5 * (u_high - u_low))

    def forward(self, controller_input, h=None):
        if h is None:
            h = torch.zeros(
                self.num_layers,
                controller_input.size(0),
                self.hidden_size,
                device=controller_input.device
            )

        out, h_n = self.gru(controller_input, h)
        raw_u = self.fc(out)
        u = self.u_mid + self.u_half_range * torch.tanh(raw_u)

        return u, h_n

X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
Y_val = np.load(Y_VAL_PATH)

X_test = np.load(X_TEST_PATH)
Y_test = np.load(Y_TEST_PATH)

print("Loaded normalized data:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:  ", X_val.shape)
print("Y_val:  ", Y_val.shape)
print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)

assert X_train.shape[-1] == 2, "Expected X features [Tc_norm, EoverR_norm]."
assert Y_train.shape[-1] == 2, "Expected Y features [Ca_norm, T_norm]."

EOVER_R_NORM_VALUES = np.sort(np.unique(np.round(X_train[:, 0, 1], 8)))

print("\nDetected normalized activation-energy conditions:")
print(EOVER_R_NORM_VALUES)

plant_ckpt = torch.load(PLANT_CHECKPOINT, map_location=DEVICE, weights_only=False)

plant_input_size = plant_ckpt.get("input_size", PLANT_INPUT_SIZE)
plant_hidden_size = plant_ckpt.get("hidden_size", PLANT_HIDDEN_SIZE)
plant_num_layers = plant_ckpt.get("num_layers", PLANT_NUM_LAYERS)
plant_output_size = plant_ckpt.get("output_size", PLANT_OUTPUT_SIZE)

plant_model = CSTR_GRU(
    input_size=plant_input_size,
    hidden_size=plant_hidden_size,
    num_layers=plant_num_layers,
    output_size=plant_output_size
).to(DEVICE)

plant_model.load_state_dict(plant_ckpt["model_state_dict"])
plant_model.eval()

for param in plant_model.parameters():
    param.requires_grad = False

print("\nFrozen NN plant model loaded successfully.")

def make_eoverr_column(batch_size, seq_len, eoverr_norm, device):
    return torch.full(
        (batch_size, seq_len, 1),
        float(eoverr_norm),
        dtype=torch.float32,
        device=device
    )

def make_plant_input(u_seq, e_seq):
    return torch.cat([u_seq, e_seq], dim=2)

def make_controller_input(ref_seq_np, eoverr_norm):
    e_feature = np.ones(
        (ref_seq_np.shape[0], ref_seq_np.shape[1], 1),
        dtype=np.float32
    ) * float(eoverr_norm)

    return np.concatenate([ref_seq_np, e_feature], axis=2)

def first_order_reference_filter(setpoints, tau=TAU_REF, dt=DT):
    alpha = np.clip(dt / tau, 0.0, 1.0)

    y_ref = np.zeros_like(setpoints, dtype=np.float32)
    y_ref[0] = setpoints[0]

    for k in range(len(setpoints) - 1):
        y_ref[k + 1] = y_ref[k] + alpha * (setpoints[k] - y_ref[k])

    return y_ref

def inf_norm(mat):
    return torch.max(torch.sum(torch.abs(mat), dim=1))

def gru_delta_iss_penalty(gru_module, num_layers, margin=ISS_MARGIN):
    penalty = 0.0

    for layer in range(num_layers):
        w_hh = getattr(gru_module, f"weight_hh_l{layer}")
        w_ih = getattr(gru_module, f"weight_ih_l{layer}")
        b_hh = getattr(gru_module, f"bias_hh_l{layer}")
        b_ih = getattr(gru_module, f"bias_ih_l{layer}")

        bias = b_hh + b_ih

        Ur, Uz, Uh = w_hh.chunk(3, 0)
        Wr, Wz, Wh = w_ih.chunk(3, 0)
        br, bz, bh = bias.chunk(3, 0)

        sz = torch.sigmoid(inf_norm(torch.cat([Wz, Uz, bz.unsqueeze(1)], dim=1)))
        sf = torch.sigmoid(inf_norm(torch.cat([Wr, Ur, br.unsqueeze(1)], dim=1)))
        pr = torch.tanh(inf_norm(torch.cat([Wh, Uh, bh.unsqueeze(1)], dim=1)))

        nu = (
            inf_norm(Uh) * (0.25 * inf_norm(Uh) + sf)
            + 0.25 * ((1.0 + pr) / (1.0 - sz + 1e-8)) * inf_norm(Uz)
            - 1.0
        )

        penalty = penalty + torch.relu(nu + margin)

    return penalty

flat_tc = X_train[:, :, 0].reshape(-1, 1)

u_low = np.percentile(flat_tc, 1, axis=0)
u_high = np.percentile(flat_tc, 99, axis=0)

print("\nController normalized Tc bounds:")
print("u_low: ", u_low)
print("u_high:", u_high)

def simulate_nn_equilibrium_constant_u(plant, u_value, eoverr_norm):
    u_col = torch.full(
        (1, N_SETTLING_STEPS, 1),
        float(u_value),
        dtype=torch.float32,
        device=DEVICE
    )

    e_col = make_eoverr_column(
        batch_size=1,
        seq_len=N_SETTLING_STEPS,
        eoverr_norm=eoverr_norm,
        device=DEVICE
    )

    plant_input = make_plant_input(u_col, e_col)

    with torch.no_grad():
        y_seq, _ = plant(plant_input)

    y_np = y_seq.cpu().numpy()[0]
    return np.mean(y_np[-N_AVG_TAIL:], axis=0)

def longest_true_segment(mask):
    best_start = 0
    best_len = 0
    current_start = None

    for i, value in enumerate(mask):
        if value and current_start is None:
            current_start = i

        if (not value or i == len(mask) - 1) and current_start is not None:
            end = i if not value else i + 1
            length = end - current_start

            if length > best_len:
                best_start = current_start
                best_len = length

            current_start = None

    output = np.zeros_like(mask, dtype=bool)

    if best_len > 0:
        output[best_start:best_start + best_len] = True

    return output

def filter_feasible_set_to_observed_branch(feasible_y, observed_y):
    lower = np.percentile(observed_y, BRANCH_SUPPORT_LOWER_PCT, axis=0)
    upper = np.percentile(observed_y, BRANCH_SUPPORT_UPPER_PCT, axis=0)

    ca_margin_norm = CA_BRANCH_MARGIN / (Ca_std + 1e-8)
    t_margin_norm = T_BRANCH_MARGIN / (T_std + 1e-8)

    lower = lower - np.array([ca_margin_norm, t_margin_norm])
    upper = upper + np.array([ca_margin_norm, t_margin_norm])

    mask = (
        (feasible_y[:, 0] >= lower[0])
        & (feasible_y[:, 0] <= upper[0])
        & (feasible_y[:, 1] >= lower[1])
        & (feasible_y[:, 1] <= upper[1])
    )

    if USE_OBSERVED_SUPPORT_FILTER:
        observed_sample = observed_y[::OBSERVED_SUPPORT_STRIDE]

        distances = np.sqrt(
            np.sum(
                (feasible_y[:, None, :] - observed_sample[None, :, :]) ** 2,
                axis=2
            )
        )

        nearest_distance = np.min(distances, axis=1)
        mask = mask & (nearest_distance <= SUPPORT_DISTANCE_MAX)

    segment_mask = longest_true_segment(mask)

    if np.sum(segment_mask) >= MIN_FEASIBLE_POINTS:
        mask = segment_mask

    if np.sum(mask) < MIN_FEASIBLE_POINTS:
        print("  Warning: feasible support filter too strict; using percentile-only branch support.")

        mask = (
            (feasible_y[:, 0] >= lower[0])
            & (feasible_y[:, 0] <= upper[0])
            & (feasible_y[:, 1] >= lower[1])
            & (feasible_y[:, 1] <= upper[1])
        )

    return mask

def build_feasible_set_for_e(plant, eoverr_norm):
    u_grid = np.linspace(float(u_low[0]), float(u_high[0]), N_FEASIBLE_GRID)
    y_eq = []

    print(f"\nBuilding feasible reference set for EoverR_norm = {eoverr_norm:.6f}")

    for i, u_value in enumerate(u_grid):
        y_eq.append(
            simulate_nn_equilibrium_constant_u(
                plant=plant,
                u_value=u_value,
                eoverr_norm=eoverr_norm
            )
        )

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{N_FEASIBLE_GRID}")

    y_eq = np.asarray(y_eq, dtype=np.float32)

    e_mask = np.isclose(np.round(X_train[:, 0, 1], 8), np.round(eoverr_norm, 8))
    observed_y = Y_train[e_mask].reshape(-1, 2)

    support_mask = filter_feasible_set_to_observed_branch(y_eq, observed_y)

    u_grid_filtered = u_grid[support_mask]
    y_eq_filtered = y_eq[support_mask]

    print(f"  retained feasible points: {len(y_eq_filtered)} / {len(y_eq)}")

    return u_grid_filtered, y_eq_filtered

def generate_mprb_reference(feasible_y, seq_len):
    feasible_y = np.asarray(feasible_y, dtype=np.float32)
    n_points = len(feasible_y)

    y0 = np.zeros((seq_len, 2), dtype=np.float32)

    idx = np.random.randint(0, n_points)
    current = feasible_y[idx]
    hold = 0

    for k in range(seq_len):
        if hold <= 0:
            old_idx = idx

            for _ in range(MAX_REF_RESAMPLE_TRIES):
                jump = np.random.randint(-MAX_REF_INDEX_JUMP, MAX_REF_INDEX_JUMP + 1)
                candidate_idx = int(np.clip(old_idx + jump, 0, n_points - 1))
                candidate = feasible_y[candidate_idx]

                old_real = denorm_y(current)
                candidate_real = denorm_y(candidate)

                move_ca = abs(candidate_real[0] - old_real[0])
                move_t = abs(candidate_real[1] - old_real[1])

                if move_ca >= MIN_REF_MOVE_CA or move_t >= MIN_REF_MOVE_T:
                    idx = candidate_idx
                    current = candidate
                    break

            hold = np.random.randint(MIN_HOLD, MAX_HOLD + 1)

        y0[k] = current
        hold -= 1

    return y0

def generate_controller_reference_dataset(feasible_y, eoverr_norm, n_sequences, seq_len):
    raw_setpoints = []
    filtered_refs = []

    for _ in range(n_sequences):
        for _attempt in range(MAX_REF_RESAMPLE_TRIES):
            y0 = generate_mprb_reference(feasible_y, seq_len)
            y_ref = first_order_reference_filter(y0)

            y_ref_real = denorm_y(y_ref)

            ca_range = np.max(y_ref_real[:, 0]) - np.min(y_ref_real[:, 0])
            t_range = np.max(y_ref_real[:, 1]) - np.min(y_ref_real[:, 1])

            if ca_range >= MIN_SEQ_RANGE_CA or t_range >= MIN_SEQ_RANGE_T:
                break

        raw_setpoints.append(y0)
        filtered_refs.append(y_ref)

    raw_setpoints = np.asarray(raw_setpoints, dtype=np.float32)
    filtered_refs = np.asarray(filtered_refs, dtype=np.float32)

    controller_inputs = make_controller_input(filtered_refs, eoverr_norm)

    return raw_setpoints, filtered_refs, controller_inputs

all_Y0_train = []
all_Yref_train = []
all_Cin_train = []

all_Y0_val = []
all_Yref_val = []
all_Cin_val = []

all_Y0_test = []
all_Yref_test = []
all_Cin_test = []

feasible_sets_by_e = {}

for eoverr_norm in EOVER_R_NORM_VALUES:
    u_grid_e, feasible_y_e = build_feasible_set_for_e(
        plant=plant_model,
        eoverr_norm=eoverr_norm
    )

    feasible_sets_by_e[float(eoverr_norm)] = {
        "u_grid": u_grid_e,
        "y_model_feasible": feasible_y_e,
    }

    Y0_all, Y_ref_all, C_input_all = generate_controller_reference_dataset(
        feasible_y=feasible_y_e,
        eoverr_norm=eoverr_norm,
        n_sequences=N_REF_TOTAL_PER_E,
        seq_len=REF_SEQ_LEN
    )

    all_Y0_train.append(Y0_all[:N_REF_TRAIN_PER_E])
    all_Yref_train.append(Y_ref_all[:N_REF_TRAIN_PER_E])
    all_Cin_train.append(C_input_all[:N_REF_TRAIN_PER_E])

    all_Y0_val.append(Y0_all[N_REF_TRAIN_PER_E:N_REF_TRAIN_PER_E + N_REF_VAL_PER_E])
    all_Yref_val.append(Y_ref_all[N_REF_TRAIN_PER_E:N_REF_TRAIN_PER_E + N_REF_VAL_PER_E])
    all_Cin_val.append(C_input_all[N_REF_TRAIN_PER_E:N_REF_TRAIN_PER_E + N_REF_VAL_PER_E])

    all_Y0_test.append(Y0_all[N_REF_TRAIN_PER_E + N_REF_VAL_PER_E:])
    all_Yref_test.append(Y_ref_all[N_REF_TRAIN_PER_E + N_REF_VAL_PER_E:])
    all_Cin_test.append(C_input_all[N_REF_TRAIN_PER_E + N_REF_VAL_PER_E:])

Y0_train = np.concatenate(all_Y0_train, axis=0)
Y_ref_train = np.concatenate(all_Yref_train, axis=0)
C_input_train = np.concatenate(all_Cin_train, axis=0)

Y0_val = np.concatenate(all_Y0_val, axis=0)
Y_ref_val = np.concatenate(all_Yref_val, axis=0)
C_input_val = np.concatenate(all_Cin_val, axis=0)

Y0_test = np.concatenate(all_Y0_test, axis=0)
Y_ref_test = np.concatenate(all_Yref_test, axis=0)
C_input_test = np.concatenate(all_Cin_test, axis=0)

perm = np.random.permutation(len(C_input_train))
C_input_train = C_input_train[perm]
Y_ref_train = Y_ref_train[perm]
Y0_train = Y0_train[perm]

print("\nController reference dataset:")
print("C_input_train:", C_input_train.shape)
print("Y_ref_train:  ", Y_ref_train.shape)
print("C_input_val:  ", C_input_val.shape)
print("Y_ref_val:    ", Y_ref_val.shape)
print("C_input_test: ", C_input_test.shape)
print("Y_ref_test:   ", Y_ref_test.shape)

ref_plot_idx = 0
raw_ref_real = denorm_y(Y0_test[ref_plot_idx])
filtered_ref_real = denorm_y(Y_ref_test[ref_plot_idx])
e_ref_real = denorm_e(C_input_test[ref_plot_idx, :, 2])
t_ref = np.arange(REF_SEQ_LEN) * DT

fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.8), sharex=True)

axes[0].step(t_ref, raw_ref_real[:, 0], where="post", color="#8a8a8a", linestyle=":", linewidth=1.8, label=r"Raw set-point $C_A$")
axes[0].plot(t_ref, filtered_ref_real[:, 0], color="#1f77b4", linewidth=2.0, label=r"Filtered reference $C_A$")
axes[0].set_ylabel(r"$C_A$ [mol/L]")
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[0])

axes[1].step(t_ref, raw_ref_real[:, 1], where="post", color="#8a8a8a", linestyle=":", linewidth=1.8, label=r"Raw set-point $T$")
axes[1].plot(t_ref, filtered_ref_real[:, 1], color="#d62728", linewidth=2.0, label=r"Filtered reference $T$")
axes[1].set_ylabel(r"$T$ [K]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[1])

axes[2].plot(t_ref, e_ref_real, color="#7b3294", linewidth=2.0)
axes[2].set_ylabel(r"$E/R$ [K]")
axes[2].set_xlabel("Time [s]")
polish_axes(axes[2])

fig.suptitle("Representative Feasible Reference Trajectory", y=0.98)
fig.tight_layout()
save_figure(fig, "controller_reference_trajectory")

train_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(C_input_train),
        torch.FloatTensor(Y_ref_train)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(C_input_val),
        torch.FloatTensor(Y_ref_val)
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

controller = ControllerGRU(
    input_size=CONTROLLER_INPUT_SIZE,
    hidden_size=CONTROLLER_HIDDEN_SIZE,
    num_layers=CONTROLLER_NUM_LAYERS,
    output_size=CONTROLLER_OUTPUT_SIZE,
    u_low=u_low,
    u_high=u_high
).to(DEVICE)

optimizer = optim.AdamW(
    controller.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=12
)

criterion = nn.MSELoss()

best_val_tracking = float("inf")
best_epoch = 0
best_controller_state = copy.deepcopy(controller.state_dict())

early_stop_counter = 0
stopped_epoch = EPOCHS

train_tracking_losses = []
val_tracking_losses = []

def overshoot_penalty(y_pred, y_ref):
    error = y_pred[:, WASHOUT:, :] - y_ref[:, WASHOUT:, :]
    direction = y_ref[:, WASHOUT:, :] - y_ref[:, WASHOUT:WASHOUT + 1, :]
    signed_overshoot = error * torch.sign(direction + 1e-8)
    return torch.mean(torch.relu(signed_overshoot) ** 2)

def compute_controller_loss(controller_input_batch, ref_batch, epoch):
    u_seq, _ = controller(controller_input_batch)

    e_seq = controller_input_batch[:, :, 2:3]
    plant_input = make_plant_input(u_seq, e_seq)

    y_pred, _ = plant_model(plant_input)

    tracking_loss = criterion(
        y_pred[:, WASHOUT:, :],
        ref_batch[:, WASHOUT:, :]
    )

    du = u_seq[:, 1:, :] - u_seq[:, :-1, :]
    smooth_loss = torch.mean(du ** 2)
    mag_loss = torch.mean(u_seq ** 2)
    os_loss = overshoot_penalty(y_pred, ref_batch)

    iss_loss = gru_delta_iss_penalty(
        controller.gru,
        controller.num_layers,
        margin=ISS_MARGIN
    )

    iss_weight = CONTROL_ISS_WEIGHT * min(1.0, epoch / 100)

    total_loss = (
        tracking_loss
        + CONTROL_SMOOTH_WEIGHT * smooth_loss
        + CONTROL_MAG_WEIGHT * mag_loss
        + OVERSHOOT_WEIGHT * os_loss
        + iss_weight * iss_loss
    )

    return total_loss, tracking_loss

print(f"\nTraining controller on {DEVICE}...\n")

for epoch in range(EPOCHS):
    controller.train()

    train_tracking_epoch = []

    for c_batch, r_batch in train_loader:
        c_batch = c_batch.to(DEVICE)
        r_batch = r_batch.to(DEVICE)

        optimizer.zero_grad()

        total_loss, tracking_loss = compute_controller_loss(
            c_batch,
            r_batch,
            epoch
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            controller.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_tracking_epoch.append(tracking_loss.item())

    train_tracking = float(np.mean(train_tracking_epoch))
    train_tracking_losses.append(train_tracking)

    controller.eval()

    val_tracking_epoch = []

    with torch.no_grad():
        for c_batch, r_batch in val_loader:
            c_batch = c_batch.to(DEVICE)
            r_batch = r_batch.to(DEVICE)

            _, tracking_loss = compute_controller_loss(
                c_batch,
                r_batch,
                epoch
            )

            val_tracking_epoch.append(tracking_loss.item())

    val_tracking = float(np.mean(val_tracking_epoch))
    val_tracking_losses.append(val_tracking)

    scheduler.step(val_tracking)

    if val_tracking < best_val_tracking - MIN_DELTA:
        best_val_tracking = val_tracking
        best_epoch = epoch + 1
        best_controller_state = copy.deepcopy(controller.state_dict())
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"| Train tracking: {train_tracking:.6e} "
            f"| Val tracking: {val_tracking:.6e} "
            f"| Best val: {best_val_tracking:.6e} @ {best_epoch} "
            f"| No improve: {early_stop_counter}/{EARLY_STOP_PATIENCE} "
            f"| LR: {current_lr:.2e}"
        )

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        stopped_epoch = epoch + 1
        print(
            f"\nEarly stopping at epoch {stopped_epoch}. "
            f"Best validation tracking loss = {best_val_tracking:.6e} @ epoch {best_epoch}"
        )
        break

controller.load_state_dict(best_controller_state)
controller.eval()

print(f"\nLoaded best controller from epoch {best_epoch}")
print(f"Best validation tracking loss: {best_val_tracking:.6e}")

torch.save(
    {
        "controller_state_dict": controller.state_dict(),
        "controller_config": {
            "input_size": CONTROLLER_INPUT_SIZE,
            "hidden_size": CONTROLLER_HIDDEN_SIZE,
            "num_layers": CONTROLLER_NUM_LAYERS,
            "output_size": CONTROLLER_OUTPUT_SIZE,
            "input_features": [
                "Ca_ref_norm",
                "T_ref_norm",
                "EoverR_norm"
            ],
            "output_features": ["Tc_norm"],
            "u_low": u_low.tolist(),
            "u_high": u_high.tolist(),
        },
        "plant_config": {
            "input_size": plant_input_size,
            "hidden_size": plant_hidden_size,
            "num_layers": plant_num_layers,
            "output_size": plant_output_size,
            "input_features": [
                "Tc_norm",
                "EoverR_norm"
            ],
            "output_features": [
                "Ca_norm",
                "T_norm"
            ],
        },
        "best_epoch": best_epoch,
        "best_val_tracking_loss": best_val_tracking,
        "train_tracking_losses": train_tracking_losses,
        "val_tracking_losses": val_tracking_losses,
        "early_stopping": {
            "enabled": True,
            "patience": EARLY_STOP_PATIENCE,
            "min_delta": MIN_DELTA,
            "stopped_epoch": stopped_epoch,
        },
        "reference_generation": {
            "single_branch": True,
            "n_feasible_grid": N_FEASIBLE_GRID,
            "support_distance_max": SUPPORT_DISTANCE_MAX,
            "max_ref_index_jump": MAX_REF_INDEX_JUMP,
            "min_ref_move_ca": MIN_REF_MOVE_CA,
            "min_ref_move_t": MIN_REF_MOVE_T,
            "tau_ref": TAU_REF,
            "dt": DT,
        },
    },
    CONTROLLER_SAVE_PATH
)

print(f"\nController saved to: {CONTROLLER_SAVE_PATH}")

epochs = np.arange(1, len(train_tracking_losses) + 1)

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.scatter(
    epochs,
    train_tracking_losses,
    s=28,
    color="#1f77b4",
    label="Training"
)

ax.scatter(
    epochs,
    val_tracking_losses,
    s=28,
    color="#d62728",
    label="Validation"
)

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Tracking loss")
ax.set_title("Controller Training Loss")
polish_axes(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

fig.tight_layout()
save_figure(fig, "controller_training_loss_evolution")

def fit_index(y_true, y_pred):
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(
        y_true - np.mean(y_true, axis=0, keepdims=True)
    )
    return 100.0 * (1.0 - numerator / (denominator + 1e-8))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def evaluate_controller(C_input):
    controller.eval()
    plant_model.eval()

    preds = []
    us = []

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(C_input)),
        batch_size=64,
        shuffle=False
    )

    with torch.no_grad():
        for (c_batch,) in loader:
            c_batch = c_batch.to(DEVICE)

            u_seq, _ = controller(c_batch)
            e_seq = c_batch[:, :, 2:3]

            plant_input = make_plant_input(u_seq, e_seq)
            y_pred, _ = plant_model(plant_input)

            preds.append(y_pred.cpu().numpy())
            us.append(u_seq.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(us, axis=0)

Y_pred_test, U_pred_test = evaluate_controller(C_input_test)

seq_fits = []
seq_ca_rmse = []
seq_t_rmse = []

for i in range(len(Y_ref_test)):
    y_true_real = denorm_y(Y_ref_test[i, WASHOUT:, :])
    y_pred_real = denorm_y(Y_pred_test[i, WASHOUT:, :])

    seq_fits.append(fit_index(y_true_real, y_pred_real))
    seq_ca_rmse.append(rmse(y_true_real[:, 0], y_pred_real[:, 0]))
    seq_t_rmse.append(rmse(y_true_real[:, 1], y_pred_real[:, 1]))

seq_fits = np.asarray(seq_fits)
seq_ca_rmse = np.asarray(seq_ca_rmse)
seq_t_rmse = np.asarray(seq_t_rmse)

best_seq_id = int(np.argmax(seq_fits))
median_seq_id = int(np.argsort(np.abs(seq_fits - np.median(seq_fits)))[0])
worst_seq_id = int(np.argmin(seq_fits))

overall_fit = fit_index(
    denorm_y(Y_ref_test[:, WASHOUT:, :].reshape(-1, 2)),
    denorm_y(Y_pred_test[:, WASHOUT:, :].reshape(-1, 2))
)

fit_table = pd.DataFrame({
    "Metric": [
        "Overall FIT [%]",
        "Median sequence FIT [%]",
        "Best sequence FIT [%]",
        "Worst sequence FIT [%]",
        "Mean Ca RMSE [mol/L]",
        "Mean T RMSE [K]",
        "Median Ca RMSE [mol/L]",
        "Median T RMSE [K]",
        "Best sequence ID",
        "Representative sequence ID",
        "Worst sequence ID",
    ],
    "Value": [
        overall_fit,
        float(np.median(seq_fits)),
        float(seq_fits[best_seq_id]),
        float(seq_fits[worst_seq_id]),
        float(np.mean(seq_ca_rmse)),
        float(np.mean(seq_t_rmse)),
        float(np.median(seq_ca_rmse)),
        float(np.median(seq_t_rmse)),
        best_seq_id,
        median_seq_id,
        worst_seq_id,
    ]
})

fit_table.to_csv(BASE_DIR / "controller_fit_metric_table.csv", index=False)

print("\nController FIT metric table:")
print(fit_table.to_string(index=False))

plot_id = best_seq_id

ref_real = denorm_y(Y_ref_test[plot_id])
pred_real = denorm_y(Y_pred_test[plot_id])
u_real = denorm_tc(U_pred_test[plot_id, :, 0])
e_real = denorm_e(C_input_test[plot_id, :, 2])
time = np.arange(REF_SEQ_LEN) * DT

fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.2), sharex=True)

axes[0].plot(time, e_real, color="#7b3294", linewidth=2.0)
axes[0].set_ylabel(r"$E/R$ [K]")
polish_axes(axes[0])

axes[1].plot(time, ref_real[:, 0], color="#111111", linewidth=2.0, label=r"Reference $C_A$")
axes[1].plot(time, pred_real[:, 0], color="#d62728", linestyle="--", linewidth=2.0, label=r"NN plant output $C_A$")
axes[1].set_ylabel(r"$C_A$ [mol/L]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[1])

axes[2].plot(time, ref_real[:, 1], color="#111111", linewidth=2.0, label=r"Reference $T$")
axes[2].plot(time, pred_real[:, 1], color="#d62728", linestyle="--", linewidth=2.0, label=r"NN plant output $T$")
axes[2].set_ylabel(r"$T$ [K]")
axes[2].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[2])

axes[3].plot(time, u_real, color="#1f77b4", linewidth=2.0, label=r"Controller output $T_c$")
axes[3].set_ylabel(r"$T_c$ [K]")
axes[3].set_xlabel("Time [s]")
axes[3].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[3])

fig.suptitle(
    f"Best Controller Test Tracking | FIT = {seq_fits[plot_id]:.2f}%",
    y=0.98
)

fig.tight_layout()
save_figure(fig, "controller_representative_test_prediction")

print("\nDone.")
print("Generated:")
print("- cstr_gru_controller_deltaISS_transfer.pth")
print("- controller_training_loss_evolution.png / .pdf")
print("- controller_reference_trajectory.png / .pdf")
print("- controller_representative_test_prediction.png / .pdf")
print("- controller_fit_metric_table.csv")
