import copy
import random
from pathlib import Path

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

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR
SHOW_FIGURES = False

PLANT_CHECKPOINT = BASE_DIR / "cstr_gru_deltaISS.pth"
CONTROLLER_SAVE_PATH = OUTPUT_DIR / "cstr_gru_controller_deltaISS_transfer.pth"

X_TRAIN_PATH = BASE_DIR / "X_train.npy"
Y_TRAIN_PATH = BASE_DIR / "Y_train.npy"
X_VAL_PATH = BASE_DIR / "X_val.npy"
Y_VAL_PATH = BASE_DIR / "Y_val.npy"
X_TEST_PATH = BASE_DIR / "X_test.npy"
Y_TEST_PATH = BASE_DIR / "Y_test.npy"

NORMALIZATION_STATS_PATH = BASE_DIR / "normalization_stats.csv"
if not NORMALIZATION_STATS_PATH.exists():
    NORMALIZATION_STATS_PATH = BASE_DIR / "Normalization_Stats_Multi_EoverR.csv"

EXPECTED_PLANT_INPUT_SIZE = 2
PLANT_OUTPUT_SIZE = 2

CONTROLLER_INPUT_SIZE = 3
CONTROLLER_HIDDEN_SIZE = 64
CONTROLLER_NUM_LAYERS = 3
CONTROLLER_OUTPUT_SIZE = 1

BATCH_SIZE = 32
EPOCHS = 250
LR = 3e-4
WEIGHT_DECAY = 1e-6
WASHOUT = 100

N_REF_TOTAL_PER_E = 430
N_REF_TRAIN_PER_E = 380
N_REF_VAL_PER_E = 40
N_REF_TEST_PER_E = 10

REF_SEQ_LEN = 700
MIN_HOLD = 80
MAX_HOLD = 300
TAU_REF = 16.0
DT = 1.0

N_FEASIBLE_GRID = 400
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

KEEP_LONGEST_CONTIGUOUS_FEASIBLE_SEGMENT = True
MAX_REF_INDEX_JUMP = 20

MIN_REF_MOVE_CA = 0.005
MIN_REF_MOVE_T = 2.0
MIN_SEQ_RANGE_CA = 0.010
MIN_SEQ_RANGE_T = 4.0
MAX_REF_RESAMPLE_TRIES = 80

CONTROL_SMOOTH_WEIGHT = 2e-3
CONTROL_MAG_WEIGHT = 1e-6
CONTROL_ISS_WEIGHT = 1e-5
CONTROL_OVERSHOOT_WEIGHT = 2e-2
OVERSHOOT_DEADBAND = 0.02
ISS_MARGIN = 1e-3
ISS_RAMP_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 35
BEST_MIN_DELTA = 1e-7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.1,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

COLORS = {
    "train": "#1f77b4",
    "val": "#d62728",
    "ref": "#111111",
    "pred": "#d62728",
    "u": "#1f77b4",
}


def polish_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, filename):
    png_path = OUTPUT_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
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


def denorm_y(y_norm):
    y_norm = np.asarray(y_norm)
    y_real = np.empty_like(y_norm, dtype=float)
    y_real[..., 0] = y_norm[..., 0] * Ca_std + Ca_mean
    y_real[..., 1] = y_norm[..., 1] * T_std + T_mean
    return y_real


def denorm_u(u_norm):
    return np.asarray(u_norm) * Tc_std + Tc_mean


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
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                self.num_layers,
                x.size(0),
                self.hidden_size,
                device=x.device,
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
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

        u_low = torch.tensor(u_low, dtype=torch.float32).view(1, 1, output_size)
        u_high = torch.tensor(u_high, dtype=torch.float32).view(1, 1, output_size)

        self.register_buffer("u_mid", 0.5 * (u_low + u_high))
        self.register_buffer("u_half_range", 0.5 * (u_high - u_low))

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                self.num_layers,
                x.size(0),
                self.hidden_size,
                device=x.device,
            )

        out, h_n = self.gru(x, h)
        raw_u = self.fc(out)
        u = self.u_mid + self.u_half_range * torch.tanh(raw_u)

        return u, h_n


X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
Y_val = np.load(Y_VAL_PATH)
X_test = np.load(X_TEST_PATH)
Y_test = np.load(Y_TEST_PATH)

if X_train.shape[-1] != EXPECTED_PLANT_INPUT_SIZE:
    raise RuntimeError(
        f"Expected X feature size {EXPECTED_PLANT_INPUT_SIZE}, got {X_train.shape[-1]}."
    )

if Y_train.shape[-1] != PLANT_OUTPUT_SIZE:
    raise RuntimeError(
        f"Expected Y output size {PLANT_OUTPUT_SIZE}, got {Y_train.shape[-1]}."
    )

print("Loaded normalized data:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:  ", X_val.shape)
print("Y_val:  ", Y_val.shape)
print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)

EOVER_R_NORM_VALUES = np.unique(np.round(X_train[:, 0, 1], 8))
EOVER_R_NORM_VALUES = np.sort(EOVER_R_NORM_VALUES)

print("\nDetected normalized activation-energy values:")
print(EOVER_R_NORM_VALUES)

Y_TRAIN_FLAT = Y_train.reshape(-1, Y_train.shape[-1]).astype(np.float32)
E_TRAIN_FLAT = X_train[:, :, 1].reshape(-1).astype(np.float32)


def make_eoverr_column(batch_size, seq_len, eoverr_norm, device):
    return torch.full(
        (batch_size, seq_len, 1),
        float(eoverr_norm),
        dtype=torch.float32,
        device=device,
    )


def make_plant_input_from_u_and_e(u_seq, e_seq):
    return torch.cat([u_seq, e_seq], dim=2)


def make_controller_input(ref_seq_np, eoverr_norm):
    e_feature = np.ones(
        (ref_seq_np.shape[0], ref_seq_np.shape[1], 1),
        dtype=np.float32,
    ) * float(eoverr_norm)

    return np.concatenate([ref_seq_np, e_feature], axis=2)


def load_frozen_plant():
    checkpoint = torch.load(PLANT_CHECKPOINT, map_location=DEVICE, weights_only=False)

    input_size = checkpoint.get("input_size", EXPECTED_PLANT_INPUT_SIZE)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 3)
    output_size = checkpoint.get("output_size", PLANT_OUTPUT_SIZE)

    plant = CSTR_GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    ).to(DEVICE)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    plant.load_state_dict(state_dict)

    plant.eval()
    for param in plant.parameters():
        param.requires_grad = False

    return plant, checkpoint


plant_model, plant_checkpoint = load_frozen_plant()

print("\nFrozen NN plant model loaded successfully.")


def observed_support_for_e(eoverr_norm):
    mask = np.isclose(E_TRAIN_FLAT, float(eoverr_norm), atol=1e-5)
    support = Y_TRAIN_FLAT[mask]

    if len(support) == 0:
        return Y_TRAIN_FLAT[::OBSERVED_SUPPORT_STRIDE]

    return support[::OBSERVED_SUPPORT_STRIDE]


def filter_feasible_by_observed_support(y_feasible, eoverr_norm):
    if not USE_OBSERVED_SUPPORT_FILTER:
        return np.ones(len(y_feasible), dtype=bool)

    support = observed_support_for_e(eoverr_norm)

    keep = np.zeros(len(y_feasible), dtype=bool)

    for i, y in enumerate(y_feasible):
        distances = np.linalg.norm(support - y.reshape(1, -1), axis=1)
        keep[i] = np.min(distances) <= SUPPORT_DISTANCE_MAX

    return keep


def filter_feasible_by_branch_bounds(y_feasible, eoverr_norm):
    support = observed_support_for_e(eoverr_norm)
    support_real = denorm_y(support)

    ca_low, ca_high = np.percentile(
        support_real[:, 0],
        [BRANCH_SUPPORT_LOWER_PCT, BRANCH_SUPPORT_UPPER_PCT],
    )

    t_low, t_high = np.percentile(
        support_real[:, 1],
        [BRANCH_SUPPORT_LOWER_PCT, BRANCH_SUPPORT_UPPER_PCT],
    )

    y_real = denorm_y(y_feasible)

    keep = (
        (y_real[:, 0] >= ca_low - CA_BRANCH_MARGIN)
        & (y_real[:, 0] <= ca_high + CA_BRANCH_MARGIN)
        & (y_real[:, 1] >= t_low - T_BRANCH_MARGIN)
        & (y_real[:, 1] <= t_high + T_BRANCH_MARGIN)
    )

    return keep


def longest_contiguous_true_segment(mask):
    best_start = 0
    best_len = 0
    current_start = None

    for i, value in enumerate(mask):
        if value and current_start is None:
            current_start = i

        if (not value or i == len(mask) - 1) and current_start is not None:
            end = i if value and i == len(mask) - 1 else i - 1
            length = end - current_start + 1

            if length > best_len:
                best_start = current_start
                best_len = length

            current_start = None

    final = np.zeros_like(mask, dtype=bool)

    if best_len > 0:
        final[best_start:best_start + best_len] = True

    return final


def simulate_model_equilibrium_for_constant_u(u_value, eoverr_norm):
    u_seq = torch.full(
        (1, N_SETTLING_STEPS, 1),
        float(u_value),
        dtype=torch.float32,
        device=DEVICE,
    )

    e_seq = make_eoverr_column(
        batch_size=1,
        seq_len=N_SETTLING_STEPS,
        eoverr_norm=eoverr_norm,
        device=DEVICE,
    )

    plant_input = make_plant_input_from_u_and_e(u_seq, e_seq)

    with torch.no_grad():
        y_seq, _ = plant_model(plant_input)

    y_np = y_seq.cpu().numpy()[0]

    return np.mean(y_np[-N_AVG_TAIL:], axis=0)


def build_feasible_set_for_e(u_low, u_high, eoverr_norm):
    u_grid = np.linspace(float(u_low), float(u_high), N_FEASIBLE_GRID)

    y_eq = []

    print(f"\nBuilding feasible set for EoverR_norm = {eoverr_norm:.6f}")

    for i, u_value in enumerate(u_grid):
        y_eq.append(simulate_model_equilibrium_for_constant_u(u_value, eoverr_norm))

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{N_FEASIBLE_GRID}")

    y_eq = np.asarray(y_eq, dtype=np.float32)

    keep_support = filter_feasible_by_observed_support(y_eq, eoverr_norm)
    keep_branch = filter_feasible_by_branch_bounds(y_eq, eoverr_norm)

    keep = keep_support & keep_branch

    if KEEP_LONGEST_CONTIGUOUS_FEASIBLE_SEGMENT:
        keep = longest_contiguous_true_segment(keep)

    if np.sum(keep) < MIN_FEASIBLE_POINTS:
        print("  Warning: feasible filter too strict; falling back to branch bounds only.")
        keep = keep_branch

    if np.sum(keep) < MIN_FEASIBLE_POINTS:
        print("  Warning: branch filter too strict; using full NN equilibrium set.")
        keep = np.ones(len(y_eq), dtype=bool)

    return u_grid[keep], y_eq[keep]


def first_order_reference_filter(setpoints):
    alpha = np.clip(DT / TAU_REF, 0.0, 1.0)

    y_ref = np.zeros_like(setpoints, dtype=np.float32)
    y_ref[0] = setpoints[0]

    for k in range(len(setpoints) - 1):
        y_ref[k + 1] = y_ref[k] + alpha * (setpoints[k] - y_ref[k])

    return y_ref


def generate_raw_reference(feasible_y, n_steps):
    y0 = np.zeros((n_steps, 2), dtype=np.float32)

    hold = 0
    current_index = np.random.randint(0, len(feasible_y))
    current = feasible_y[current_index]

    for k in range(n_steps):
        if hold <= 0:
            for _ in range(MAX_REF_RESAMPLE_TRIES):
                candidate_index = np.random.randint(0, len(feasible_y))
                candidate = feasible_y[candidate_index]

                current_real = denorm_y(current)
                candidate_real = denorm_y(candidate)

                ca_move = abs(candidate_real[0] - current_real[0])
                t_move = abs(candidate_real[1] - current_real[1])
                index_move = abs(candidate_index - current_index)

                if (
                    ca_move >= MIN_REF_MOVE_CA
                    or t_move >= MIN_REF_MOVE_T
                ) and index_move <= MAX_REF_INDEX_JUMP:
                    current_index = candidate_index
                    current = candidate
                    break

            hold = np.random.randint(MIN_HOLD, MAX_HOLD + 1)

        y0[k] = current
        hold -= 1

    return y0


def generate_reference_dataset_for_e(feasible_y, eoverr_norm, n_sequences, seq_len):
    raw_setpoints = []
    filtered_refs = []

    attempts = 0

    while len(filtered_refs) < n_sequences:
        attempts += 1

        y0 = generate_raw_reference(feasible_y, seq_len)
        y_ref = first_order_reference_filter(y0)

        y_ref_real = denorm_y(y_ref)
        ca_range = np.max(y_ref_real[:, 0]) - np.min(y_ref_real[:, 0])
        t_range = np.max(y_ref_real[:, 1]) - np.min(y_ref_real[:, 1])

        if ca_range < MIN_SEQ_RANGE_CA and t_range < MIN_SEQ_RANGE_T:
            if attempts < n_sequences * MAX_REF_RESAMPLE_TRIES:
                continue

        raw_setpoints.append(y0)
        filtered_refs.append(y_ref)

    raw_setpoints = np.asarray(raw_setpoints, dtype=np.float32)
    filtered_refs = np.asarray(filtered_refs, dtype=np.float32)

    controller_inputs = make_controller_input(filtered_refs, eoverr_norm)

    return raw_setpoints, filtered_refs, controller_inputs


def inf_norm(mat):
    return torch.max(torch.sum(torch.abs(mat), dim=1))


def gru_delta_iss_penalty(gru_module, num_layers, margin=ISS_MARGIN):
    penalty = 0.0

    for layer in range(num_layers):
        weight_hh = getattr(gru_module, f"weight_hh_l{layer}")
        weight_ih = getattr(gru_module, f"weight_ih_l{layer}")

        bias_hh = getattr(gru_module, f"bias_hh_l{layer}")
        bias_ih = getattr(gru_module, f"bias_ih_l{layer}")

        bias = bias_hh + bias_ih

        Ur, Uz, Uh = weight_hh.chunk(3, 0)
        Wr, Wz, Wh = weight_ih.chunk(3, 0)
        br, bz, bh = bias.chunk(3, 0)

        sz = torch.sigmoid(
            inf_norm(torch.cat([Wz, Uz, bz.unsqueeze(1)], dim=1))
        )

        sf = torch.sigmoid(
            inf_norm(torch.cat([Wr, Ur, br.unsqueeze(1)], dim=1))
        )

        pr = torch.tanh(
            inf_norm(torch.cat([Wh, Uh, bh.unsqueeze(1)], dim=1))
        )

        nu = (
            inf_norm(Uh) * (0.25 * inf_norm(Uh) + sf)
            + 0.25 * ((1.0 + pr) / (1.0 - sz + 1e-8)) * inf_norm(Uz)
            - 1.0
        )

        penalty = penalty + torch.relu(nu + margin)

    return penalty


def overshoot_penalty(y_pred, y_ref):
    error = y_pred - y_ref
    excess = torch.relu(torch.abs(error) - OVERSHOOT_DEADBAND)
    return torch.mean(excess ** 2)


flat_tc = X_train[:, :, 0].reshape(-1, 1)
u_low = np.percentile(flat_tc, 1, axis=0)
u_high = np.percentile(flat_tc, 99, axis=0)

print("\nController normalized Tc bounds:")
print("u_low: ", u_low)
print("u_high:", u_high)

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
    u_grid, y_feasible = build_feasible_set_for_e(
        u_low=u_low[0],
        u_high=u_high[0],
        eoverr_norm=eoverr_norm,
    )

    feasible_sets_by_e[float(eoverr_norm)] = {
        "u_grid": u_grid,
        "y_model_feasible": y_feasible,
    }

    Y0_all, Y_ref_all, C_input_all = generate_reference_dataset_for_e(
        feasible_y=y_feasible,
        eoverr_norm=eoverr_norm,
        n_sequences=N_REF_TOTAL_PER_E,
        seq_len=REF_SEQ_LEN,
    )

    all_Y0_train.append(Y0_all[:N_REF_TRAIN_PER_E])
    all_Yref_train.append(Y_ref_all[:N_REF_TRAIN_PER_E])
    all_Cin_train.append(C_input_all[:N_REF_TRAIN_PER_E])

    val_start = N_REF_TRAIN_PER_E
    val_end = val_start + N_REF_VAL_PER_E

    all_Y0_val.append(Y0_all[val_start:val_end])
    all_Yref_val.append(Y_ref_all[val_start:val_end])
    all_Cin_val.append(C_input_all[val_start:val_end])

    all_Y0_test.append(Y0_all[val_end:])
    all_Yref_test.append(Y_ref_all[val_end:])
    all_Cin_test.append(C_input_all[val_end:])

Y_ref_train = np.concatenate(all_Yref_train, axis=0)
C_input_train = np.concatenate(all_Cin_train, axis=0)

Y_ref_val = np.concatenate(all_Yref_val, axis=0)
C_input_val = np.concatenate(all_Cin_val, axis=0)

Y_ref_test = np.concatenate(all_Yref_test, axis=0)
C_input_test = np.concatenate(all_Cin_test, axis=0)

perm = np.random.permutation(len(C_input_train))
C_input_train = C_input_train[perm]
Y_ref_train = Y_ref_train[perm]

print("\nController reference dataset:")
print("C_input_train:", C_input_train.shape)
print("Y_ref_train:  ", Y_ref_train.shape)
print("C_input_val:  ", C_input_val.shape)
print("Y_ref_val:    ", Y_ref_val.shape)
print("C_input_test: ", C_input_test.shape)
print("Y_ref_test:   ", Y_ref_test.shape)

train_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(C_input_train),
        torch.FloatTensor(Y_ref_train),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(C_input_val),
        torch.FloatTensor(Y_ref_val),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

controller = ControllerGRU(
    input_size=CONTROLLER_INPUT_SIZE,
    hidden_size=CONTROLLER_HIDDEN_SIZE,
    num_layers=CONTROLLER_NUM_LAYERS,
    output_size=CONTROLLER_OUTPUT_SIZE,
    u_low=u_low,
    u_high=u_high,
).to(DEVICE)

optimizer = optim.AdamW(
    controller.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=12,
)

criterion = nn.MSELoss()


def controller_loss(controller_input, ref_batch, epoch):
    u_seq, _ = controller(controller_input)

    e_seq = controller_input[:, :, 2:3]

    plant_input = make_plant_input_from_u_and_e(
        u_seq=u_seq,
        e_seq=e_seq,
    )

    y_pred, _ = plant_model(plant_input)

    tracking_loss = criterion(
        y_pred[:, WASHOUT:, :],
        ref_batch[:, WASHOUT:, :],
    )

    du = u_seq[:, 1:, :] - u_seq[:, :-1, :]
    smooth_loss = torch.mean(du ** 2)
    mag_loss = torch.mean(u_seq ** 2)

    os_loss = overshoot_penalty(
        y_pred[:, WASHOUT:, :],
        ref_batch[:, WASHOUT:, :],
    )

    iss_loss = gru_delta_iss_penalty(
        controller.gru,
        controller.num_layers,
    )

    iss_weight = CONTROL_ISS_WEIGHT * min(1.0, epoch / ISS_RAMP_EPOCHS)

    total_loss = (
        tracking_loss
        + CONTROL_SMOOTH_WEIGHT * smooth_loss
        + CONTROL_MAG_WEIGHT * mag_loss
        + CONTROL_OVERSHOOT_WEIGHT * os_loss
        + iss_weight * iss_loss
    )

    return total_loss, tracking_loss, os_loss


best_selection_loss = float("inf")
best_tracking_loss = float("inf")
best_epoch = 0
best_state = copy.deepcopy(controller.state_dict())
epochs_without_improvement = 0

train_tracking_losses = []
val_tracking_losses = []
train_total_losses = []
val_selection_losses = []

print(f"\nTraining controller on {DEVICE}...\n")

for epoch in range(1, EPOCHS + 1):
    controller.train()

    batch_total = []
    batch_tracking = []

    for c_batch, r_batch in train_loader:
        c_batch = c_batch.to(DEVICE)
        r_batch = r_batch.to(DEVICE)

        optimizer.zero_grad()

        total_loss, tracking_loss, _ = controller_loss(
            c_batch,
            r_batch,
            epoch,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        optimizer.step()

        batch_total.append(float(total_loss.detach().cpu()))
        batch_tracking.append(float(tracking_loss.detach().cpu()))

    train_total = float(np.mean(batch_total))
    train_tracking = float(np.mean(batch_tracking))

    controller.eval()

    val_tracking_batch = []
    val_selection_batch = []

    with torch.no_grad():
        for c_batch, r_batch in val_loader:
            c_batch = c_batch.to(DEVICE)
            r_batch = r_batch.to(DEVICE)

            total_loss, tracking_loss, os_loss = controller_loss(
                c_batch,
                r_batch,
                epoch,
            )

            selection_loss = tracking_loss + CONTROL_OVERSHOOT_WEIGHT * os_loss

            val_tracking_batch.append(float(tracking_loss.detach().cpu()))
            val_selection_batch.append(float(selection_loss.detach().cpu()))

    val_tracking = float(np.mean(val_tracking_batch))
    val_selection = float(np.mean(val_selection_batch))

    train_total_losses.append(train_total)
    train_tracking_losses.append(train_tracking)
    val_tracking_losses.append(val_tracking)
    val_selection_losses.append(val_selection)

    scheduler.step(val_tracking)

    if val_selection < best_selection_loss - BEST_MIN_DELTA:
        best_selection_loss = val_selection
        best_tracking_loss = val_tracking
        best_epoch = epoch
        best_state = copy.deepcopy(controller.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch % 10 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"| Train tracking: {train_tracking:.6e} "
            f"| Val tracking: {val_tracking:.6e} "
            f"| Selection: {val_selection:.6e} "
            f"| Best: {best_selection_loss:.6e} @ {best_epoch} "
            f"| LR: {lr:.2e}"
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

controller.load_state_dict(best_state)
controller.eval()

print(f"\nLoaded best controller from epoch {best_epoch}")
print(f"Best validation tracking loss: {best_tracking_loss:.6e}")


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
                "EoverR_norm",
            ],
            "output_features": [
                "Tc_norm",
            ],
            "u_low": u_low.tolist(),
            "u_high": u_high.tolist(),
        },
        "plant_config": {
            "input_size": EXPECTED_PLANT_INPUT_SIZE,
            "output_size": PLANT_OUTPUT_SIZE,
            "input_features": [
                "Tc_norm",
                "EoverR_norm",
            ],
            "output_features": [
                "Ca_norm",
                "T_norm",
            ],
        },
        "training": {
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_val_tracking_loss": best_tracking_loss,
            "best_selection_loss": best_selection_loss,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "washout": WASHOUT,
        },
        "loss_config": {
            "control_smooth_weight": CONTROL_SMOOTH_WEIGHT,
            "control_mag_weight": CONTROL_MAG_WEIGHT,
            "control_iss_weight": CONTROL_ISS_WEIGHT,
            "control_overshoot_weight": CONTROL_OVERSHOOT_WEIGHT,
            "overshoot_deadband": OVERSHOOT_DEADBAND,
            "iss_margin": ISS_MARGIN,
        },
        "train_total_losses": train_total_losses,
        "train_tracking_losses": train_tracking_losses,
        "val_tracking_losses": val_tracking_losses,
        "val_selection_losses": val_selection_losses,
        "reference_generation": {
            "n_ref_total_per_e": N_REF_TOTAL_PER_E,
            "n_ref_train_per_e": N_REF_TRAIN_PER_E,
            "n_ref_val_per_e": N_REF_VAL_PER_E,
            "n_ref_test_per_e": N_REF_TEST_PER_E,
            "tau_ref": TAU_REF,
            "dt": DT,
            "n_feasible_grid": N_FEASIBLE_GRID,
            "use_observed_support_filter": USE_OBSERVED_SUPPORT_FILTER,
            "support_distance_max": SUPPORT_DISTANCE_MAX,
            "branch_support_lower_pct": BRANCH_SUPPORT_LOWER_PCT,
            "branch_support_upper_pct": BRANCH_SUPPORT_UPPER_PCT,
            "ca_branch_margin": CA_BRANCH_MARGIN,
            "t_branch_margin": T_BRANCH_MARGIN,
        },
    },
    CONTROLLER_SAVE_PATH,
)

print(f"\nController saved to: {CONTROLLER_SAVE_PATH}")


def fit_index(y_true, y_pred):
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(
        y_true - np.mean(y_true, axis=0, keepdims=True)
    )
    return 100.0 * (1.0 - numerator / (denominator + 1e-8))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_controller_dataset(C_input):
    controller.eval()
    plant_model.eval()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(C_input)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    y_preds = []
    u_preds = []

    with torch.no_grad():
        for (c_batch,) in loader:
            c_batch = c_batch.to(DEVICE)

            u_seq, _ = controller(c_batch)
            e_seq = c_batch[:, :, 2:3]

            plant_input = make_plant_input_from_u_and_e(
                u_seq=u_seq,
                e_seq=e_seq,
            )

            y_pred, _ = plant_model(plant_input)

            y_preds.append(y_pred.cpu().numpy())
            u_preds.append(u_seq.cpu().numpy())

    return np.concatenate(y_preds, axis=0), np.concatenate(u_preds, axis=0)


Y_pred_test, U_pred_test = evaluate_controller_dataset(C_input_test)

fit_total = fit_index(
    Y_ref_test[:, WASHOUT:, :].reshape(-1, 2),
    Y_pred_test[:, WASHOUT:, :].reshape(-1, 2),
)

fit_ca = fit_index(
    Y_ref_test[:, WASHOUT:, 0:1].reshape(-1, 1),
    Y_pred_test[:, WASHOUT:, 0:1].reshape(-1, 1),
)

fit_t = fit_index(
    Y_ref_test[:, WASHOUT:, 1:2].reshape(-1, 1),
    Y_pred_test[:, WASHOUT:, 1:2].reshape(-1, 1),
)

seq_rows = []

for seq_id in range(len(Y_ref_test)):
    ref_norm = Y_ref_test[seq_id, WASHOUT:, :]
    pred_norm = Y_pred_test[seq_id, WASHOUT:, :]

    ref_real = denorm_y(ref_norm)
    pred_real = denorm_y(pred_norm)

    seq_rows.append({
        "SequenceID": seq_id,
        "EoverR_norm": float(C_input_test[seq_id, 0, 2]),
        "EoverR": float(denorm_e(C_input_test[seq_id, 0, 2])),
        "FIT_percent": fit_index(ref_norm, pred_norm),
        "Ca_RMSE_mol_L": rmse(ref_real[:, 0], pred_real[:, 0]),
        "T_RMSE_K": rmse(ref_real[:, 1], pred_real[:, 1]),
    })

seq_metrics = pd.DataFrame(seq_rows)

representative_seq_id = int(
    (seq_metrics["FIT_percent"] - seq_metrics["FIT_percent"].median())
    .abs()
    .idxmin()
)

representative_fit = float(seq_metrics.loc[representative_seq_id, "FIT_percent"])
representative_ca_rmse = float(seq_metrics.loc[representative_seq_id, "Ca_RMSE_mol_L"])
representative_t_rmse = float(seq_metrics.loc[representative_seq_id, "T_RMSE_K"])

fit_metric_table = pd.DataFrame({
    "Metric": [
        "Overall FIT [%]",
        "Ca FIT [%]",
        "T FIT [%]",
        "Median sequence FIT [%]",
        "Best sequence FIT [%]",
        "Worst sequence FIT [%]",
        "Representative sequence ID",
        "Representative sequence FIT [%]",
        "Representative Ca RMSE [mol/L]",
        "Representative T RMSE [K]",
        "Best validation tracking loss",
        "Best epoch",
    ],
    "Value": [
        fit_total,
        fit_ca,
        fit_t,
        seq_metrics["FIT_percent"].median(),
        seq_metrics["FIT_percent"].max(),
        seq_metrics["FIT_percent"].min(),
        representative_seq_id,
        representative_fit,
        representative_ca_rmse,
        representative_t_rmse,
        best_tracking_loss,
        best_epoch,
    ],
})

fit_metric_table.to_csv(OUTPUT_DIR / "fit_metric_table.csv", index=False)

print("\nFIT metric table:")
print(fit_metric_table.to_string(index=False))


fig, ax = plt.subplots(figsize=(8.4, 4.8))
epochs = np.arange(1, len(train_tracking_losses) + 1)

ax.scatter(
    epochs,
    train_tracking_losses,
    s=22,
    color=COLORS["train"],
    label="Training",
)

ax.scatter(
    epochs,
    val_tracking_losses,
    s=22,
    color=COLORS["val"],
    label="Validation",
)

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Tracking loss")
ax.set_title("Training Loss Evolution")
polish_axes(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

save_figure(fig, "training_loss_evolution.png")


def plot_representative_controller_sequence(seq_id):
    ref_norm = Y_ref_test[seq_id]
    pred_norm = Y_pred_test[seq_id]
    u_norm = U_pred_test[seq_id, :, 0]
    e_norm = C_input_test[seq_id, :, 2]

    ref_real = denorm_y(ref_norm)
    pred_real = denorm_y(pred_norm)
    u_real = denorm_u(u_norm)
    e_real = denorm_e(e_norm)

    seq_fit = fit_index(
        ref_norm[WASHOUT:, :],
        pred_norm[WASHOUT:, :],
    )

    ca_rmse = rmse(
        ref_real[WASHOUT:, 0],
        pred_real[WASHOUT:, 0],
    )

    t_rmse = rmse(
        ref_real[WASHOUT:, 1],
        pred_real[WASHOUT:, 1],
    )

    t_axis = np.arange(len(ref_norm)) * DT

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.4, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.35, 1.35, 1.0]},
    )

    axes[0].plot(
        t_axis,
        e_real,
        color="#7b3294",
        linewidth=1.8,
    )

    axes[1].plot(
        t_axis,
        ref_real[:, 0],
        color=COLORS["ref"],
        linewidth=1.8,
        label=r"Reference $C_A$",
    )

    axes[1].plot(
        t_axis,
        pred_real[:, 0],
        color=COLORS["pred"],
        linestyle="--",
        linewidth=1.8,
        label=r"NN plant output $C_A$",
    )

    axes[2].plot(
        t_axis,
        ref_real[:, 1],
        color=COLORS["ref"],
        linewidth=1.8,
        label=r"Reference $T$",
    )

    axes[2].plot(
        t_axis,
        pred_real[:, 1],
        color=COLORS["pred"],
        linestyle="--",
        linewidth=1.8,
        label=r"NN plant output $T$",
    )

    axes[3].plot(
        t_axis,
        u_real,
        color=COLORS["u"],
        linewidth=1.8,
        label=r"Controller output $T_c$",
    )

    axes[0].set_ylabel(r"$E/R$ [K]")
    axes[1].set_ylabel(r"$C_A$ [mol/L]")
    axes[2].set_ylabel(r"$T$ [K]")
    axes[3].set_ylabel(r"$T_c$ [K]")
    axes[3].set_xlabel("Time [s]")

    axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
    axes[2].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
    axes[3].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

    for ax in axes:
        polish_axes(ax)

    fig.suptitle(
        (
            "Controller Prediction on a Representative Test Sequence | "
            f"FIT = {seq_fit:.2f}% | "
            rf"$C_A$ RMSE = {ca_rmse:.4f} mol/L | "
            rf"$T$ RMSE = {t_rmse:.2f} K"
        ),
        y=0.995,
    )

    save_figure(fig, "representative_controller_test_prediction.png")


plot_representative_controller_sequence(representative_seq_id)


print("\nDone.")
print("Generated:")
print("- cstr_gru_controller_deltaISS_transfer.pth")
print("- training_loss_evolution.png / .pdf")
print("- representative_controller_test_prediction.png / .pdf")
print("- fit_metric_table.csv")
