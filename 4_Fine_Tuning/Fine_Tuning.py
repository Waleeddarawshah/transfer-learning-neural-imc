from pathlib import Path
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

SOURCE_PLANT_CHECKPOINT = BASE_DIR / "cstr_gru_deltaISS.pth"
SOURCE_CONTROLLER_CHECKPOINT = BASE_DIR / "cstr_gru_controller_deltaISS_transfer.pth"

NORMALIZATION_STATS_PATH = BASE_DIR / "normalization_stats.csv"
if not NORMALIZATION_STATS_PATH.exists():
    NORMALIZATION_STATS_PATH = BASE_DIR / "Normalization_Stats_Multi_EoverR.csv"

FINETUNED_PLANT_SAVE_PATH = BASE_DIR / "cstr_gru_deltaISS_target_finetuned.pth"
FINETUNED_CONTROLLER_SAVE_PATH = BASE_DIR / "cstr_gru_controller_target_finetuned.pth"

TARGET_EOVER_R = 0.995e4
TC_NOMINAL = 350.0

DT = 1.0
SIM_TIME = 5000.0
TIME = np.arange(0.0, SIM_TIME, DT)

TARGET_PROFILES = 8
MAX_TARGET_PROFILE_ATTEMPTS = 1000

APRBS_TC_MIN = 320.0
APRBS_TC_MAX = 385.0
APRBS_MIN_HOLD = 500
APRBS_MAX_HOLD = 800

SEQUENCE_LENGTH = 700
STRIDE = 150

PLANT_BATCH_SIZE = 32
PLANT_FINETUNE_EPOCHS = 120
PLANT_LR = 3e-5
PLANT_WEIGHT_DECAY = 1e-6
PLANT_WASHOUT = 100
PLANT_RHO = 1e-4
PLANT_EARLY_STOP_PATIENCE = 25

CTRL_BATCH_SIZE = 32
CTRL_FINETUNE_EPOCHS = 100
CTRL_LR = 2e-5
CTRL_WEIGHT_DECAY = 1e-6
CTRL_WASHOUT = 100
CTRL_EARLY_STOP_PATIENCE = 25

N_REF_TOTAL = 240
N_REF_TRAIN = 200
N_REF_VAL = 30
N_REF_TEST = 10

REF_SEQ_LEN = 700
REF_MIN_HOLD = 80
REF_MAX_HOLD = 220
TAU_REF = 16.0

N_FEASIBLE_GRID = 500
N_SETTLING_STEPS = 1200
N_AVG_TAIL = 100

USE_TARGET_SUPPORT_FILTER = True
TARGET_SUPPORT_STRIDE = 8
SUPPORT_DISTANCE_MAX = 0.16
MIN_FEASIBLE_POINTS = 35

MAX_REF_INDEX_JUMP = 20
MIN_REF_MOVE_CA = 0.003
MIN_REF_MOVE_T = 1.5
MIN_SEQ_RANGE_CA = 0.008
MIN_SEQ_RANGE_T = 4.0
MAX_REF_RESAMPLE_TRIES = 400

CONTROL_SMOOTH_WEIGHT = 2e-3
CONTROL_MAG_WEIGHT = 1e-6
CONTROL_ISS_WEIGHT = 1e-5
CONTROL_OVERSHOOT_WEIGHT = 2e-2
OVERSHOOT_DEADBAND = 0.02
ISS_MARGIN = 1e-3

TC_MIN = 280.0
TC_MAX = 400.0
T_MIN = 250.0
T_MAX = 550.0
CA_MIN = 0.0
CA_MAX = 1.0

BRANCH_GUARD_T_MARGIN = 8.0
BRANCH_GUARD_CA_MARGIN = 0.03

for path in [SOURCE_PLANT_CHECKPOINT, SOURCE_CONTROLLER_CHECKPOINT, NORMALIZATION_STATS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

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

COLORS = {
    "train": "#1f77b4",
    "val": "#d62728",
    "true": "#111111",
    "pred": "#d62728",
    "input": "#1f77b4",
    "raw": "#8a8a8a",
    "axis": "#222222",
}

def polish_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])

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

TARGET_EOVER_R_NORM = (TARGET_EOVER_R - E_mean) / (E_std + 1e-8)

def norm_tc(Tc):
    return (Tc - Tc_mean) / (Tc_std + 1e-8)

def denorm_y(y_norm):
    y_norm = np.asarray(y_norm)
    return np.stack([
        y_norm[..., 0] * Ca_std + Ca_mean,
        y_norm[..., 1] * T_std + T_mean,
    ], axis=-1)

print("\nTarget EoverR:", TARGET_EOVER_R)
print("Target EoverR_norm:", TARGET_EOVER_R_NORM)

base_parms = {
    "q": 1.0,
    "V": 1.0,
    "rho": 1000.0,
    "Cp": 1.0,
    "deltaH": 2e5,
    "EoverR": TARGET_EOVER_R,
    "k0": 7.2e10,
    "UA": 1000.0,
    "Tf": 350.0,
    "CAf": 1.0,
}

def reaction_rate(T, parms):
    return parms["k0"] * np.exp(-parms["EoverR"] / T)

def cstr_rhs(t, x, Tc, parms):
    Ca, T = x

    Ca = np.clip(Ca, CA_MIN, CA_MAX)
    T = np.clip(T, T_MIN, T_MAX)
    Tc = np.clip(Tc, TC_MIN, TC_MAX)

    k = reaction_rate(T, parms)

    dCa = (parms["q"] / parms["V"]) * (parms["CAf"] - Ca) - k * Ca

    dT = (
        (parms["q"] / parms["V"]) * (parms["Tf"] - T)
        + (parms["deltaH"] / (parms["rho"] * parms["Cp"])) * k * Ca
        + (parms["UA"] / (parms["rho"] * parms["Cp"] * parms["V"])) * (Tc - T)
    )

    return np.array([dCa, dT])

def find_steady_states(parms):
    guesses = [
        [0.95, 355.0],
        [0.50, 400.0],
        [0.10, 440.0],
    ]

    steady_states = []

    for guess in guesses:
        ss = fsolve(
            lambda x: cstr_rhs(0.0, x, TC_NOMINAL, parms),
            guess,
            xtol=1e-9
        )

        ss[0] = np.clip(ss[0], CA_MIN, CA_MAX)
        ss[1] = np.clip(ss[1], T_MIN, T_MAX)

        if not any(np.allclose(ss, old, atol=1e-3) for old in steady_states):
            steady_states.append(ss)

    return sorted(steady_states, key=lambda z: z[1])

target_parms = base_parms.copy()
steady_states = find_steady_states(target_parms)

print("\nTarget steady states:")
for i, ss in enumerate(steady_states):
    print(f"SS{i + 1}: Ca={ss[0]:.4f}, T={ss[1]:.2f}")

target_low_ss = steady_states[0]
middle_ss = steady_states[1] if len(steady_states) >= 3 else None

print("\nUsing low-temperature branch:")
print(f"Ca={target_low_ss[0]:.4f}, T={target_low_ss[1]:.2f}")

def is_on_low_branch(x):
    Ca, T = x

    if middle_ss is not None:
        if T > middle_ss[1] - BRANCH_GUARD_T_MARGIN:
            return False
        if Ca < middle_ss[0] + BRANCH_GUARD_CA_MARGIN:
            return False

    return CA_MIN <= Ca <= CA_MAX and T_MIN <= T <= T_MAX

def generate_aprbs(n_steps):
    signal = np.zeros(n_steps)
    current_value = TC_NOMINAL
    hold_counter = 0

    for k in range(n_steps):
        if hold_counter <= 0:
            current_value = np.random.uniform(APRBS_TC_MIN, APRBS_TC_MAX)
            hold_counter = np.random.randint(APRBS_MIN_HOLD, APRBS_MAX_HOLD + 1)

        signal[k] = current_value
        hold_counter -= 1

    return signal

def simulate_target_profile(profile_id, Tc_signal):
    x = np.array(target_low_ss, dtype=float)

    Ca_hist = []
    T_hist = []

    for k in range(len(TIME) - 1):
        Tc = Tc_signal[k]

        sol = solve_ivp(
            lambda t, x_: cstr_rhs(t, x_, Tc, target_parms),
            [TIME[k], TIME[k + 1]],
            x,
            method="BDF",
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success:
            return None, False

        x = sol.y[:, -1]
        x[0] = np.clip(x[0], CA_MIN, CA_MAX)
        x[1] = np.clip(x[1], T_MIN, T_MAX)

        if not is_on_low_branch(x):
            return None, False

        Ca_hist.append(x[0] + np.random.normal(0.0, 0.001))
        T_hist.append(x[1] + np.random.normal(0.0, 0.5))

    df = pd.DataFrame({
        "Trajectory": profile_id,
        "Time": TIME[:-1],
        "Tc": Tc_signal[:-1],
        "Ca": Ca_hist,
        "T": T_hist,
        "EoverR": TARGET_EOVER_R,
        "EoverR_norm": TARGET_EOVER_R_NORM,
    })

    return df, True

def generate_target_dataset():
    accepted = []
    attempts = 0

    while len(accepted) < TARGET_PROFILES and attempts < MAX_TARGET_PROFILE_ATTEMPTS:
        attempts += 1

        Tc_signal = generate_aprbs(len(TIME))
        df, ok = simulate_target_profile(len(accepted), Tc_signal)

        if ok:
            accepted.append(df)
            print(f"Generated target profile {len(accepted)}/{TARGET_PROFILES}")

    if len(accepted) < TARGET_PROFILES:
        raise RuntimeError(
            f"Only accepted {len(accepted)}/{TARGET_PROFILES} target profiles. "
            "Relax the APRBS limits or branch guard."
        )

    target_df = pd.concat(accepted, ignore_index=True)

    target_df["Tc_norm"] = norm_tc(target_df["Tc"])
    target_df["Ca_norm"] = (target_df["Ca"] - Ca_mean) / (Ca_std + 1e-8)
    target_df["T_norm"] = (target_df["T"] - T_mean) / (T_std + 1e-8)

    return target_df

target_df = generate_target_dataset()

target_df.to_csv(BASE_DIR / "target_small_dataset_normalized.csv", index=False

def create_sequences(df):
    X = []
    Y = []

    for _, group in df.groupby("Trajectory"):
        group = group.reset_index(drop=True)

        u = group[["Tc_norm", "EoverR_norm"]].values.astype(np.float32)
        y = group[["Ca_norm", "T_norm"]].values.astype(np.float32)

        for i in range(0, len(group) - SEQUENCE_LENGTH, STRIDE):
            X.append(u[i:i + SEQUENCE_LENGTH])
            Y.append(y[i:i + SEQUENCE_LENGTH])

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

X_target, Y_target = create_sequences(target_df)

perm = np.random.permutation(len(X_target))
X_target = X_target[perm]
Y_target = Y_target[perm]

split_idx = int(0.8 * len(X_target))

X_target_train = X_target[:split_idx]
Y_target_train = Y_target[:split_idx]
X_target_val = X_target[split_idx:]
Y_target_val = Y_target[split_idx:]

print("\nTarget sequences:")
print("X_target_train:", X_target_train.shape)
print("Y_target_train:", Y_target_train.shape)
print("X_target_val:  ", X_target_val.shape)
print("Y_target_val:  ", Y_target_val.shape)

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
        return self.fc(out), h_n


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

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                self.num_layers,
                x.size(0),
                self.hidden_size,
                device=x.device
            )

        out, h_n = self.gru(x, h)
        raw_u = self.fc(out)
        u = self.u_mid + self.u_half_range * torch.tanh(raw_u)

        return u, h_n

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

criterion = nn.MSELoss()

plant_ckpt = torch.load(SOURCE_PLANT_CHECKPOINT, map_location=DEVICE, weights_only=False)

plant_model = CSTR_GRU(
    input_size=plant_ckpt.get("input_size", 2),
    hidden_size=plant_ckpt.get("hidden_size", 32),
    num_layers=plant_ckpt.get("num_layers", 3),
    output_size=plant_ckpt.get("output_size", 2),
).to(DEVICE)

plant_model.load_state_dict(plant_ckpt["model_state_dict"])

for name, param in plant_model.named_parameters():
    if "gru.weight_ih_l0" in name or "gru.weight_hh_l0" in name:
        param.requires_grad = False
    elif "gru.bias_ih_l0" in name or "gru.bias_hh_l0" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

plant_train_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_target_train),
        torch.FloatTensor(Y_target_train)
    ),
    batch_size=PLANT_BATCH_SIZE,
    shuffle=True
)

plant_val_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_target_val),
        torch.FloatTensor(Y_target_val)
    ),
    batch_size=PLANT_BATCH_SIZE,
    shuffle=False
)

plant_optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, plant_model.parameters()),
    lr=PLANT_LR,
    weight_decay=PLANT_WEIGHT_DECAY
)

plant_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    plant_optimizer,
    mode="min",
    factor=0.5,
    patience=8
)

best_plant_val_mse = float("inf")
best_plant_state = copy.deepcopy(plant_model.state_dict())
best_plant_epoch = 0
plant_patience_counter = 0

plant_train_mse_losses = []
plant_val_mse_losses = []

print("\nFine-tuning plant model...")

for epoch in range(PLANT_FINETUNE_EPOCHS):
    plant_model.train()

    train_mse_epoch = []

    for x_batch, y_batch in plant_train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        plant_optimizer.zero_grad()

        y_pred, _ = plant_model(x_batch)

        mse_loss = criterion(
            y_pred[:, PLANT_WASHOUT:, :],
            y_batch[:, PLANT_WASHOUT:, :]
        )

        iss_loss = gru_delta_iss_penalty(
            plant_model.gru,
            plant_model.num_layers
        )

        total_loss = mse_loss + PLANT_RHO * iss_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, plant_model.parameters()),
            max_norm=1.0
        )

        plant_optimizer.step()

        train_mse_epoch.append(mse_loss.item())

    train_mse = float(np.mean(train_mse_epoch))
    plant_train_mse_losses.append(train_mse)

    plant_model.eval()
    val_mse_epoch = []

    with torch.no_grad():
        for x_batch, y_batch in plant_val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_pred, _ = plant_model(x_batch)

            val_mse = criterion(
                y_pred[:, PLANT_WASHOUT:, :],
                y_batch[:, PLANT_WASHOUT:, :]
            )

            val_mse_epoch.append(val_mse.item())

    val_mse = float(np.mean(val_mse_epoch))
    plant_val_mse_losses.append(val_mse)

    plant_scheduler.step(val_mse)

    if val_mse < best_plant_val_mse:
        best_plant_val_mse = val_mse
        best_plant_state = copy.deepcopy(plant_model.state_dict())
        best_plant_epoch = epoch + 1
        plant_patience_counter = 0
    else:
        plant_patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Plant epoch [{epoch + 1}/{PLANT_FINETUNE_EPOCHS}] "
            f"| Train MSE: {train_mse:.6e} "
            f"| Val MSE: {val_mse:.6e} "
            f"| Best: {best_plant_val_mse:.6e} @ {best_plant_epoch}"
        )

    if plant_patience_counter >= PLANT_EARLY_STOP_PATIENCE:
        print(f"Plant early stopping at epoch {epoch + 1}")
        break

plant_model.load_state_dict(best_plant_state)
plant_model.eval()

torch.save(
    {
        **plant_ckpt,
        "model_state_dict": plant_model.state_dict(),
        "target_finetuned": True,
        "target_eoverr": TARGET_EOVER_R,
        "target_eoverr_norm": TARGET_EOVER_R_NORM,
        "target_profiles": TARGET_PROFILES,
        "best_target_val_mse": best_plant_val_mse,
        "best_target_epoch": best_plant_epoch,
        "plant_train_mse_losses_target": plant_train_mse_losses,
        "plant_val_mse_losses_target": plant_val_mse_losses,
    },
    FINETUNED_PLANT_SAVE_PATH
)

print(f"\nFine-tuned plant saved to: {FINETUNED_PLANT_SAVE_PATH}")

epochs = np.arange(1, len(plant_train_mse_losses) + 1)

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.scatter(epochs, plant_train_mse_losses, s=28, color=COLORS["train"], label="Training")
ax.scatter(epochs, plant_val_mse_losses, s=28, color=COLORS["val"], label="Validation")

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean squared error")
ax.set_title("Target Fine-Tuning: Plant Model")
polish_axes(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

fig.tight_layout()
save_figure(fig, "target_plant_finetuning_loss_evolution")

controller_ckpt = torch.load(SOURCE_CONTROLLER_CHECKPOINT, map_location=DEVICE, weights_only=False)
ctrl_cfg = controller_ckpt["controller_config"]

u_low = np.array(ctrl_cfg["u_low"]).reshape(-1)
u_high = np.array(ctrl_cfg["u_high"]).reshape(-1)

controller = ControllerGRU(
    input_size=ctrl_cfg["input_size"],
    hidden_size=ctrl_cfg["hidden_size"],
    num_layers=ctrl_cfg["num_layers"],
    output_size=ctrl_cfg["output_size"],
    u_low=u_low,
    u_high=u_high
).to(DEVICE)

controller.load_state_dict(controller_ckpt["controller_state_dict"])

for name, param in controller.named_parameters():
    if "gru.weight_ih_l0" in name or "gru.weight_hh_l0" in name:
        param.requires_grad = False
    elif "gru.bias_ih_l0" in name or "gru.bias_hh_l0" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

for param in plant_model.parameters():
    param.requires_grad = False

plant_model.eval()

def make_plant_input(u_seq):
    e_seq = torch.full(
        (u_seq.size(0), u_seq.size(1), 1),
        float(TARGET_EOVER_R_NORM),
        dtype=torch.float32,
        device=u_seq.device
    )

    return torch.cat([u_seq, e_seq], dim=2)

def simulate_model_equilibrium_for_constant_u(u_value):
    u_seq = torch.full(
        (1, N_SETTLING_STEPS, 1),
        float(u_value),
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        y_seq, _ = plant_model(make_plant_input(u_seq))

    y_np = y_seq.cpu().numpy()[0]
    return np.mean(y_np[-N_AVG_TAIL:], axis=0)

def filter_target_feasible_set(y_eq):
    if not USE_TARGET_SUPPORT_FILTER:
        return np.ones(len(y_eq), dtype=bool)

    target_y = target_df[["Ca_norm", "T_norm"]].values.astype(np.float32)
    target_sample = target_y[::TARGET_SUPPORT_STRIDE]

    distances = np.sqrt(
        np.sum(
            (y_eq[:, None, :] - target_sample[None, :, :]) ** 2,
            axis=2
        )
    )

    nearest_distance = np.min(distances, axis=1)
    mask = nearest_distance <= SUPPORT_DISTANCE_MAX

    if np.sum(mask) < MIN_FEASIBLE_POINTS:
        print("Warning: target feasible support filter too strict; using all feasible points.")
        mask = np.ones(len(y_eq), dtype=bool)

    return mask

def build_target_feasible_set():
    u_grid = np.linspace(float(u_low[0]), float(u_high[0]), N_FEASIBLE_GRID)
    y_eq = []

    print("\nBuilding target feasible set from fine-tuned NN plant...")

    for i, u in enumerate(u_grid):
        y_eq.append(simulate_model_equilibrium_for_constant_u(u))

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{N_FEASIBLE_GRID}")

    y_eq = np.asarray(y_eq, dtype=np.float32)
    mask = filter_target_feasible_set(y_eq)

    return u_grid[mask], y_eq[mask]

def first_order_reference_filter(setpoints):
    alpha = np.clip(DT / TAU_REF, 0.0, 1.0)

    y_ref = np.zeros_like(setpoints, dtype=np.float32)
    y_ref[0] = setpoints[0]

    for k in range(len(setpoints) - 1):
        y_ref[k + 1] = y_ref[k] + alpha * (setpoints[k] - y_ref[k])

    return y_ref

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

            hold = np.random.randint(REF_MIN_HOLD, REF_MAX_HOLD + 1)

        y0[k] = current
        hold -= 1

    return y0

def make_controller_input(Y_ref):
    e_feature = np.ones(
        (Y_ref.shape[0], Y_ref.shape[1], 1),
        dtype=np.float32
    ) * TARGET_EOVER_R_NORM

    return np.concatenate([Y_ref, e_feature], axis=2)

u_grid, target_feasible_y = build_target_feasible_set()

Y0_all = []
Y_ref_all = []

for _ in range(N_REF_TOTAL):
    for _attempt in range(MAX_REF_RESAMPLE_TRIES):
        y0 = generate_mprb_reference(target_feasible_y, REF_SEQ_LEN)
        y_ref = first_order_reference_filter(y0)

        y_ref_real = denorm_y(y_ref)
        ca_range = np.max(y_ref_real[:, 0]) - np.min(y_ref_real[:, 0])
        t_range = np.max(y_ref_real[:, 1]) - np.min(y_ref_real[:, 1])

        if ca_range >= MIN_SEQ_RANGE_CA or t_range >= MIN_SEQ_RANGE_T:
            break

    Y0_all.append(y0)
    Y_ref_all.append(y_ref)

Y0_all = np.asarray(Y0_all, dtype=np.float32)
Y_ref_all = np.asarray(Y_ref_all, dtype=np.float32)
C_input_all = make_controller_input(Y_ref_all)

C_train = C_input_all[:N_REF_TRAIN]
Y_train_ref = Y_ref_all[:N_REF_TRAIN]

C_val = C_input_all[N_REF_TRAIN:N_REF_TRAIN + N_REF_VAL]
Y_val_ref = Y_ref_all[N_REF_TRAIN:N_REF_TRAIN + N_REF_VAL]

plot_ref_id = 0
raw_ref_real = denorm_y(Y0_all[plot_ref_id])
filtered_ref_real = denorm_y(Y_ref_all[plot_ref_id])
t_ref = np.arange(REF_SEQ_LEN) * DT

fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)

axes[0].step(
    t_ref,
    raw_ref_real[:, 0],
    where="post",
    color=COLORS["raw"],
    linestyle=":",
    linewidth=1.8,
    label=r"Raw set-point $C_A$"
)
axes[0].plot(
    t_ref,
    filtered_ref_real[:, 0],
    color=COLORS["input"],
    linewidth=2.0,
    label=r"Filtered reference $C_A$"
)
axes[0].set_ylabel(r"$C_A$ [mol/L]")
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[0])

axes[1].step(
    t_ref,
    raw_ref_real[:, 1],
    where="post",
    color=COLORS["raw"],
    linestyle=":",
    linewidth=1.8,
    label=r"Raw set-point $T$"
)
axes[1].plot(
    t_ref,
    filtered_ref_real[:, 1],
    color=COLORS["pred"],
    linewidth=2.0,
    label=r"Filtered reference $T$"
)
axes[1].set_ylabel(r"$T$ [K]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
polish_axes(axes[1])

fig.suptitle("Target Feasible Reference Trajectory", y=0.98)
fig.tight_layout()
save_figure(fig, "target_finetuning_reference_trajectory")

ctrl_train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(C_train), torch.FloatTensor(Y_train_ref)),
    batch_size=CTRL_BATCH_SIZE,
    shuffle=True
)

ctrl_val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(C_val), torch.FloatTensor(Y_val_ref)),
    batch_size=CTRL_BATCH_SIZE,
    shuffle=False
)

ctrl_optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, controller.parameters()),
    lr=CTRL_LR,
    weight_decay=CTRL_WEIGHT_DECAY
)

ctrl_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    ctrl_optimizer,
    mode="min",
    factor=0.5,
    patience=8
)

def overshoot_penalty(y_pred, y_ref):
    error = y_pred[:, CTRL_WASHOUT:, :] - y_ref[:, CTRL_WASHOUT:, :]
    direction = y_ref[:, CTRL_WASHOUT:, :] - y_ref[:, CTRL_WASHOUT:CTRL_WASHOUT + 1, :]
    signed_overshoot = error * torch.sign(direction + 1e-8)
    return torch.mean(torch.relu(signed_overshoot - OVERSHOOT_DEADBAND) ** 2)

def controller_loss(controller_input, ref_batch, epoch):
    u_seq, _ = controller(controller_input)
    y_pred, _ = plant_model(make_plant_input(u_seq))

    tracking_loss = criterion(
        y_pred[:, CTRL_WASHOUT:, :],
        ref_batch[:, CTRL_WASHOUT:, :]
    )

    du = u_seq[:, 1:, :] - u_seq[:, :-1, :]
    smooth_loss = torch.mean(du ** 2)
    mag_loss = torch.mean(u_seq ** 2)
    os_loss = overshoot_penalty(y_pred, ref_batch)

    iss_loss = gru_delta_iss_penalty(controller.gru, controller.num_layers)
    iss_weight = CONTROL_ISS_WEIGHT * min(1.0, epoch / 50)

    total_loss = (
        tracking_loss
        + CONTROL_SMOOTH_WEIGHT * smooth_loss
        + CONTROL_MAG_WEIGHT * mag_loss
        + CONTROL_OVERSHOOT_WEIGHT * os_loss
        + iss_weight * iss_loss
    )

    return total_loss, tracking_loss

best_ctrl_val_tracking = float("inf")
best_ctrl_state = copy.deepcopy(controller.state_dict())
best_ctrl_epoch = 0
ctrl_patience_counter = 0

ctrl_train_tracking_losses = []
ctrl_val_tracking_losses = []

print("\nFine-tuning controller through fine-tuned target plant...")

for epoch in range(CTRL_FINETUNE_EPOCHS):
    controller.train()

    train_tracking_epoch = []

    for c_batch, r_batch in ctrl_train_loader:
        c_batch = c_batch.to(DEVICE)
        r_batch = r_batch.to(DEVICE)

        ctrl_optimizer.zero_grad()

        total_loss, tracking_loss = controller_loss(c_batch, r_batch, epoch)

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, controller.parameters()),
            max_norm=1.0
        )

        ctrl_optimizer.step()

        train_tracking_epoch.append(tracking_loss.item())

    train_tracking = float(np.mean(train_tracking_epoch))
    ctrl_train_tracking_losses.append(train_tracking)

    controller.eval()

    val_tracking_epoch = []

    with torch.no_grad():
        for c_batch, r_batch in ctrl_val_loader:
            c_batch = c_batch.to(DEVICE)
            r_batch = r_batch.to(DEVICE)

            _, tracking_loss = controller_loss(c_batch, r_batch, epoch)
            val_tracking_epoch.append(tracking_loss.item())

    val_tracking = float(np.mean(val_tracking_epoch))
    ctrl_val_tracking_losses.append(val_tracking)

    ctrl_scheduler.step(val_tracking)

    if val_tracking < best_ctrl_val_tracking:
        best_ctrl_val_tracking = val_tracking
        best_ctrl_state = copy.deepcopy(controller.state_dict())
        best_ctrl_epoch = epoch + 1
        ctrl_patience_counter = 0
    else:
        ctrl_patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Controller epoch [{epoch + 1}/{CTRL_FINETUNE_EPOCHS}] "
            f"| Train tracking: {train_tracking:.6e} "
            f"| Val tracking: {val_tracking:.6e} "
            f"| Best: {best_ctrl_val_tracking:.6e} @ {best_ctrl_epoch}"
        )

    if ctrl_patience_counter >= CTRL_EARLY_STOP_PATIENCE:
        print(f"Controller early stopping at epoch {epoch + 1}")
        break

controller.load_state_dict(best_ctrl_state)
controller.eval()

torch.save(
    {
        **controller_ckpt,
        "controller_state_dict": controller.state_dict(),
        "target_finetuned": True,
        "target_eoverr": TARGET_EOVER_R,
        "target_eoverr_norm": TARGET_EOVER_R_NORM,
        "best_target_val_tracking_loss": best_ctrl_val_tracking,
        "best_target_epoch": best_ctrl_epoch,
        "controller_train_tracking_losses_target": ctrl_train_tracking_losses,
        "controller_val_tracking_losses_target": ctrl_val_tracking_losses,
        "target_feasible_u_grid": u_grid.tolist(),
        "target_feasible_outputs": target_feasible_y.tolist(),
    },
    FINETUNED_CONTROLLER_SAVE_PATH
)

print(f"\nFine-tuned controller saved to: {FINETUNED_CONTROLLER_SAVE_PATH}")

epochs = np.arange(1, len(ctrl_train_tracking_losses) + 1)

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.scatter(epochs, ctrl_train_tracking_losses, s=28, color=COLORS["train"], label="Training")
ax.scatter(epochs, ctrl_val_tracking_losses, s=28, color=COLORS["val"], label="Validation")

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Tracking loss")
ax.set_title("Target Fine-Tuning: Controller")
polish_axes(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

fig.tight_layout()
save_figure(fig, "target_controller_finetuning_loss_evolution")

print("\nDone.")
print("Generated:")
print("- cstr_gru_deltaISS_target_finetuned.pth")
print("- cstr_gru_controller_target_finetuned.pth")
print("- target_plant_finetuning_loss_evolution.png / .pdf")
print("- target_controller_finetuning_loss_evolution.png / .pdf")
print("- target_finetuning_reference_trajectory.png / .pdf")
