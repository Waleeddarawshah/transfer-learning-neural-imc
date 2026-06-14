from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import control as ct

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

SOURCE_PLANT_CHECKPOINT = BASE_DIR / "cstr_gru_deltaISS.pth"
SOURCE_CONTROLLER_CHECKPOINT = BASE_DIR / "cstr_gru_controller_deltaISS_transfer.pth"

FINETUNED_PLANT_CHECKPOINT = BASE_DIR / "cstr_gru_deltaISS_target_finetuned.pth"
FINETUNED_CONTROLLER_CHECKPOINT = BASE_DIR / "cstr_gru_controller_target_finetuned.pth"

NORMALIZATION_STATS_PATH = BASE_DIR / "normalization_stats.csv"
if not NORMALIZATION_STATS_PATH.exists():
    NORMALIZATION_STATS_PATH = BASE_DIR / "Normalization_Stats_Multi_EoverR.csv"

Y_TRAIN_PATH = BASE_DIR / "Y_train.npy"

TARGET_EOVER_R = 1.005e4

DT = 1.0
N_STEPS = 8000

TAU_REF = 16.0
MIN_HOLD = 80
MAX_HOLD = 300

TC_MIN = 280.0
TC_MAX = 400.0
TC_NOMINAL = 350.0

IMC_LAMBDA = 80.0
IMC_FILTER_ORDER = 3

NEURAL_U_SMOOTHING = 0.12
MAX_TC_RATE = 1.0

USE_TRUE_IMC_CORRECTION = True
IMC_ERROR_FILTER_ALPHA = 0.85
IMC_ERROR_CLIP_CA = 0.35
IMC_ERROR_CLIP_T = 0.35

N_FEASIBLE_GRID = 400
N_SETTLING_STEPS = 1200
N_AVG_TAIL = 100

SUPPORT_DISTANCE_MAX = 0.20
OBSERVED_SUPPORT_STRIDE = 80
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

OVERSHOOT_GUARD_ENABLED = True
T_OVERSHOOT_BAND = 4.0

for path in [
    SOURCE_PLANT_CHECKPOINT,
    SOURCE_CONTROLLER_CHECKPOINT,
    FINETUNED_PLANT_CHECKPOINT,
    FINETUNED_CONTROLLER_CHECKPOINT,
    NORMALIZATION_STATS_PATH,
    Y_TRAIN_PATH,
]:
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
    "reference": "#111111",
    "source": "#1f77b4",
    "fine": "#d62728",
    "pid": "#2ca02c",
    "raw": "#8a8a8a",
    "axis": "#222222",
    "bar_source": "#34495e",
    "bar_fine": "#e74c3c",
    "bar_pid": "#95a5a6",
}

def polish_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])

def save_figure(fig, filename):
    fig.savefig(BASE_DIR / filename, bbox_inches="tight")
    plt.show()
    plt.close(fig)

BASE_PARMS = {
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

    Ca = np.clip(Ca, 0.0, 1.2)
    T = np.clip(T, 250.0, 550.0)
    Tc = np.clip(Tc, TC_MIN, TC_MAX)

    k = reaction_rate(T, parms)

    dCa = (parms["q"] / parms["V"]) * (parms["CAf"] - Ca) - k * Ca

    dT = (
        (parms["q"] / parms["V"]) * (parms["Tf"] - T)
        + (parms["deltaH"] / (parms["rho"] * parms["Cp"])) * k * Ca
        + (parms["UA"] / (parms["rho"] * parms["Cp"] * parms["V"])) * (Tc - T)
    )

    return np.array([dCa, dT])

def cstr_step(x, Tc, parms, dt=DT):
    sol = solve_ivp(
        lambda t, x_: cstr_rhs(t, x_, Tc, parms),
        [0.0, dt],
        x,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
    )

    x_next = sol.y[:, -1]
    x_next[0] = np.clip(x_next[0], 0.0, 1.2)
    x_next[1] = np.clip(x_next[1], 250.0, 550.0)

    return x_next

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

def norm_u(Tc):
    return (Tc - Tc_mean) / (Tc_std + 1e-8)

def denorm_u(Tc_norm):
    return np.asarray(Tc_norm) * Tc_std + Tc_mean

def norm_y(y):
    y = np.asarray(y)
    return np.array([
        (y[0] - Ca_mean) / (Ca_std + 1e-8),
        (y[1] - T_mean) / (T_std + 1e-8),
    ])

def denorm_y(y_norm):
    y_norm = np.asarray(y_norm)
    y_real = np.empty_like(y_norm, dtype=float)
    y_real[..., 0] = y_norm[..., 0] * Ca_std + Ca_mean
    y_real[..., 1] = y_norm[..., 1] * T_std + T_mean
    return y_real

print("Target EoverR:", TARGET_EOVER_R)
print("Target EoverR_norm:", TARGET_EOVER_R_NORM)

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

def load_plant_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    model = CSTR_GRU(
        input_size=ckpt.get("input_size", ckpt.get("plant_config", {}).get("input_size", 2)),
        hidden_size=ckpt.get("hidden_size", ckpt.get("plant_config", {}).get("hidden_size", 32)),
        num_layers=ckpt.get("num_layers", ckpt.get("plant_config", {}).get("num_layers", 3)),
        output_size=ckpt.get("output_size", ckpt.get("plant_config", {}).get("output_size", 2)),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, ckpt

def load_controller(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["controller_config"]

    u_low = np.array(cfg["u_low"]).reshape(-1)
    u_high = np.array(cfg["u_high"]).reshape(-1)

    controller = ControllerGRU(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        output_size=cfg["output_size"],
        u_low=u_low,
        u_high=u_high
    ).to(DEVICE)

    controller.load_state_dict(ckpt["controller_state_dict"])
    controller.eval()

    return controller, ckpt, u_low, u_high

source_plant_model, source_plant_ckpt = load_plant_model(SOURCE_PLANT_CHECKPOINT)
finetuned_plant_model, finetuned_plant_ckpt = load_plant_model(FINETUNED_PLANT_CHECKPOINT)

source_controller, source_controller_ckpt, u_low, u_high = load_controller(SOURCE_CONTROLLER_CHECKPOINT)
finetuned_controller, finetuned_controller_ckpt, _, _ = load_controller(FINETUNED_CONTROLLER_CHECKPOINT)

print("\nLoaded:")
print("- source NN plant")
print("- fine-tuned NN plant")
print("- source controller")
print("- fine-tuned controller")

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

        ss[0] = np.clip(ss[0], 0.0, 1.2)
        ss[1] = np.clip(ss[1], 250.0, 550.0)

        if not any(np.allclose(ss, old, atol=1e-3) for old in steady_states):
            steady_states.append(ss)

    return sorted(steady_states, key=lambda z: z[1])

target_parms = BASE_PARMS.copy()
steady_states = find_steady_states(target_parms)

print("\nTarget plant steady states:")
for i, ss in enumerate(steady_states):
    print(f"SS{i + 1}: Ca={ss[0]:.4f}, T={ss[1]:.2f}")

target_low_ss = steady_states[0].copy()

Y_train_observed = np.load(Y_TRAIN_PATH).reshape(-1, 2)

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

def plant_model_equilibrium(model, u_norm):
    u_seq = torch.full(
        (1, N_SETTLING_STEPS, 1),
        float(u_norm),
        dtype=torch.float32,
        device=DEVICE
    )

    e_seq = torch.full(
        (1, N_SETTLING_STEPS, 1),
        float(TARGET_EOVER_R_NORM),
        dtype=torch.float32,
        device=DEVICE
    )

    plant_input = torch.cat([u_seq, e_seq], dim=2)

    with torch.no_grad():
        y_seq, _ = model(plant_input)

    y_np = y_seq.cpu().numpy()[0]

    return np.mean(y_np[-N_AVG_TAIL:], axis=0)

def filter_feasible_set_to_observed_branch(feasible_y):
    lower = np.percentile(Y_train_observed, BRANCH_SUPPORT_LOWER_PCT, axis=0)
    upper = np.percentile(Y_train_observed, BRANCH_SUPPORT_UPPER_PCT, axis=0)

    ca_margin_norm = CA_BRANCH_MARGIN / (Ca_std + 1e-8)
    t_margin_norm = T_BRANCH_MARGIN / (T_std + 1e-8)

    lower = lower - np.array([ca_margin_norm, t_margin_norm])
    upper = upper + np.array([ca_margin_norm, t_margin_norm])

    percentile_mask = (
        (feasible_y[:, 0] >= lower[0])
        & (feasible_y[:, 0] <= upper[0])
        & (feasible_y[:, 1] >= lower[1])
        & (feasible_y[:, 1] <= upper[1])
    )

    observed_sample = Y_train_observed[::OBSERVED_SUPPORT_STRIDE]

    distances = np.sqrt(
        np.sum(
            (feasible_y[:, None, :] - observed_sample[None, :, :]) ** 2,
            axis=2
        )
    )

    nearest_distance = np.min(distances, axis=1)

    support_mask = percentile_mask & (nearest_distance <= SUPPORT_DISTANCE_MAX)
    support_mask = longest_true_segment(support_mask)

    if np.sum(support_mask) < MIN_FEASIBLE_POINTS:
        print("Warning: feasible branch filter too strict; using percentile branch support.")
        support_mask = percentile_mask

    if np.sum(support_mask) < MIN_FEASIBLE_POINTS:
        print("Warning: percentile branch support too strict; using all model feasible points.")
        support_mask = np.ones(len(feasible_y), dtype=bool)

    return support_mask

def build_target_feasible_set(model):
    u_grid = np.linspace(float(u_low[0]), float(u_high[0]), N_FEASIBLE_GRID)

    y_eq = []

    print("\nBuilding target feasible reference set from fine-tuned NN plant...")

    for i, u in enumerate(u_grid):
        y_eq.append(plant_model_equilibrium(model, u))

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{N_FEASIBLE_GRID}")

    y_eq = np.asarray(y_eq, dtype=np.float32)
    mask = filter_feasible_set_to_observed_branch(y_eq)

    return u_grid[mask], y_eq[mask]

def first_order_reference_filter(raw_setpoints):
    alpha = np.clip(DT / TAU_REF, 0.0, 1.0)

    yref = np.zeros_like(raw_setpoints, dtype=np.float32)
    yref[0] = raw_setpoints[0]

    for k in range(len(raw_setpoints) - 1):
        yref[k + 1] = yref[k] + alpha * (raw_setpoints[k] - yref[k])

    return yref

def generate_mprb_reference(feasible_y, n_steps):
    feasible_y = np.asarray(feasible_y, dtype=np.float32)
    n_points = len(feasible_y)

    y0 = np.zeros((n_steps, 2), dtype=np.float32)

    idx = np.random.randint(0, n_points)
    current = feasible_y[idx]
    hold = 0

    for k in range(n_steps):
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

u_grid_ref, feasible_y_target = build_target_feasible_set(finetuned_plant_model)

for _attempt in range(MAX_REF_RESAMPLE_TRIES):
    y0_norm = generate_mprb_reference(feasible_y_target, N_STEPS)
    yref_norm = first_order_reference_filter(y0_norm)

    yref_real_candidate = denorm_y(yref_norm)

    ca_range = np.max(yref_real_candidate[:, 0]) - np.min(yref_real_candidate[:, 0])
    t_range = np.max(yref_real_candidate[:, 1]) - np.min(yref_real_candidate[:, 1])

    if ca_range >= MIN_SEQ_RANGE_CA or t_range >= MIN_SEQ_RANGE_T:
        break

y0_real = denorm_y(y0_norm)
yref_real = denorm_y(yref_norm)

time = np.arange(N_STEPS) * DT

x0 = yref_real[0].copy()

print("\nInitial target state set from first filtered reference:")
print(f"Ca0={x0[0]:.4f}, T0={x0[1]:.2f}")

def simulate_neural_controller(controller, plant_model, x_initial, yref_norm):
    x = x_initial.copy()

    x_hist = []
    u_hist = []

    h_c = None
    h_p = None

    u_prev = TC_NOMINAL

    model_y_prev_norm = norm_y(x)
    error_filter = np.zeros(2, dtype=float)

    for k in range(len(yref_norm)):
        measured_y_norm = norm_y(x)

        if USE_TRUE_IMC_CORRECTION:
            model_error = measured_y_norm - model_y_prev_norm

            model_error = np.clip(
                model_error,
                [-IMC_ERROR_CLIP_CA, -IMC_ERROR_CLIP_T],
                [IMC_ERROR_CLIP_CA, IMC_ERROR_CLIP_T],
            )

            error_filter = (
                IMC_ERROR_FILTER_ALPHA * error_filter
                + (1.0 - IMC_ERROR_FILTER_ALPHA) * model_error
            )

            corrected_ref_norm = yref_norm[k] - error_filter
        else:
            corrected_ref_norm = yref_norm[k].copy()

        controller_input = np.array([
            corrected_ref_norm[0],
            corrected_ref_norm[1],
            TARGET_EOVER_R_NORM,
        ], dtype=np.float32).reshape(1, 1, 3)

        controller_input = torch.FloatTensor(controller_input).to(DEVICE)

        with torch.no_grad():
            u_norm_tensor, h_c = controller(controller_input, h_c)

        u_norm = float(u_norm_tensor.cpu().numpy()[0, 0, 0])
        Tc_cmd = float(denorm_u(u_norm))

        Tc_cmd = (
            NEURAL_U_SMOOTHING * Tc_cmd
            + (1.0 - NEURAL_U_SMOOTHING) * u_prev
        )

        if OVERSHOOT_GUARD_ENABLED:
            T_ref = yref_real[k, 1]
            if x[1] > T_ref + T_OVERSHOOT_BAND:
                Tc_cmd = min(Tc_cmd, u_prev)

        delta_tc = np.clip(
            Tc_cmd - u_prev,
            -MAX_TC_RATE,
            MAX_TC_RATE
        )

        Tc_cmd = u_prev + delta_tc
        Tc_cmd = np.clip(Tc_cmd, TC_MIN, TC_MAX)

        u_actual_norm = norm_u(Tc_cmd)

        plant_input = torch.FloatTensor([[[u_actual_norm, TARGET_EOVER_R_NORM]]]).to(DEVICE)

        with torch.no_grad():
            model_y_tensor, h_p = plant_model(plant_input, h_p)

        model_y_prev_norm = model_y_tensor.cpu().numpy()[0, 0, :]

        x = cstr_step(x, Tc_cmd, target_parms)

        x_hist.append(x.copy())
        u_hist.append(Tc_cmd)

        u_prev = Tc_cmd

    return np.asarray(x_hist), np.asarray(u_hist)

y_source_nn, u_source_nn = simulate_neural_controller(
    source_controller,
    source_plant_model,
    x0,
    yref_norm
)

y_finetuned_nn, u_finetuned_nn = simulate_neural_controller(
    finetuned_controller,
    finetuned_plant_model,
    x0,
    yref_norm
)

def linearize_cstr(parms, x_ss, u_ss):
    nx = 2
    nu = 1

    eps_x = np.array([1e-5, 1e-3])
    eps_u = 1e-3

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps_x[i]

        f_plus = cstr_rhs(0.0, x_ss + dx, u_ss, parms)
        f_minus = cstr_rhs(0.0, x_ss - dx, u_ss, parms)

        A[:, i] = (f_plus - f_minus) / (2 * eps_x[i])

    f_plus = cstr_rhs(0.0, x_ss, u_ss + eps_u, parms)
    f_minus = cstr_rhs(0.0, x_ss, u_ss - eps_u, parms)

    B[:, 0] = (f_plus - f_minus) / (2 * eps_u)

    C = np.eye(2)
    D = np.zeros((2, 1))

    return ct.ss(A, B, C, D)

def design_imc_temperature_controller(parms, x_ss, u_ss):
    lin_sys = linearize_cstr(parms, x_ss, u_ss)
    Gp_T = ct.ss2tf(lin_sys[1, 0])

    s = ct.tf([1, 0], [1])

    for order in [IMC_FILTER_ORDER, 4, 5, 6]:
        try:
            F = 1 / (IMC_LAMBDA * s + 1) ** order
            Q = ct.minreal(F / Gp_T, verbose=False)
            C_imc = ct.minreal(Q / (1 - Q * Gp_T), verbose=False)

            C_ss = ct.ss(C_imc)
            C_d = ct.sample_system(C_ss, DT, method="tustin")

            print("\nTraditional IMC controller designed:")
            print("Filter order:", order)
            print("Gp_T:", Gp_T)

            return C_d

        except Exception as err:
            print(f"IMC design failed for filter order {order}: {err}")

    raise RuntimeError("Could not design a proper IMC controller.")

def simulate_imc_pid(x_initial, yref_real):
    x = x_initial.copy()

    x_ss = x_initial.copy()
    u_ss = TC_NOMINAL

    C_d = design_imc_temperature_controller(
        target_parms,
        x_ss,
        u_ss
    )

    A = np.asarray(C_d.A)
    B = np.asarray(C_d.B)
    C = np.asarray(C_d.C)
    D = np.asarray(C_d.D)

    x_c = np.zeros((A.shape[0], 1))

    x_hist = []
    u_hist = []

    for k in range(len(yref_real)):
        T_ref = yref_real[k, 1]
        T_meas = x[1]

        e = np.array([[T_ref - T_meas]])

        du = float((C @ x_c + D @ e).item())
        x_c = A @ x_c + B @ e

        Tc_cmd = u_ss + du
        Tc_cmd = np.clip(Tc_cmd, TC_MIN, TC_MAX)

        x = cstr_step(x, Tc_cmd, target_parms)

        x_hist.append(x.copy())
        u_hist.append(Tc_cmd)

    return np.asarray(x_hist), np.asarray(u_hist)

y_pid, u_pid = simulate_imc_pid(x0, yref_real)

rmse_source = np.sqrt(np.mean((yref_real - y_source_nn) ** 2, axis=0))
rmse_finetuned = np.sqrt(np.mean((yref_real - y_finetuned_nn) ** 2, axis=0))
rmse_pid = np.sqrt(np.mean((yref_real - y_pid) ** 2, axis=0))

ss_indices = []
current_ref = y0_real[0]

for k in range(1, len(y0_real)):
    if not np.allclose(y0_real[k], current_ref):
        ss_indices.append(k - 1)
        current_ref = y0_real[k]

ss_indices.append(len(y0_real) - 1)

def compute_ss_metrics(y_true, y_pred, ss_idx, tail=15):
    ss_errors = []

    for idx in ss_idx:
        true_ss = np.mean(y_true[max(0, idx - tail):idx + 1], axis=0)
        pred_ss = np.mean(y_pred[max(0, idx - tail):idx + 1], axis=0)

        ss_errors.append(np.abs(true_ss - pred_ss))

    ss_errors = np.asarray(ss_errors)

    return np.mean(ss_errors, axis=0), np.max(ss_errors, axis=0)

def compute_overshoot_metrics(y_response, y_setpoint, ss_idx):
    overshoots = []
    start = 0

    for end in ss_idx:
        if end <= start + 1:
            start = end
            continue

        y_initial = y_setpoint[start]
        y_final = y_setpoint[end]
        segment = y_response[start:end + 1]

        direction = np.sign(y_final - y_initial)
        overshoot = np.zeros(2)

        for j in range(2):
            if direction[j] > 0:
                overshoot[j] = max(0.0, np.max(segment[:, j] - y_final[j]))
            elif direction[j] < 0:
                overshoot[j] = max(0.0, np.max(y_final[j] - segment[:, j]))
            else:
                overshoot[j] = max(0.0, np.max(np.abs(segment[:, j] - y_final[j])))

        overshoots.append(overshoot)
        start = end + 1

    overshoots = np.asarray(overshoots)

    if len(overshoots) == 0:
        return np.zeros(2), np.zeros(2)

    return np.mean(overshoots, axis=0), np.max(overshoots, axis=0)

mean_ss_source, max_ss_source = compute_ss_metrics(yref_real, y_source_nn, ss_indices)
mean_ss_fine, max_ss_fine = compute_ss_metrics(yref_real, y_finetuned_nn, ss_indices)
mean_ss_pid, max_ss_pid = compute_ss_metrics(yref_real, y_pid, ss_indices)

mean_os_source, max_os_source = compute_overshoot_metrics(y_source_nn, y0_real, ss_indices)
mean_os_fine, max_os_fine = compute_overshoot_metrics(y_finetuned_nn, y0_real, ss_indices)
mean_os_pid, max_os_pid = compute_overshoot_metrics(y_pid, y0_real, ss_indices)

results = pd.DataFrame({
    "Controller": [
        "Source IMC-NN",
        "Fine-tuned IMC-NN",
        "Traditional IMC-PID"
    ],
    "Target_EoverR": [TARGET_EOVER_R] * 3,
    "Ca_RMSE": [rmse_source[0], rmse_finetuned[0], rmse_pid[0]],
    "T_RMSE": [rmse_source[1], rmse_finetuned[1], rmse_pid[1]],
    "Ca_Mean_SS_Error": [mean_ss_source[0], mean_ss_fine[0], mean_ss_pid[0]],
    "T_Mean_SS_Error": [mean_ss_source[1], mean_ss_fine[1], mean_ss_pid[1]],
    "Ca_Max_SS_Error": [max_ss_source[0], max_ss_fine[0], max_ss_pid[0]],
    "T_Max_SS_Error": [max_ss_source[1], max_ss_fine[1], max_ss_pid[1]],
    "Ca_Mean_Overshoot": [mean_os_source[0], mean_os_fine[0], mean_os_pid[0]],
    "T_Mean_Overshoot": [mean_os_source[1], mean_os_fine[1], mean_os_pid[1]],
    "Ca_Max_Overshoot": [max_os_source[0], max_os_fine[0], max_os_pid[0]],
    "T_Max_Overshoot": [max_os_source[1], max_os_fine[1], max_os_pid[1]],
})

results.to_csv(BASE_DIR / "closed_loop_summary_table.csv", index=False)

print("\nClosed-loop summary table:")
print(results.to_string(index=False))

err_source = yref_real - y_source_nn
err_finetuned = yref_real - y_finetuned_nn
err_pid = yref_real - y_pid

fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)

axes[0].step(
    time,
    y0_real[:, 0],
    where="post",
    color=COLORS["raw"],
    linestyle=":",
    linewidth=1.8,
    label=r"Raw set-point $C_A$"
)
axes[0].plot(
    time,
    yref_real[:, 0],
    color=COLORS["source"],
    linewidth=2.0,
    label=r"Filtered reference $C_A$"
)
axes[0].set_ylabel(r"$C_A$ [mol/L]")
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[0])

axes[1].step(
    time,
    y0_real[:, 1],
    where="post",
    color=COLORS["raw"],
    linestyle=":",
    linewidth=1.8,
    label=r"Raw set-point $T$"
)
axes[1].plot(
    time,
    yref_real[:, 1],
    color=COLORS["fine"],
    linewidth=2.0,
    label=r"Filtered reference $T$"
)
axes[1].set_ylabel(r"$T$ [K]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[1])

fig.suptitle("Target Feasible Reference Trajectory", y=0.98)
fig.tight_layout()
save_figure(fig, "target_reference.png")

fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)

axes[0].plot(time, yref_real[:, 0], color=COLORS["reference"], linestyle="--", linewidth=1.7, label="Reference")
axes[0].plot(time, y_source_nn[:, 0], color=COLORS["source"], linewidth=1.8, label="Source IMC-NN")
axes[0].plot(time, y_finetuned_nn[:, 0], color=COLORS["fine"], linewidth=2.0, label="Fine-tuned IMC-NN")
axes[0].plot(time, y_pid[:, 0], color=COLORS["pid"], linestyle="-.", linewidth=1.6, label="Traditional IMC-PID")
axes[0].set_ylabel(r"$C_A$ [mol/L]")
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[0])

axes[1].plot(time, yref_real[:, 1], color=COLORS["reference"], linestyle="--", linewidth=1.7, label="Reference")
axes[1].plot(time, y_source_nn[:, 1], color=COLORS["source"], linewidth=1.8, label="Source IMC-NN")
axes[1].plot(time, y_finetuned_nn[:, 1], color=COLORS["fine"], linewidth=2.0, label="Fine-tuned IMC-NN")
axes[1].plot(time, y_pid[:, 1], color=COLORS["pid"], linestyle="-.", linewidth=1.6, label="Traditional IMC-PID")
axes[1].set_ylabel(r"$T$ [K]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[1])

fig.suptitle("Closed-Loop Target Tracking", y=0.98)
fig.tight_layout()
save_figure(fig, "target_closed_loop_tracking.png")

fig, ax = plt.subplots(figsize=(9.0, 4.2))

ax.plot(time, u_source_nn, color=COLORS["source"], linewidth=1.8, label="Source IMC-NN")
ax.plot(time, u_finetuned_nn, color=COLORS["fine"], linewidth=2.0, label="Fine-tuned IMC-NN")
ax.plot(time, u_pid, color=COLORS["pid"], linestyle="-.", linewidth=1.6, label="Traditional IMC-PID")

ax.axhline(TC_MIN, color=COLORS["raw"], linestyle=":", linewidth=1.4, label="Input limits")
ax.axhline(TC_MAX, color=COLORS["raw"], linestyle=":", linewidth=1.4)

ax.set_ylabel(r"$T_c$ [K]")
ax.set_xlabel("Time [s]")
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(ax)

fig.suptitle("Closed-Loop Control Inputs", y=0.98)
fig.tight_layout()
save_figure(fig, "target_control_inputs.png")

fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0))

axes[0].axhline(0.0, color="#444444", linewidth=1.0)
axes[0].plot(time, err_source[:, 0], color=COLORS["source"], linewidth=1.5, label="Source IMC-NN")
axes[0].plot(time, err_finetuned[:, 0], color=COLORS["fine"], linewidth=1.7, label="Fine-tuned IMC-NN")
axes[0].plot(time, err_pid[:, 0], color=COLORS["pid"], linestyle="-.", linewidth=1.5, label="Traditional IMC-PID")
axes[0].set_ylabel(r"$C_A$ error")
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[0])

axes[1].axhline(0.0, color="#444444", linewidth=1.0)
axes[1].plot(time, err_source[:, 1], color=COLORS["source"], linewidth=1.5, label="Source IMC-NN")
axes[1].plot(time, err_finetuned[:, 1], color=COLORS["fine"], linewidth=1.7, label="Fine-tuned IMC-NN")
axes[1].plot(time, err_pid[:, 1], color=COLORS["pid"], linestyle="-.", linewidth=1.5, label="Traditional IMC-PID")
axes[1].set_ylabel(r"$T$ error [K]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
polish_axes(axes[1])

labels = ["Source IMC-NN", "Fine-tuned IMC-NN", "Traditional IMC-PID"]
rmse_values = [rmse_source[1], rmse_finetuned[1], rmse_pid[1]]
bar_colors = [COLORS["bar_source"], COLORS["bar_fine"], COLORS["bar_pid"]]

x_bars = np.arange(len(labels))

bars = axes[2].bar(
    x_bars,
    rmse_values,
    width=0.46,
    color=bar_colors,
    edgecolor="none"
)

axes[2].set_ylabel(r"Temperature RMSE [K]")
axes[2].set_xticks(x_bars)
axes[2].set_xticklabels(labels)

for bar in bars:
    height = bar.get_height()
    axes[2].annotate(
        f"{height:.2f} K",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

polish_axes(axes[2])

fig.suptitle("Closed-Loop Tracking Errors and RMSE", y=0.98)
fig.tight_layout()
save_figure(fig, "target_errors_rmse.png")

print("\nDone.")
print("Generated:")
print("- closed_loop_summary_table.csv")
print("- target_reference.png")
print("- target_closed_loop_tracking.png")
print("- target_control_inputs.png")
print("- target_errors_rmse.png")
