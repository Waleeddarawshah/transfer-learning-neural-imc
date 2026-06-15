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

X_TRAIN_PATH = BASE_DIR / "X_train.npy"
Y_TRAIN_PATH = BASE_DIR / "Y_train.npy"
X_VAL_PATH = BASE_DIR / "X_val.npy"
Y_VAL_PATH = BASE_DIR / "Y_val.npy"
X_TEST_PATH = BASE_DIR / "X_test.npy"
Y_TEST_PATH = BASE_DIR / "Y_test.npy"

NORMALIZATION_STATS_PATH = BASE_DIR / "normalization_stats.csv"
if not NORMALIZATION_STATS_PATH.exists():
    NORMALIZATION_STATS_PATH = BASE_DIR / "Normalization_Stats_Multi_EoverR.csv"

MODEL_SAVE_PATH = OUTPUT_DIR / "cstr_gru_deltaISS.pth"

INPUT_SIZE = 2
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 2

BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 3e-5

WASHOUT = 100
RHO = 2e-4
ISS_RAMP_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 100
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
    "true": "#111111",
    "pred": "#d62728",
    "input": "#2c3e50",
    "param": "#7b3294",
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


def denorm_u(u_norm):
    return np.asarray(u_norm) * Tc_std + Tc_mean


def denorm_e(e_norm):
    return np.asarray(e_norm) * E_std + E_mean


def denorm_y(y_norm):
    y_norm = np.asarray(y_norm)
    y_real = np.empty_like(y_norm, dtype=float)
    y_real[..., 0] = y_norm[..., 0] * Ca_std + Ca_mean
    y_real[..., 1] = y_norm[..., 1] * T_std + T_mean
    return y_real

X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
Y_val = np.load(Y_VAL_PATH)

X_test = np.load(X_TEST_PATH)
Y_test = np.load(Y_TEST_PATH)

if X_train.shape[-1] != INPUT_SIZE:
    raise RuntimeError(f"Expected X input size {INPUT_SIZE}, got {X_train.shape[-1]}.")

if Y_train.shape[-1] != OUTPUT_SIZE:
    raise RuntimeError(f"Expected Y output size {OUTPUT_SIZE}, got {Y_train.shape[-1]}.")

print("Loaded normalized data:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:  ", X_val.shape)
print("Y_val:  ", Y_val.shape)
print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)

print("\nInput format:")
print("X[:, :, 0] = Tc_norm")
print("X[:, :, 1] = EoverR_norm")

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

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


def inf_norm(mat):
    return torch.max(torch.sum(torch.abs(mat), dim=1))


def stability_penalty(model):
    penalty = 0.0
    nu_values = []

    for layer in range(model.num_layers):
        weight_hh = getattr(model.gru, f"weight_hh_l{layer}")
        weight_ih = getattr(model.gru, f"weight_ih_l{layer}")
        bias_hh = getattr(model.gru, f"bias_hh_l{layer}")
        bias_ih = getattr(model.gru, f"bias_ih_l{layer}")

        bias = bias_hh + bias_ih

        _, Uz, Uh = weight_hh.chunk(3, 0)
        _, Wz, _ = weight_ih.chunk(3, 0)
        _, bz, _ = bias.chunk(3, 0)

        z_bound = torch.sigmoid(
            inf_norm(torch.cat([Wz, Uz, bz.unsqueeze(1)], dim=1))
        )

        nu = inf_norm(Uh) + 0.25 * inf_norm(Uz) / (1.0 - z_bound + 1e-8) - 1.0

        penalty = penalty + torch.relu(nu)
        nu_values.append(float(nu.detach().cpu()))

    return penalty, nu_values


model = CSTR_GRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE,
).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
)

criterion = nn.MSELoss()

best_val_mse = float("inf")
best_epoch = 0
best_state = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

train_total_losses = []
train_mse_losses = []
val_mse_losses = []
nu_history = []

print(f"\nTraining on {DEVICE}...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()

    batch_total_losses = []
    batch_mse_losses = []

    iss_weight = RHO * min(1.0, epoch / ISS_RAMP_EPOCHS)

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        y_pred, _ = model(x_batch)

        mse_loss = criterion(
            y_pred[:, WASHOUT:, :],
            y_batch[:, WASHOUT:, :],
        )

        iss_loss, nu_values = stability_penalty(model)
        total_loss = mse_loss + iss_weight * iss_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_total_losses.append(float(total_loss.detach().cpu()))
        batch_mse_losses.append(float(mse_loss.detach().cpu()))

    train_total = float(np.mean(batch_total_losses))
    train_mse = float(np.mean(batch_mse_losses))

    model.eval()
    val_batch_mse = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_pred, _ = model(x_batch)

            val_loss = criterion(
                y_pred[:, WASHOUT:, :],
                y_batch[:, WASHOUT:, :],
            )

            val_batch_mse.append(float(val_loss.detach().cpu()))

    val_mse = float(np.mean(val_batch_mse))

    train_total_losses.append(train_total)
    train_mse_losses.append(train_mse)
    val_mse_losses.append(val_mse)
    nu_history.append(nu_values)

    scheduler.step(val_mse)

    if val_mse < best_val_mse - BEST_MIN_DELTA:
        best_val_mse = val_mse
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch % 10 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"| Train total: {train_total:.6e} "
            f"| Train MSE: {train_mse:.6e} "
            f"| Val MSE: {val_mse:.6e} "
            f"| Best Val: {best_val_mse:.6e} @ {best_epoch} "
            f"| LR: {lr:.2e}"
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

model.load_state_dict(best_state)
model.eval()

print(f"\nLoaded best model from epoch {best_epoch}")
print(f"Best validation MSE: {best_val_mse:.6e}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "output_size": OUTPUT_SIZE,
        "input_features": ["Tc_norm", "EoverR_norm"],
        "output_features": ["Ca_norm", "T_norm"],
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "train_total_losses": train_total_losses,
        "train_mse_losses": train_mse_losses,
        "val_mse_losses": val_mse_losses,
        "nu_history": nu_history,
        "normalization_stats_path": NORMALIZATION_STATS_PATH.name,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "washout": WASHOUT,
            "rho": RHO,
            "iss_ramp_epochs": ISS_RAMP_EPOCHS,
        },
    },
    MODEL_SAVE_PATH,
)

print(f"\nSaved model to: {MODEL_SAVE_PATH}")

def evaluate_model(loader):
    preds = []
    trues = []
    inputs = []

    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)

            y_pred, _ = model(x_batch)

            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.numpy())
            inputs.append(x_batch.cpu().numpy())

    return (
        np.concatenate(inputs, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(preds, axis=0),
    )


X_true_test, Y_true_test, Y_pred_test = evaluate_model(test_loader)


def fit_index(y_true, y_pred):
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true - np.mean(y_true, axis=0, keepdims=True))
    return 100.0 * (1.0 - numerator / (denominator + 1e-8))


def rmse_components_real(y_true_norm, y_pred_norm):
    y_true_real = denorm_y(y_true_norm)
    y_pred_real = denorm_y(y_pred_norm)

    ca_rmse = np.sqrt(np.mean((y_true_real[..., 0] - y_pred_real[..., 0]) ** 2))
    t_rmse = np.sqrt(np.mean((y_true_real[..., 1] - y_pred_real[..., 1]) ** 2))

    return float(ca_rmse), float(t_rmse)


Y_true_eval = Y_true_test[:, WASHOUT:, :]
Y_pred_eval = Y_pred_test[:, WASHOUT:, :]

fit_all = fit_index(
    Y_true_eval.reshape(-1, OUTPUT_SIZE),
    Y_pred_eval.reshape(-1, OUTPUT_SIZE),
)

fit_ca = fit_index(
    Y_true_eval[..., 0:1].reshape(-1, 1),
    Y_pred_eval[..., 0:1].reshape(-1, 1),
)

fit_t = fit_index(
    Y_true_eval[..., 1:2].reshape(-1, 1),
    Y_pred_eval[..., 1:2].reshape(-1, 1),
)

test_ca_rmse_real, test_t_rmse_real = rmse_components_real(
    Y_true_eval,
    Y_pred_eval,
)

sequence_rows = []

for seq_id in range(len(Y_true_test)):
    y_true_seq = Y_true_test[seq_id, WASHOUT:, :]
    y_pred_seq = Y_pred_test[seq_id, WASHOUT:, :]

    seq_fit = fit_index(y_true_seq, y_pred_seq)
    ca_rmse, t_rmse = rmse_components_real(y_true_seq, y_pred_seq)

    sequence_rows.append({
        "SequenceID": seq_id,
        "FIT_percent": seq_fit,
        "Ca_RMSE_mol_L": ca_rmse,
        "T_RMSE_K": t_rmse,
    })

sequence_metrics = pd.DataFrame(sequence_rows)

representative_seq_id = int(
    (sequence_metrics["FIT_percent"] - sequence_metrics["FIT_percent"].median())
    .abs()
    .idxmin()
)

representative_fit = float(sequence_metrics.loc[representative_seq_id, "FIT_percent"])
representative_ca_rmse = float(sequence_metrics.loc[representative_seq_id, "Ca_RMSE_mol_L"])
representative_t_rmse = float(sequence_metrics.loc[representative_seq_id, "T_RMSE_K"])

fit_metric_table = pd.DataFrame({
    "Metric": [
        "Overall FIT [%]",
        "Ca FIT [%]",
        "T FIT [%]",
        "Ca RMSE [mol/L]",
        "T RMSE [K]",
        "Representative sequence ID",
        "Representative sequence FIT [%]",
        "Representative Ca RMSE [mol/L]",
        "Representative T RMSE [K]",
        "Best validation MSE",
        "Best epoch",
    ],
    "Value": [
        fit_all,
        fit_ca,
        fit_t,
        test_ca_rmse_real,
        test_t_rmse_real,
        representative_seq_id,
        representative_fit,
        representative_ca_rmse,
        representative_t_rmse,
        best_val_mse,
        best_epoch,
    ],
})

fit_metric_table.to_csv(OUTPUT_DIR / "fit_metric_table.csv", index=False)

print("\nFIT metric table:")
print(fit_metric_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(8.4, 4.8))

epochs = np.arange(1, len(train_mse_losses) + 1)

ax.scatter(
    epochs,
    train_mse_losses,
    s=22,
    color=COLORS["train"],
    label="Training",
)

ax.scatter(
    epochs,
    val_mse_losses,
    s=22,
    color=COLORS["val"],
    label="Validation",
)

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean squared error")
ax.set_title("Training Loss Evolution")
polish_axes(ax)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

save_figure(fig, "training_loss_evolution.png")

def plot_representative_sequence(seq_id):
    y_true_norm = Y_true_test[seq_id]
    y_pred_norm = Y_pred_test[seq_id]
    x_norm = X_true_test[seq_id]

    y_true_real = denorm_y(y_true_norm)
    y_pred_real = denorm_y(y_pred_norm)

    u_real = denorm_u(x_norm[:, 0])
    e_real = denorm_e(x_norm[:, 1])

    seq_fit = fit_index(
        y_true_norm[WASHOUT:, :],
        y_pred_norm[WASHOUT:, :],
    )

    ca_rmse_real, t_rmse_real = rmse_components_real(
        y_true_norm[WASHOUT:, :],
        y_pred_norm[WASHOUT:, :],
    )

    t_axis = np.arange(len(y_true_norm))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.4, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.35, 1.35]},
    )

    axes[0].step(
        t_axis,
        u_real,
        where="post",
        color=COLORS["input"],
        linewidth=1.8,
    )

    axes[1].plot(
        t_axis,
        e_real,
        color=COLORS["param"],
        linewidth=1.8,
    )

    axes[2].plot(
        t_axis,
        y_true_real[:, 0],
        color=COLORS["true"],
        linewidth=1.8,
        label=r"True $C_A$",
    )

    axes[2].plot(
        t_axis,
        y_pred_real[:, 0],
        color=COLORS["pred"],
        linestyle="--",
        linewidth=1.8,
        label=r"GRU $C_A$",
    )

    axes[3].plot(
        t_axis,
        y_true_real[:, 1],
        color=COLORS["true"],
        linewidth=1.8,
        label=r"True $T$",
    )

    axes[3].plot(
        t_axis,
        y_pred_real[:, 1],
        color=COLORS["pred"],
        linestyle="--",
        linewidth=1.8,
        label=r"GRU $T$",
    )

    axes[0].set_ylabel(r"$T_c$ [K]")
    axes[1].set_ylabel(r"$E/R$ [K]")
    axes[2].set_ylabel(r"$C_A$ [mol/L]")
    axes[3].set_ylabel(r"$T$ [K]")
    axes[3].set_xlabel("Time step")

    axes[2].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
    axes[3].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

    for ax in axes:
        polish_axes(ax)

    fig.suptitle(
        (
            f"Model Prediction on a Representative Test Sequence | "
            f"FIT = {seq_fit:.2f}% | "
            rf"$C_A$ RMSE = {ca_rmse_real:.4f} mol/L | "
            rf"$T$ RMSE = {t_rmse_real:.2f} K"
        ),
        y=0.995,
    )

    save_figure(fig, "representative_test_prediction.png")


plot_representative_sequence(representative_seq_id)


print("\nDone.")
print("Generated:")
print("- cstr_gru_deltaISS.pth")
print("- training_loss_evolution.png / .pdf")
print("- representative_test_prediction.png / .pdf")
print("- fit_metric_table.csv")
