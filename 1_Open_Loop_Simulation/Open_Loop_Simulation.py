import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

base_parms = {
    "q": 1.0,
    "V": 1.0,
    "rho": 1000.0,
    "Cp": 1.0,
    "deltaH": 2e5,
    "EoverR": 1e4,
    "k0": 7.2e10,
    "UA": 1000.0,
    "Tf": 350.0,
    "CAf": 1.0,
}

EOVER_R_VALUES = [0.99e4, 1.00e4, 1.01e4]
SELECTED_BRANCH = "low_temperature_branch"

TC_MIN = 280.0
TC_MAX = 400.0
TC_NOMINAL = 350.0
TC_SIGNAL_MIN = 320.0
TC_SIGNAL_MAX = 360.0

T_MIN = 250.0
T_MAX = 550.0
CA_MIN = 0.0
CA_MAX = 1.0

DT = 1.0
SIM_TIME = 5000.0
TIME = np.arange(0.0, SIM_TIME, DT)

N_PROFILES = 60
MAX_PROFILE_ATTEMPTS = 5000

APRBS_AMPLITUDE = 25.0
APRBS_MIN_HOLD = 500
APRBS_MAX_HOLD = 800

SEQUENCE_LENGTH = 700
STRIDE = 150

CA_NOISE_STD = 0.001
T_NOISE_STD = 0.5
BRANCH_GUARD_MARGIN_T = 20.0

TC_STD_FLOOR = 10.0
CA_STD_FLOOR = 0.05
T_STD_FLOOR = 10.0


def reaction_rate(T, parms):
    return parms["k0"] * np.exp(-parms["EoverR"] / T)


def cstr_model(t, x, Tc, parms):
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

    return [dCa, dT]


def cstr_step(x, Tc, parms):
    sol = solve_ivp(
        lambda t, x_: cstr_model(t, x_, Tc, parms),
        [0.0, DT],
        x,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
    )

    x_next = sol.y[:, -1]
    x_next[0] = np.clip(x_next[0], CA_MIN, CA_MAX)
    x_next[1] = np.clip(x_next[1], T_MIN, T_MAX)

    return x_next


def steady_state_equations(x, parms):
    return cstr_model(0.0, x, TC_NOMINAL, parms)


def find_steady_states(parms):
    guesses = [[0.9, 350.0], [0.5, 400.0], [0.1, 450.0]]
    steady_states = []

    for guess in guesses:
        ss = fsolve(lambda x: steady_state_equations(x, parms), guess, xtol=1e-8)
        ss[0] = np.clip(ss[0], CA_MIN, CA_MAX)
        ss[1] = np.clip(ss[1], T_MIN, T_MAX)

        if not any(np.allclose(ss, s, atol=1e-2) for s in steady_states):
            steady_states.append(ss)

    return sorted(steady_states, key=lambda x: x[1])


def select_branch_steady_state(steady_states):
    if SELECTED_BRANCH == "low_temperature_branch":
        return steady_states[0]

    if SELECTED_BRANCH == "high_temperature_branch":
        return steady_states[-1]

    raise ValueError("SELECTED_BRANCH must be low_temperature_branch or high_temperature_branch.")


def branch_guard_ok(x, steady_states):
    if len(steady_states) < 3:
        return True

    middle_temperature = steady_states[1][1]

    if SELECTED_BRANCH == "low_temperature_branch":
        return x[1] <= middle_temperature - BRANCH_GUARD_MARGIN_T

    if SELECTED_BRANCH == "high_temperature_branch":
        return x[1] >= middle_temperature + BRANCH_GUARD_MARGIN_T

    return False


def generate_aprbs(n_steps):
    signal = np.zeros(n_steps)
    current_value = TC_NOMINAL
    hold_counter = 0

    for k in range(n_steps):
        if hold_counter <= 0:
            current_value = TC_NOMINAL + np.random.uniform(
                -APRBS_AMPLITUDE,
                APRBS_AMPLITUDE,
            )
            current_value = np.clip(current_value, TC_SIGNAL_MIN, TC_SIGNAL_MAX)
            hold_counter = np.random.randint(APRBS_MIN_HOLD, APRBS_MAX_HOLD + 1)

        signal[k] = current_value
        hold_counter -= 1

    return signal


steady_state_by_E = {}
all_steady_states_by_E = {}

for EoverR in EOVER_R_VALUES:
    parms = base_parms.copy()
    parms["EoverR"] = EoverR

    steady_states = find_steady_states(parms)
    selected_ss = select_branch_steady_state(steady_states)

    steady_state_by_E[EoverR] = selected_ss
    all_steady_states_by_E[EoverR] = steady_states

    print(f"\nEoverR = {EoverR:.0f}")
    for i, ss in enumerate(steady_states):
        print(f"SS{i + 1}: Ca={ss[0]:.4f}, T={ss[1]:.2f}")
    print(f"Using {SELECTED_BRANCH}: Ca={selected_ss[0]:.4f}, T={selected_ss[1]:.2f}")


def simulate_single_trajectory(Tc_signal, EoverR, e_idx, profile_id, trajectory_id):
    parms = base_parms.copy()
    parms["EoverR"] = EoverR

    steady_states = all_steady_states_by_E[EoverR]
    x = steady_state_by_E[EoverR].copy().astype(float)

    Ca_hist = []
    T_hist = []

    for k in range(len(TIME) - 1):
        x = cstr_step(x, Tc_signal[k], parms)

        if not branch_guard_ok(x, steady_states):
            return None, False

        Ca_meas = x[0] + np.random.normal(0.0, CA_NOISE_STD)
        T_meas = x[1] + np.random.normal(0.0, T_NOISE_STD)

        Ca_hist.append(np.clip(Ca_meas, CA_MIN, CA_MAX))
        T_hist.append(np.clip(T_meas, T_MIN, T_MAX))

    df = pd.DataFrame({
        "Trajectory": trajectory_id,
        "ProfileID": profile_id,
        "Branch": SELECTED_BRANCH,
        "EoverR": EoverR,
        "EoverR_Label": e_idx,
        "Time": TIME[:-1],
        "Tc": Tc_signal[:-1],
        "Ca": Ca_hist,
        "T": T_hist,
    })

    return df, True


all_data = []
accepted_profiles = 0
attempts = 0
global_trajectory_id = 0

print("\nGenerating branch-safe open-loop source dataset...")

while accepted_profiles < N_PROFILES and attempts < MAX_PROFILE_ATTEMPTS:
    attempts += 1
    Tc_signal = generate_aprbs(len(TIME))

    candidate_data = []
    profile_ok = True

    for e_idx, EoverR in enumerate(EOVER_R_VALUES):
        df, ok = simulate_single_trajectory(
            Tc_signal=Tc_signal,
            EoverR=EoverR,
            e_idx=e_idx,
            profile_id=accepted_profiles,
            trajectory_id=global_trajectory_id + e_idx,
        )

        if not ok:
            profile_ok = False
            break

        candidate_data.append(df)

    if profile_ok:
        all_data.extend(candidate_data)
        global_trajectory_id += len(EOVER_R_VALUES)
        accepted_profiles += 1

        if accepted_profiles % 10 == 0:
            print(f"Accepted profile {accepted_profiles}/{N_PROFILES} after {attempts} attempts")

if accepted_profiles < N_PROFILES:
    raise RuntimeError(
        f"Only accepted {accepted_profiles}/{N_PROFILES} profiles. "
        "Relax APRBS limits or branch guard."
    )

source_data = pd.concat(all_data, ignore_index=True)
source_data = source_data.sort_values(["ProfileID", "EoverR", "Time"]).reset_index(drop=True)

Tc_mean = source_data["Tc"].mean()
Ca_mean = source_data["Ca"].mean()
T_mean = source_data["T"].mean()
E_mean = source_data["EoverR"].mean()

Tc_std_raw = source_data["Tc"].std()
Ca_std_raw = source_data["Ca"].std()
T_std_raw = source_data["T"].std()
E_std_raw = source_data["EoverR"].std()

Tc_std = max(Tc_std_raw, TC_STD_FLOOR)
Ca_std = max(Ca_std_raw, CA_STD_FLOOR)
T_std = max(T_std_raw, T_STD_FLOOR)
E_std = E_std_raw

source_data["Tc_norm"] = (source_data["Tc"] - Tc_mean) / (Tc_std + 1e-8)
source_data["Ca_norm"] = (source_data["Ca"] - Ca_mean) / (Ca_std + 1e-8)
source_data["T_norm"] = (source_data["T"] - T_mean) / (T_std + 1e-8)
source_data["EoverR_norm"] = (source_data["EoverR"] - E_mean) / (E_std + 1e-8)

normalization_stats = pd.DataFrame({
    "Variable": ["Tc", "Ca", "T", "EoverR"],
    "Mean": [Tc_mean, Ca_mean, T_mean, E_mean],
    "Std": [Tc_std, Ca_std, T_std, E_std],
    "RawStd": [Tc_std_raw, Ca_std_raw, T_std_raw, E_std_raw],
    "StdFloor": [TC_STD_FLOOR, CA_STD_FLOOR, T_STD_FLOOR, np.nan],
    "StdWasFloored": [
        Tc_std_raw < TC_STD_FLOOR,
        Ca_std_raw < CA_STD_FLOOR,
        T_std_raw < T_STD_FLOOR,
        False,
    ],
    "ComputedFrom": ["single_branch_full_source_dataset"] * 4,
})

normalization_stats.to_csv("normalization_stats.csv", index=False)

profile_ids = source_data["ProfileID"].unique()

train_profiles, temp_profiles = train_test_split(
    profile_ids,
    test_size=0.3,
    random_state=SEED,
)

val_profiles, test_profiles = train_test_split(
    temp_profiles,
    test_size=0.5,
    random_state=SEED,
)

train_df = source_data[source_data["ProfileID"].isin(train_profiles)].copy()
val_df = source_data[source_data["ProfileID"].isin(val_profiles)].copy()
test_df = source_data[source_data["ProfileID"].isin(test_profiles)].copy()


def create_sequences(df):
    X = []
    Y = []

    for _, group in df.groupby("Trajectory"):
        group = group.reset_index(drop=True)

        u = group[["Tc_norm", "EoverR_norm"]].values.astype(np.float32)
        y = group[["Ca_norm", "T_norm"]].values.astype(np.float32)

        for i in range(0, len(group) - SEQUENCE_LENGTH + 1, STRIDE):
            X.append(u[i:i + SEQUENCE_LENGTH])
            Y.append(y[i:i + SEQUENCE_LENGTH])

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


X_train, Y_train = create_sequences(train_df)
X_val, Y_val = create_sequences(val_df)
X_test, Y_test = create_sequences(test_df)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

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

sample_profile = int(source_data["ProfileID"].min())
colors = ["#1f77b4", "#d62728", "#2ca02c"]

fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True)

sample_input = source_data[source_data["ProfileID"] == sample_profile].iloc[::len(EOVER_R_VALUES)]

axes[0].step(
    sample_input["Time"],
    sample_input["Tc"],
    where="post",
    color="#2c3e50",
    linewidth=1.8,
    label=r"$T_c$ APRBS",
)

for idx, EoverR in enumerate(EOVER_R_VALUES):
    sample = source_data[
        (source_data["ProfileID"] == sample_profile)
        & (source_data["EoverR"] == EoverR)
    ]

    axes[1].plot(
        sample["Time"],
        sample["Ca"],
        color=colors[idx],
        linewidth=1.6,
        label=rf"$E/R={EoverR:.0f}$",
    )

    axes[2].plot(
        sample["Time"],
        sample["T"],
        color=colors[idx],
        linewidth=1.6,
        label=rf"$E/R={EoverR:.0f}$",
    )

axes[0].set_ylabel(r"$T_c$ [K]")
axes[1].set_ylabel(r"$C_A$ [mol/L]")
axes[2].set_ylabel(r"$T$ [K]")
axes[2].set_xlabel("Time [s]")

axes[0].set_title("Representative Open-Loop CSTR Trajectory")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(frameon=True, loc="best")
axes[1].legend(frameon=True, loc="best")

fig.tight_layout()
fig.savefig("representative_open_loop_trajectory.png", bbox_inches="tight")
plt.show()

print("\nSaved files:")
print("- X_train.npy")
print("- Y_train.npy")
print("- X_val.npy")
print("- Y_val.npy")
print("- X_test.npy")
print("- Y_test.npy")
print("- normalization_stats.csv")
print("- representative_open_loop_trajectory.png")

print("\nArray shapes:")
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:  ", X_val.shape, "Y_val:  ", Y_val.shape)
print("X_test: ", X_test.shape, "Y_test: ", Y_test.shape)
