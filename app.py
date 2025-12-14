# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Physical constants (SI) ---
hbar = 1.054_571_817e-34  # J·s
m_e  = 9.109_383_7015e-31 # kg
eV   = 1.602_176_634e-19  # J

st.set_page_config(page_title="Double-Barrier Resonant Tunneling", layout="wide")

st.title("Double-Barrier Resonant Tunneling (5 Regions)")
st.write(
    "Electron incident from the left. Two identical barriers of height Vb and width b, "
    "separated by a well of width L and potential Vw."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Parameters (eV, nm)")

E_eV  = st.sidebar.slider("Electron energy E (eV)", 0.001, 20.0, 2.0, 0.001)
Vb_eV = st.sidebar.slider("Barrier height Vb (eV)", 0.0,  20.0, 5.0, 0.001)
b_nm  = st.sidebar.slider("Barrier width b (nm)",  0.01,  10.0, 1.0, 0.01)

Vw_eV = st.sidebar.slider("Well potential Vw (eV)", -5.0,  5.0, 0.0, 0.001)
L_nm  = st.sidebar.slider("Well width L (nm)",     0.01,  20.0, 2.0, 0.01)

st.sidebar.divider()
phi0 = st.sidebar.slider("Global phase φ₀ (rad)", -np.pi, np.pi, 0.0, 0.01)

Lmult = st.sidebar.slider("Plot padding (× total length)", 0.5, 6.0, 2.0, 0.5)
Npts  = st.sidebar.slider("Number of x points", 600, 8000, 2400, 100)

plot_mode = st.sidebar.selectbox(
    "Plot",
    ["Both (|ψ|² and Re/Im)", "|ψ(x)|²", "Re/Im ψ(x)"]
)

show_potential = st.sidebar.checkbox("Show potential profile", True)

# ---------------- Units ----------------
E  = E_eV  * eV
Vb = Vb_eV * eV
Vw = Vw_eV * eV

b  = b_nm * 1e-9
L  = L_nm * 1e-9

if E <= 0:
    st.error("E must be > 0.")
    st.stop()

# Region boundaries (absolute x, meters)
x0 = 0.0
x1 = b
x2 = b + L
x3 = 2*b + L  # end of 2nd barrier

# Outside leads potential (fixed to 0 here)
V_out = 0.0

def k_of(EJ, VJ):
    """Complex wave number sqrt(2m(E-V))/ħ, allowing evanescent regions."""
    return np.sqrt(2*m_e*(EJ - VJ) / (hbar**2) + 0j)

k1 = k_of(E, V_out)   # Region I
k2 = k_of(E, Vb)      # Region II barrier
k3 = k_of(E, Vw)      # Region III well
k4 = k2               # Region IV barrier
k5 = k1               # Region V

# ---------------- Solve boundary matching (8 unknowns) ----------------
# Unknown vector: [r, A2, B2, A3, B3, A4, B4, t]
# Region forms:
# I:  psi1 = exp(ik1 x) + r exp(-ik1 x)
# II: psi2 = A2 exp(ik2 x) + B2 exp(-ik2 x)
# III:psi3 = A3 exp(ik3 x) + B3 exp(-ik3 x)
# IV: psi4 = A4 exp(ik4 x) + B4 exp(-ik4 x)
# V:  psi5 = t exp(ik5 x)

def expi(k, x):  # exp(i k x)
    return np.exp(1j * k * x)

def expmi(k, x): # exp(-i k x)
    return np.exp(-1j * k * x)

def d_expi(k, x):   # d/dx exp(i k x) = i k exp(i k x)
    return 1j * k * np.exp(1j * k * x)

def d_expmi(k, x):  # d/dx exp(-i k x) = -i k exp(-i k x)
    return -1j * k * np.exp(-1j * k * x)

M = np.zeros((8, 8), dtype=complex)
bvec = np.zeros(8, dtype=complex)

# Helper: fill continuity at boundary xb between region left and right.
# Left psi = left_inc + left_ref ; Right psi = A exp(i k x)+B exp(-i k x) etc.
# We'll write each equation as M @ u = b.

# --- Boundary at x0 (Region I <-> II) ---
# (1) psi1(x0) = psi2(x0)
# exp(i k1 x0) + r exp(-i k1 x0) - A2 exp(i k2 x0) - B2 exp(-i k2 x0) = 0
row = 0
M[row, 0] = expmi(k1, x0)            # r coefficient
M[row, 1] = -expi(k2, x0)            # A2
M[row, 2] = -expmi(k2, x0)           # B2
bvec[row] = -expi(k1, x0)            # move incident term to RHS

# (2) dpsi1(x0) = dpsi2(x0)
# i k1 exp(i k1 x0) + r (-i k1 exp(-i k1 x0)) - A2 (i k2 exp(i k2 x0)) - B2 (-i k2 exp(-i k2 x0)) = 0
row = 1
M[row, 0] = -1j * k1 * expmi(k1, x0) # r term
M[row, 1] = -d_expi(k2, x0)          # A2
M[row, 2] = -d_expmi(k2, x0)         # B2
bvec[row] = -d_expi(k1, x0)          # incident derivative to RHS

# --- Boundary at x1 (Region II <-> III) ---
# (3) psi2(x1) = psi3(x1)
row = 2
M[row, 1] = expi(k2, x1)             # A2
M[row, 2] = expmi(k2, x1)            # B2
M[row, 3] = -expi(k3, x1)            # A3
M[row, 4] = -expmi(k3, x1)           # B3
bvec[row] = 0

# (4) dpsi2(x1) = dpsi3(x1)
row = 3
M[row, 1] = d_expi(k2, x1)
M[row, 2] = d_expmi(k2, x1)
M[row, 3] = -d_expi(k3, x1)
M[row, 4] = -d_expmi(k3, x1)
bvec[row] = 0

# --- Boundary at x2 (Region III <-> IV) ---
# (5) psi3(x2) = psi4(x2)
row = 4
M[row, 3] = expi(k3, x2)             # A3
M[row, 4] = expmi(k3, x2)            # B3
M[row, 5] = -expi(k4, x2)            # A4
M[row, 6] = -expmi(k4, x2)           # B4
bvec[row] = 0

# (6) dpsi3(x2) = dpsi4(x2)
row = 5
M[row, 3] = d_expi(k3, x2)
M[row, 4] = d_expmi(k3, x2)
M[row, 5] = -d_expi(k4, x2)
M[row, 6] = -d_expmi(k4, x2)
bvec[row] = 0

# --- Boundary at x3 (Region IV <-> V) ---
# (7) psi4(x3) = psi5(x3)
row = 6
M[row, 5] = expi(k4, x3)             # A4
M[row, 6] = expmi(k4, x3)            # B4
M[row, 7] = -expi(k5, x3)            # t
bvec[row] = 0

# (8) dpsi4(x3) = dpsi5(x3)
row = 7
M[row, 5] = d_expi(k4, x3)
M[row, 6] = d_expmi(k4, x3)
M[row, 7] = -d_expi(k5, x3)          # t
bvec[row] = 0

try:
    sol = np.linalg.solve(M, bvec)
except np.linalg.LinAlgError:
    st.error("Singular system for these parameters (rare). Try slight parameter changes.")
    st.stop()

r, A2, B2, A3, B3, A4, B4, t = sol

# Apply a global phase rotation if desired (visual only; does not change R,T)
phase = np.exp(1j * phi0)

# ---------------- Reflection & Transmission ----------------
# Flux ratio: T = (Re(k5)/Re(k1)) |t|^2 if both propagating.
# Here k1=k5 for V_out=0, so T=|t|^2 when k1 is real.
k1r = np.real(k1)
k5r = np.real(k5)

R = float(np.abs(r)**2)

if k1r > 0 and k5r > 0:
    T = float((k5r / k1r) * np.abs(t)**2)
else:
    # If outside leads become evanescent (shouldn't here), define T=0
    T = 0.0

RT_sum = R + T

# ---------------- Build x grid & wavefunction ----------------
total_len = x3 - x0
pad = Lmult * total_len

x_min = -pad
x_max = x3 + pad
x = np.linspace(x_min, x_max, Npts)

psi = np.zeros_like(x, dtype=complex)

mask1 = x < x0
mask2 = (x >= x0) & (x <= x1)
mask3 = (x > x1) & (x <= x2)
mask4 = (x > x2) & (x <= x3)
mask5 = x > x3

# Regions
psi[mask1] = expi(k1, x[mask1]) + r * expmi(k1, x[mask1])
psi[mask2] = A2 * expi(k2, x[mask2]) + B2 * expmi(k2, x[mask2])
psi[mask3] = A3 * expi(k3, x[mask3]) + B3 * expmi(k3, x[mask3])
psi[mask4] = A4 * expi(k4, x[mask4]) + B4 * expmi(k4, x[mask4])
psi[mask5] = t  * expi(k5, x[mask5])

# Global phase (visual rotation)
psi *= phase

prob = np.abs(psi)**2

# Potential profile for plotting (in eV)
V_profile = np.zeros_like(x, dtype=float)
V_profile[mask2] = Vb_eV
V_profile[mask3] = Vw_eV
V_profile[mask4] = Vb_eV
# outside remains 0

# ---------------- Layout ----------------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Results")
    st.metric("Transmission T", f"{T:.6f}")
    st.metric("Reflection R", f"{R:.6f}")
    st.metric("R + T", f"{RT_sum:.6f}")
    st.caption("Expect R+T ≈ 1 for propagating leads (small numerical deviation may occur).")

    st.subheader("Geometry")
    st.write(f"Barrier width b = {b_nm:.3f} nm")
    st.write(f"Well width   L = {L_nm:.3f} nm")
    st.write(f"Total length = {(x3-x0)*1e9:.3f} nm")

    st.subheader("Coefficients (complex)")
    st.write(f"r  = {r:.6g}")
    st.write(f"t  = {t:.6g}")
    st.write(f"A2 = {A2:.6g},  B2 = {B2:.6g}")
    st.write(f"A3 = {A3:.6g},  B3 = {B3:.6g}")
    st.write(f"A4 = {A4:.6g},  B4 = {B4:.6g}")

    st.subheader("Wave numbers (1/m)")
    st.write(f"k1 (lead)   = {k1:.3e}")
    st.write(f"k2 (barrier)= {k2:.3e}")
    st.write(f"k3 (well)   = {k3:.3e}")

    if E_eV < Vb_eV:
        st.info("E < Vb: inside barriers, wave is evanescent (decays).")
    else:
        st.info("E > Vb: inside barriers, wave oscillates (over-the-barrier regime).")

with col2:
    st.subheader("Wavefunction visualization")

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    # Boundaries
    for xb in [x0, x1, x2, x3]:
        ax.axvline(xb, linestyle="--", linewidth=1)

    if plot_mode == "|ψ(x)|²":
        ax.plot(x, prob, label="|ψ(x)|²")
        ax.set_ylabel("|ψ(x)|²")

    elif plot_mode == "Re/Im ψ(x)":
        ax.plot(x, np.real(psi), label="Re[ψ(x)]")
        ax.plot(x, np.imag(psi), label="Im[ψ(x)]")
        ax.set_ylabel("Re/Im ψ(x)")

    else:
        # Combined plot with normalization for readability
        prob_scale = np.max(prob) if np.max(prob) > 0 else 1.0
        amp_scale  = np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else 1.0

        ax.plot(x, prob / prob_scale, label="|ψ(x)|² (normalized)")
        ax.plot(x, np.real(psi) / amp_scale, label="Re[ψ(x)] (normalized)")
        ax.plot(x, np.imag(psi) / amp_scale, label="Im[ψ(x)] (normalized)")
        ax.set_ylabel("Normalized curves")

    ax.set_xlabel("x (m)")
    ax.set_title(
        f"Double barrier: Vb={Vb_eV:.3f} eV, b={b_nm:.3f} nm, "
        f"Vw={Vw_eV:.3f} eV, L={L_nm:.3f} nm, E={E_eV:.3f} eV"
    )
    ax.legend(loc="upper right")

    if show_potential:
        ax2 = ax.twinx()
        ax2.plot(x, V_profile, linewidth=1)
        ax2.set_ylabel("V(x) (eV)")
        ax2.set_ylim(min(-6, np.min(V_profile)-1), max(21, np.max(V_profile)+1))

    st.pyplot(fig)

st.markdown(
    """
**How to run**
1. Save as `app.py`
2. Install: `pip install streamlit numpy matplotlib`
3. Run: `streamlit run app.py`
"""
)

st.markdown(
    """
**Tip (to see resonances clearly):** Fix Vb and b, then sweep E slowly around a few eV and watch T spike toward 1.
Resonances sharpen for higher/thicker barriers and a well width that supports quasi-bound states.
"""
)
