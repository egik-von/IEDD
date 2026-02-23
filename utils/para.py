# constants.py

# General
DELTA_T = 0.1  # Simulation time step

# Vehicle Capabilities
V_LIM = 20.0  # Vehicle speed limit (m/s)  
V_DESIRE = 20.0 # Desired speed (m/s)  
A_LIM = 3.0   # Vehicle maximum acceleration capability (m/s^2)  
A_NORMAL = 2.0  # Normalized baseline acceleration (m/s^2)  

# Q_i Weights
W_S = 0.3  # Ego pose adjustment weight  
W_R = 0.3  # Ego risk change weight  
W_P = 0.4  # Ego risk potential weight  

# --- Q_si Parameters (ADJUSTED FOR MAGNITUDE SCALING) ---
# Original parameters  ument
VEL_CHANGE_SCALE = 15.0   # (New) Velocity change scaling factor
ALPHA_Q = 150.0           # (Adjusted) Heading change weight/scaler, was 0.3
BETA_Q = 0.5              # (Adjusted) Acceleration weight, was 0.4
# --- Q_ri Parameters (Confirmed to be reasonable) ---
GAMMA_Q = 0.4  # PET change sensitive factor  

# --- Q_pi Parameters (Confirmed to be reasonable) ---
GAMMA_B = 0.3  # Backward low potential region weight  
GAMMA_F = 1.0  # Forward high potential region weight  
BETA_POTENTIAL = 2.0 # Smoothing factor  
D0 = 15.0      # (Chosen) Risk potential characteristic distance (m)
KAPPA_V = 1.0  # Speed sensitive factor  
V0 = 5.0       # (Chosen) Relative speed normalized characteristic value (m/s)

# --- E_i Parameters (Confirmed  ) ---
ALPHA_E = 0.8  # Time sensitive factor  
BETA_E = 1.2   # Smoothness sensitive factor  