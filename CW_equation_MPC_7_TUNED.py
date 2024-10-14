import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Clohessy-Wiltshire (CW) equations parameters
n = 0.00155  # Mean motion of the target spacecraft (rad/s)

# Discrete-time system matrices (CW equations in state-space form) for 3 dimensions
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [3*n**2, 0, 0, 0, 2*n, 0],
              [0, 0, 0, -2*n, 0, 0],
              [0, 0, -n**2, 0, 0, 0]])

# Input matrix B, linking control inputs (ux, uy, uz) to the state dynamics
B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Cost function matrices for MPC controller
Q = 100 * np.diag([10, 1, 1, 1, 1, 1])  # Penalize position errors more than velocity errors
R = 1000 * np.diag([1, 1, 1])           # Penalize large control inputs

# Simulation parameters
dt = 0.1  # Time step (seconds)
T = 1000  # Total number of time steps in the simulation
horizon = 20  # MPC prediction horizon (how many steps ahead we optimize)

# Initial conditions for the chaser spacecraft
x0 = np.array([2000, 1000, -800, 0.2, 0.15, 0.01])  # Initial state of the chaser

# Define two target trajectory functions
def target_trajectory_moving(t):
    """Sinusoidal target motion for a moving target."""
    return np.array([100 * np.sin(0.001*t), 50 * np.sin(0.0005*t), 0])

def target_trajectory_stationary(t):
    """Stationary target located at the origin."""
    return np.array([0, 0, 0])

# Choose which target trajectory to use for the simulation
use_moving_target = False  # Set to False to use the stationary target

if use_moving_target:
    target_trajectory = target_trajectory_moving
else:
    target_trajectory = target_trajectory_stationary

# Initialize history arrays to store simulation data for plotting later
x_hist = np.zeros((T, 6))  # Chaser state history (position and velocity over time)
u_hist = np.zeros((T, 3))  # Control input history (thrust applied over time)

# MPC objective function
def mpc_cost(u, x, target, N):
    """Cost function for MPC optimization over the prediction horizon."""
    cost = 0
    x_sim = np.copy(x)  # Simulate the state over the prediction horizon
    u = u.reshape(N, 3)  # Reshape the control input into a sequence of control vectors
    
    for i in range(N):
        # Compute position error (difference between chaser and target)
        state_error = x_sim[:3] - target_trajectory(i)
        # Calculate the cost at this step: penalize position error and control effort
        cost += state_error.T @ Q[:3, :3] @ state_error + u[i].T @ R @ u[i]
        
        # Update the simulated state using the current control input
        x_sim = x_sim + (A @ x_sim + B @ u[i]) * dt

    return cost  # Return the total cost over the horizon

# Initial state of the chaser spacecraft
x = x0

# Simulation loop
for t in range(T):
    # At each time step, retrieve the target's current position
    target_pos = target_trajectory(t)
    
    # Initial guess for control inputs over the horizon (zero thrust initially)
    u_guess = np.zeros(horizon * 3)  # 3 control inputs per time step (ux, uy, uz)
    
    # Control constraints: limit the magnitude of control inputs to ±10 N
    bounds = [(-1000, 1000)] * horizon * 3  # Define control limits for each input
    
    # Use optimization to minimize the MPC cost function
    result = minimize(mpc_cost, u_guess, args=(x, target_pos, horizon), bounds=bounds)
    
    # The optimizer gives us a sequence of control inputs; apply the first one
    u = result.x[:3]
    u_hist[t, :] = u  # Store the applied control input
    
    # Update the state of the chaser spacecraft using the applied control input
    x = x + (A @ x + B @ u) * dt
    x_hist[t, :] = x  # Store the new state for plotting later

# Display the final position and velocity of the chaser spacecraft
final_position = x_hist[-1, :3]  # Final position (x, y, z)
final_velocity = x_hist[-1, 3:]  # Final velocity (vx, vy, vz)
print(f"Final Position (x, y, z): {final_position} m")
print(f"Final Velocity (vx, vy, vz): {final_velocity} m/s")

# Plot 1: 3D Position Trajectory of the chaser spacecraft
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the chaser's trajectory
ax.plot(x_hist[:, 0], x_hist[:, 1], x_hist[:, 2], label="Chaser Position")
# Mark the initial position of the target (which is moving or stationary)
ax.scatter(0, 0, 0, color='red', marker='*', s=100, label='Target Start')
# Mark the starting position of the chaser
ax.scatter(x0[0], x0[1], x0[2], color='green', marker='o', s=100, label='Start')
ax.set_title("Chaser Spacecraft 3D Position Trajectory")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.legend()
plt.grid(True)

# Plot 2: Control Inputs over Time
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, T*dt, dt), u_hist[:, 0], label="Control input u_x")
plt.plot(np.arange(0, T*dt, dt), u_hist[:, 1], label="Control input u_y")
plt.plot(np.arange(0, T*dt, dt), u_hist[:, 2], label="Control input u_z")
plt.title("Control Inputs over Time")
plt.xlabel("Time (s)")
plt.ylabel("Control Force (N)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
