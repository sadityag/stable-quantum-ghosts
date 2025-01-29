import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.colors as colors

class QuantumGhostSimulator:
    def __init__(self, nx=128, ny=128, dx=0.2, dy=0.2, dt=0.01, 
                 x_init=2, y_init=1, sigma_x=2, sigma_y=2, lambda_coupling=1/3):
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt = dt
        
        # Physical parameters
        self.lambda_coupling = lambda_coupling
        
        # Create spatial grids
        self.x = (np.arange(nx) - nx//2) * dx
        self.y = (np.arange(ny) - ny//2) * dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Create momentum grids
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Initialize wave function (ensuring it's complex)
        self.psi = self._create_initial_state(x_init, y_init, sigma_x, sigma_y)
        
        # Storage for expectation values
        self.x_means = []
        self.y_means = []
        self.px_means = []
        self.py_means = []
        self.times = []
        
        # Precompute propagators
        self._init_propagators()

    def _create_initial_state(self, x_init, y_init, sigma_x, sigma_y):
        """Create initial Gaussian wave packet"""
        # Create as complex array from the start
        psi = np.zeros((self.nx, self.ny), dtype=np.complex128)
        
        # Real Gaussian envelope
        psi.real = np.exp(-(self.X - x_init)**2/(2*sigma_x**2) 
                         -(self.Y - y_init)**2/(2*sigma_y**2))
        
        # Add initial momentum if desired (currently zero)
        # psi *= np.exp(1j * (px_init * self.X + py_init * self.Y))
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dy)
        return psi / norm

    def _compute_potential(self):
        """Compute the interaction potential V_I"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        denominator = (self.X**2 - self.Y**2 - 1)**2 + 4*self.X**2 + epsilon
        return self.lambda_coupling * denominator**(-0.5)

    def _init_propagators(self):
        """Precompute the propagators for efficiency"""
        # Kinetic propagator in momentum space
        self.kinetic_propagator = np.exp(-1j * self.dt * (0.5 * self.KX**2 - 0.5 * self.KY**2))
        
        # Potential propagator in position space
        V = 0.5*(self.X**2 - self.Y**2) + self._compute_potential()
        self.potential_propagator = np.exp(-1j * self.dt * V / 2)  # Half step

    def compute_expectation_values(self):
        """Compute expectation values of position and momentum"""
        prob_density = np.abs(self.psi)**2
        
        # Position expectations
        x_mean = np.sum(self.X * prob_density) * self.dx * self.dy
        y_mean = np.sum(self.Y * prob_density) * self.dx * self.dy
        
        # Momentum expectations using FFT
        psi_k = fftshift(fft2(self.psi))
        k_prob_density = np.abs(psi_k)**2
        
        # Scale factors for momentum
        dk_x = 2*np.pi/(self.nx*self.dx)
        dk_y = 2*np.pi/(self.ny*self.dy)
        
        px_mean = np.sum(self.KX * k_prob_density) * dk_x * dk_y
        py_mean = np.sum(self.KY * k_prob_density) * dk_x * dk_y
        
        return x_mean, y_mean, px_mean, py_mean

    def step(self):
        """Perform one time step using the split-operator method
            with kinetic and potential propagators. This makes use of the
            Trotter decomposition of the time evolution operator.
        """
        # Half step with potential
        self.psi *= self.potential_propagator
        
        # Full step with kinetic energy in momentum space
        psi_k = fft2(self.psi)
        psi_k *= self.kinetic_propagator
        self.psi = ifft2(psi_k)
        
        # Half step with potential
        self.psi *= self.potential_propagator
        
        # Renormalize occasionally to prevent numerical drift
        if np.random.rand() < 0.01:  # 1% chance each step
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)
            self.psi /= norm

    def evolve(self, n_steps):
        """Evolve the system for n_steps"""
        for i in range(n_steps):
            self.step()
            
            if i % 10 == 0:  # Store every 10th step for efficiency
                x_mean, y_mean, px_mean, py_mean = self.compute_expectation_values()
                self.x_means.append(x_mean)
                self.y_means.append(y_mean)
                self.px_means.append(px_mean)
                self.py_means.append(py_mean)
                self.times.append(i * self.dt)

    def plot_trajectories(self):
        """Plot position and momentum space trajectories"""
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(16, 6))
        
        # Create a special gridspec that reserves space for colorbar
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        
        # Create axes
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        cax = plt.subplot(gs[2])  # Colorbar axis
        
        # Position space plot
        for i in range(len(self.times)-1):
            # Color varies with time
            color = plt.cm.viridis(i / (len(self.times)-1))
            ax1.plot(self.x_means[i:i+2], self.y_means[i:i+2], 
                    color=color, linewidth=1.5)
        
        # Momentum space plot
        for i in range(len(self.times)-1):
            color = plt.cm.viridis(i / (len(self.times)-1))
            ax2.plot(self.px_means[i:i+2], self.py_means[i:i+2], 
                    color=color, linewidth=1.5)
        
        # Set labels and titles
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Position Space Trajectory')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('px')
        ax2.set_ylabel('py')
        ax2.set_title('Momentum Space Trajectory')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=min(self.times), 
                                                   vmax=max(self.times)))
        plt.colorbar(sm, cax=cax, label='Time')
        
        # Ensure plots are square aspect ratio like in the paper
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()

def run_simulation(x_init=2, y_init=1, sigma_x=2, sigma_y=2, 
                  T_final=500, dt=0.01, lambda_coupling=1/3):
    # Calculate number of steps from T_final and dt
    n_steps = int(T_final / dt)
    
    simulator = QuantumGhostSimulator(x_init=x_init, y_init=y_init, 
                                    sigma_x=sigma_x, sigma_y=sigma_y,
                                    dt=dt, lambda_coupling=lambda_coupling)
    simulator.evolve(n_steps)
    simulator.plot_trajectories()
    return simulator

if __name__ == "__main__":
    simulator = run_simulation()