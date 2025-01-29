import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.colors as colors

class QuantumGhostSimulator:
    def __init__(self, nx=128, ny=128, dx=0.2, dy=0.2, dt=0.01, 
                 x_init=2, y_init=1, sigma_x=2, sigma_y=2, lambda_coupling=1/3):
        # Grid parameters for position and momentum space
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt = dt
        self.lambda_coupling = lambda_coupling
        
        # Create spatial grids centered around zero
        self.x = (np.arange(nx) - nx//2) * dx
        self.y = (np.arange(ny) - ny//2) * dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Create momentum space grids using FFT frequencies
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Initialize wave function and storage arrays
        self.psi = self._create_initial_state(x_init, y_init, sigma_x, sigma_y)
        self.psi_history = []
        self.x_means = []
        self.y_means = []
        self.px_means = []
        self.py_means = []
        self.times = []
        
        # Initialize the propagators for time evolution
        self._init_propagators()

    def _create_initial_state(self, x_init, y_init, sigma_x, sigma_y):
        """Create initial Gaussian wave packet in position space"""
        # Initialize complex wave function array
        psi = np.zeros((self.nx, self.ny), dtype=np.complex128)
        
        # Create Gaussian envelope
        gaussian = np.exp(-(self.X - x_init)**2/(2*sigma_x**2) 
                         -(self.Y - y_init)**2/(2*sigma_y**2))
        
        # Set real part to Gaussian (no initial momentum)
        psi.real = gaussian
        
        # Normalize the wave function
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dy)
        return psi / norm

    def _compute_potential(self):
        """Compute the ghost interaction potential V_I from the paper"""
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        denominator = (self.X**2 - self.Y**2 - 1)**2 + 4*self.X**2 + epsilon
        return self.lambda_coupling * denominator**(-0.5)

    def _init_propagators(self):
        """Initialize the split-operator propagators for time evolution"""
        # Kinetic energy propagator in momentum space
        # Note the ghost term (-0.5 * self.KY**2) has negative sign
        self.kinetic_propagator = np.exp(-1j * self.dt * 
                                       (0.5 * self.KX**2 - 0.5 * self.KY**2))
        
        # Potential energy propagator in position space
        # Includes harmonic potential and interaction term
        V = 0.5*(self.X**2 - self.Y**2) + self._compute_potential()
        self.potential_propagator = np.exp(-1j * self.dt * V / 2)  # Half step

    def compute_expectation_values(self):
        """Compute expectation values of position and momentum operators"""
        # Calculate probability density
        prob_density = np.abs(self.psi)**2
        
        # Position expectations (x and y)
        x_mean = np.sum(self.X * prob_density) * self.dx * self.dy
        y_mean = np.sum(self.Y * prob_density) * self.dx * self.dy
        
        # Transform to momentum space
        psi_k = fftshift(fft2(self.psi))
        k_prob_density = np.abs(psi_k)**2
        
        # Momentum space grid spacing
        dk_x = 2*np.pi/(self.nx*self.dx)
        dk_y = 2*np.pi/(self.ny*self.dy)
        
        # Momentum expectations (px and py)
        px_mean = np.sum(self.KX * k_prob_density) * dk_x * dk_y
        py_mean = np.sum(self.KY * k_prob_density) * dk_x * dk_y
        
        return x_mean, y_mean, px_mean, py_mean

    def step(self):
        """Perform one time step using the split-operator method"""
        # First half step with potential
        self.psi *= self.potential_propagator
        
        # Full step with kinetic energy in momentum space
        psi_k = fft2(self.psi)
        psi_k *= self.kinetic_propagator
        self.psi = ifft2(psi_k)
        
        # Second half step with potential
        self.psi *= self.potential_propagator
        
        # Occasionally renormalize to prevent numerical drift
        if np.random.rand() < 0.01:  # 1% chance each step
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)
            self.psi /= norm

    def store_frame(self):
        """Store current state for animation"""
        self.psi_history.append(np.copy(self.psi))
        x_mean, y_mean, px_mean, py_mean = self.compute_expectation_values()
        self.x_means.append(x_mean)
        self.y_means.append(y_mean)
        self.px_means.append(px_mean)
        self.py_means.append(py_mean)
        self.times.append(len(self.times) * self.dt)

    def evolve(self, n_steps, store_every=10):
        """Evolve the system and store frames for animation"""
        for i in range(n_steps):
            self.step()
            if i % store_every == 0:
                self.store_frame()

    def create_animation(self, filename='quantum_ghost', fps=30):
        """Create and save animation of probability density and trajectories"""
        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # Create axes for density plot and trajectories
        ax_density = plt.subplot(gs[0])
        ax_pos = plt.subplot(gs[1])
        ax_mom = plt.subplot(gs[2])
        
        # Initialize probability density plot with logarithmic scale
        density_plot = ax_density.imshow(np.abs(self.psi_history[0])**2, 
                                       extent=[self.y[0], self.y[-1], 
                                              self.x[0], self.x[-1]],
                                       cmap='viridis',
                                       norm=colors.LogNorm())
        plt.colorbar(density_plot, ax=ax_density, label='Probability Density')
        
        # Initialize trajectory plots
        pos_line, = ax_pos.plot([], [], 'b-', lw=1)
        pos_point, = ax_pos.plot([], [], 'ro')
        mom_line, = ax_mom.plot([], [], 'b-', lw=1)
        mom_point, = ax_mom.plot([], [], 'ro')
        
        # Set labels and titles
        ax_density.set_title('Probability Density')
        ax_density.set_xlabel('y')
        ax_density.set_ylabel('x')
        
        ax_pos.set_title('Position Space Trajectory')
        ax_pos.set_xlabel('x')
        ax_pos.set_ylabel('y')
        ax_pos.grid(True)
        
        ax_mom.set_title('Momentum Space Trajectory')
        ax_mom.set_xlabel('px')
        ax_mom.set_ylabel('py')
        ax_mom.grid(True)
        
        # Set axis limits based on data
        ax_pos.set_xlim(min(self.x_means) - 0.5, max(self.x_means) + 0.5)
        ax_pos.set_ylim(min(self.y_means) - 0.5, max(self.y_means) + 0.5)
        ax_mom.set_xlim(min(self.px_means) - 0.5, max(self.px_means) + 0.5)
        ax_mom.set_ylim(min(self.py_means) - 0.5, max(self.py_means) + 0.5)
        
        def update(frame):
            """Update function for animation"""
            # Update probability density plot
            density_plot.set_array(np.abs(self.psi_history[frame])**2)
            
            # Update position space trajectory
            pos_line.set_data(self.x_means[:frame+1], self.y_means[:frame+1])
            pos_point.set_data([self.x_means[frame]], [self.y_means[frame]])
            
            # Update momentum space trajectory
            mom_line.set_data(self.px_means[:frame+1], self.py_means[:frame+1])
            mom_point.set_data([self.px_means[frame]], [self.py_means[frame]])
            
            return density_plot, pos_line, pos_point, mom_line, mom_point

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.psi_history),
                           interval=1000/fps, blit=True)
        
        # Save as GIF
        anim.save(f'{filename}.gif', writer=PillowWriter(fps=fps))
        
        # Try to save as MP4 if ffmpeg is available
        try:
            anim.save(f'{filename}.mp4', writer=FFMpegWriter(fps=fps))
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg might not be installed): {e}")
        
        plt.close()
        return anim

def run_simulation(x_init=2, y_init=1, sigma_x=2, sigma_y=2, 
                  T_final=50, dt=0.01, lambda_coupling=1/3,
                  store_every=10):
    """Run simulation and create animation"""
    # Calculate number of steps
    n_steps = int(T_final / dt)
    
    # Create simulator and run evolution
    simulator = QuantumGhostSimulator(x_init=x_init, y_init=y_init, 
                                    sigma_x=sigma_x, sigma_y=sigma_y,
                                    dt=dt, lambda_coupling=lambda_coupling)
    simulator.evolve(n_steps, store_every=store_every)
    
    # Create and save animation
    anim = simulator.create_animation()
    return simulator, anim

if __name__ == "__main__":
    simulator, anim = run_simulation(T_final=500)