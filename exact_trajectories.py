import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class ExactGhostQuantumSystem:
    def __init__(self, n_points=100, x_max=10, lambda_coupling=1/3):
        self.n = n_points
        self.lambda_coupling = lambda_coupling
        
        # Setup spatial grid
        self.x_max = x_max
        self.dx = 2 * x_max / (n_points - 1)
        self.x_grid = np.linspace(-x_max, x_max, n_points)
        
        # Setup momentum grid
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx)
        
        # Setup operators and Hamiltonian
        self.setup_operators()
        self.setup_hamiltonian()
    
    def setup_operators(self):
        """Setup operators in real space"""
        # Single particle operators
        self.x_single = sparse.diags(self.x_grid)
        self.p_single = sparse.diags(self.k)
        I = sparse.eye(self.n)
        
        # Two-particle operators
        self.x = sparse.kron(self.x_single, I)
        self.y = sparse.kron(I, self.x_single)
        self.p_x = sparse.kron(self.p_single, I)
        self.p_y = sparse.kron(I, self.p_single)
    
    def setup_hamiltonian(self):
        """Setup the full Hamiltonian"""
        # Free parts
        H0_x = 0.5 * (self.p_x.dot(self.p_x) + self.x.dot(self.x))
        H0_y = -0.5 * (self.p_y.dot(self.p_y) + self.y.dot(self.y))
        
        # Interaction potential
        x_mat = self.x.toarray()
        y_mat = self.y.toarray()
        epsilon = 1e-10  # to avoid division by zero
        V_int = self.lambda_coupling * ((x_mat**2 - y_mat**2 - 1)**2 + 4*x_mat**2 + epsilon)**(-0.5)
        
        self.H = H0_x + H0_y + sparse.csr_matrix(V_int)
    
    def gaussian_wavepacket(self, x0, p0, sigma=0.5):
        """Create a 1D Gaussian wavepacket"""
        psi = np.exp(-(self.x_grid - x0)**2/(4*sigma**2) + 1j*p0*self.x_grid)
        return psi / np.sqrt(np.sum(np.abs(psi)**2))
    
    def create_initial_state(self, x0, y0, px0, py0):
        """Create initial two-particle state"""
        psi_x = self.gaussian_wavepacket(x0, px0)
        psi_y = self.gaussian_wavepacket(y0, py0)
        return np.kron(psi_x, psi_y)
    
    def time_evolve(self, psi0, times):
        """Time evolve the initial state and compute expectation values"""
        # Get eigenvalues and eigenvectors
        n_eigs = min(100, self.n**2 - 2)  # Use more eigenstates
        evals, evecs = eigs(self.H, k=n_eigs, which='SR')
        
        # Sort by real part
        idx = np.argsort(np.real(evals))
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Project initial state
        coeffs = evecs.conj().T @ psi0
        
        # Arrays for expectation values
        x_exp = np.zeros(len(times))
        y_exp = np.zeros(len(times))
        px_exp = np.zeros(len(times))
        py_exp = np.zeros(len(times))
        
        # Time evolution
        for i, t in enumerate(times):
            # Evolve state
            psi_t = evecs @ (coeffs * np.exp(-1j * evals * t))
            
            # Compute expectation values
            x_exp[i] = np.real(psi_t.conj() @ (self.x @ psi_t))
            y_exp[i] = np.real(psi_t.conj() @ (self.y @ psi_t))
            px_exp[i] = np.real(psi_t.conj() @ (self.p_x @ psi_t))
            py_exp[i] = np.real(psi_t.conj() @ (self.p_y @ psi_t))
            
            # Print progress periodically
            if i % 100 == 0:
                print(f"Time step {i}/{len(times)}")
                print(f"Current position: ({x_exp[i]:.3f}, {y_exp[i]:.3f})")
                print(f"Current momentum: ({px_exp[i]:.3f}, {py_exp[i]:.3f})")
        
        return x_exp, y_exp, px_exp, py_exp

def plot_trajectory_comparison(x, y, px, py, times):
    """Plot trajectories with continuous lines and color coding"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create smooth color gradient
    norm = Normalize(vmin=times.min(), vmax=times.max())
    
    # Plot position space trajectory
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc1 = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc1.set_array(times[:-1])
    ax1.add_collection(lc1)
    ax1.set_xlim(x.min()-0.5, x.max()+0.5)
    ax1.set_ylim(y.min()-0.5, y.max()+0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Position Space Trajectory')
    plt.colorbar(lc1, ax=ax1, label='time')
    ax1.grid(True)
    
    # Plot momentum space trajectory
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc2 = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc2.set_array(times[:-1])
    ax2.add_collection(lc2)
    ax2.set_xlim(px.min()-0.5, px.max()+0.5)
    ax2.set_ylim(py.min()-0.5, py.max()+0.5)
    ax2.set_xlabel('px')
    ax2.set_ylabel('py')
    ax2.set_title('Momentum Space Trajectory')
    plt.colorbar(lc2, ax=ax2, label='time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_paper_trajectory():
    """Analyze the trajectory from the paper"""
    # Initialize system
    system = ExactGhostQuantumSystem(n_points=100, x_max=10, lambda_coupling=1/3)
    
    # Create initial state matching paper
    psi0 = system.create_initial_state(x0=2, y0=1, px0=0, py0=0)
    
    # Time evolution with more points
    times = np.linspace(0, 500, 1000)
    print("Starting time evolution...")
    x_exp, y_exp, px_exp, py_exp = system.time_evolve(psi0, times)
    
    # Plot trajectories
    print("Plotting trajectories...")
    plot_trajectory_comparison(x_exp, y_exp, px_exp, py_exp, times)
    
    # Print stability metrics
    r = np.sqrt(x_exp**2 + y_exp**2)
    p = np.sqrt(px_exp**2 + py_exp**2)
    print("\nStability Analysis:")
    print(f"Maximum radius: {np.max(r):.3f}")
    print(f"Maximum momentum: {np.max(p):.3f}")
    print(f"Final radius: {r[-1]:.3f}")
    print(f"Energy conservation (std/mean): {np.std(r)/np.mean(r):.3e}")

if __name__ == "__main__":
    analyze_paper_trajectory()