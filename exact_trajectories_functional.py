import numpy as np
from scipy.special import hermite, factorial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

class QuantumGhostSystem:
    def __init__(self, max_n=30, lambda_coupling=1/3):
        """
        Initialize quantum system using harmonic oscillator basis
        
        Parameters:
        max_n: Maximum number of basis states to use
        lambda_coupling: Coupling strength λ
        """
        self.max_n = max_n
        self.lambda_coupling = lambda_coupling
        
        # Create basis functions
        self.setup_basis()
        
        # Compute matrix elements
        self.compute_matrix_elements()
    
    def psi_n(self, x, n):
        """
        Harmonic oscillator eigenfunction for quantum number n
        ψₙ(x) = Nₙ Hₙ(x) exp(-x²/2)
        """
        # Normalization
        N = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
        
        # Hermite polynomial
        H = hermite(n)
        
        return N * H(x) * np.exp(-x**2/2)
    
    def setup_basis(self):
        """Setup the basis functions for both oscillators"""
        self.basis_n = range(self.max_n)
        
        # Store hermite polynomials for efficiency
        self.hermite_polys = [hermite(n) for n in self.basis_n]
        
        # Store normalizations
        self.norms = [1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi)) 
                     for n in self.basis_n]
    
    def coherent_state_coeffs(self, alpha, basis_size=None):
        """
        Compute expansion coefficients for coherent state |α⟩
        in the harmonic oscillator basis
        
        |α⟩ = exp(-|α|²/2) Σₙ (αⁿ/√n!) |n⟩
        """
        if basis_size is None:
            basis_size = self.max_n
            
        coeffs = np.zeros(basis_size, dtype=complex)
        norm = np.exp(-np.abs(alpha)**2/2)
        
        for n in range(basis_size):
            coeffs[n] = norm * alpha**n / np.sqrt(factorial(n))
            
        return coeffs
    
    def compute_matrix_elements(self):
        """
        Compute Hamiltonian matrix elements in the tensor product basis
        ⟨m,n|H|k,l⟩ = ⟨m|p²+x²|k⟩δₙₗ - ⟨n|p²+x²|l⟩δₘₖ + λ⟨m,n|V|k,l⟩
        """
        N = self.max_n
        self.H = np.zeros((N*N, N*N), dtype=complex)
        
        # Free Hamiltonian parts
        for m in range(N):
            for n in range(N):
                for k in range(N):
                    for l in range(N):
                        idx1 = m*N + n
                        idx2 = k*N + l
                        
                        # Normal oscillator
                        if n == l:
                            self.H[idx1,idx2] += (k + 0.5) * (m == k)
                            
                        # Ghost oscillator
                        if m == k:
                            self.H[idx1,idx2] -= (l + 0.5) * (n == l)
        
        # Interaction potential matrix elements
        # This is approximate - using Gaussian quadrature would be more accurate
        x_points = np.polynomial.hermite.hermgauss(50)[0] * np.sqrt(2)
        weights = np.polynomial.hermite.hermgauss(50)[1]
        
        for m in range(N):
            for n in range(N):
                for k in range(N):
                    for l in range(N):
                        idx1 = m*N + n
                        idx2 = k*N + l
                        
                        # Compute ⟨m,n|V|k,l⟩ using quadrature
                        integrand = 0
                        for x in x_points:
                            for y in x_points:
                                psi_m = self.psi_n(x, m)
                                psi_n = self.psi_n(y, n)
                                psi_k = self.psi_n(x, k)
                                psi_l = self.psi_n(y, l)
                                
                                V = self.lambda_coupling / np.sqrt((x**2 - y**2 - 1)**2 + 4*x**2)
                                integrand += V * psi_m * psi_n * psi_k * psi_l
                        
                        self.H[idx1,idx2] += integrand
    
    def evolve_state(self, psi0, times):
        """
        Time evolve an initial state
        |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        """
        # Diagonalize Hamiltonian
        evals, evecs = np.linalg.eigh(self.H)
        
        # Project initial state onto eigenvectors
        coeffs = evecs.conj().T @ psi0
        
        # Time evolve
        states = []
        for t in times:
            # Evolve in energy basis
            evolved_coeffs = coeffs * np.exp(-1j * evals * t)
            
            # Transform back to computational basis
            psi_t = evecs @ evolved_coeffs
            states.append(psi_t)
            
        return np.array(states)
    
    def compute_expectation_values(self, states):
        """Compute expectation values of x and p"""
        N = self.max_n
        x_exp = np.zeros(len(states), dtype=complex)
        p_exp = np.zeros(len(states), dtype=complex)
        y_exp = np.zeros(len(states), dtype=complex)
        py_exp = np.zeros(len(states), dtype=complex)
        
        # Position operator matrix elements
        x_mat = np.zeros((N, N), dtype=complex)
        p_mat = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            for j in range(N):
                if j == i+1:
                    x_mat[i,j] = np.sqrt((i+1)/2)
                if j == i-1:
                    x_mat[i,j] = np.sqrt(i/2)
                    
                if j == i+1:
                    p_mat[i,j] = 1j*np.sqrt((i+1)/2)
                if j == i-1:
                    p_mat[i,j] = -1j*np.sqrt(i/2)
        
        # Compute expectations
        for i, state in enumerate(states):
            state = state.reshape(N, N)
            
            # x expectation
            x_exp[i] = np.trace(state.conj().T @ x_mat @ state)
            
            # p expectation
            p_exp[i] = np.trace(state.conj().T @ p_mat @ state)
            
            # y and py expectations (note the ghost nature)
            y_exp[i] = -np.trace(state @ x_mat @ state.conj().T)
            py_exp[i] = -np.trace(state @ p_mat @ state.conj().T)
        
        return np.real(x_exp), np.real(y_exp), np.real(p_exp), np.real(py_exp)

def create_and_evolve_coherent_state(x0=2, y0=1, px0=0, py0=0, lambda_val=1/3):
    """Create and evolve a coherent state with given initial conditions"""
    # Create system
    system = QuantumGhostSystem(max_n=30, lambda_coupling=lambda_val)
    
    # Create initial coherent state
    alpha_x = (x0 + 1j*px0)/np.sqrt(2)
    alpha_y = (y0 + 1j*py0)/np.sqrt(2)
    
    # Get coefficients in tensor product basis
    coeffs_x = system.coherent_state_coeffs(alpha_x)
    coeffs_y = system.coherent_state_coeffs(alpha_y)
    
    # Create full initial state
    psi0 = np.kron(coeffs_x, coeffs_y)
    
    # Time evolve
    times = np.linspace(0, 500, 1000)
    states = system.evolve_state(psi0, times)
    
    # Compute expectations
    x_exp, y_exp, px_exp, py_exp = system.compute_expectation_values(states)
    
    return times, x_exp, y_exp, px_exp, py_exp

def plot_quantum_trajectories(times, x_exp, y_exp, px_exp, py_exp):
    """Plot quantum trajectories with time coloring"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create points for coloring
    points_xy = np.array([x_exp, y_exp]).T.reshape(-1, 1, 2)
    segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
    
    points_pp = np.array([px_exp, py_exp]).T.reshape(-1, 1, 2)
    segments_pp = np.concatenate([points_pp[:-1], points_pp[1:]], axis=1)
    
    # Create color normalization
    norm = Normalize(vmin=times.min(), vmax=times.max())
    
    # Plot xy trajectory
    lc1 = LineCollection(segments_xy, cmap='viridis', norm=norm)
    lc1.set_array(times[:-1])
    ax1.add_collection(lc1)
    ax1.set_xlim(x_exp.min()-0.1, x_exp.max()+0.1)
    ax1.set_ylim(y_exp.min()-0.1, y_exp.max()+0.1)
    ax1.set_xlabel('⟨x⟩')
    ax1.set_ylabel('⟨y⟩')
    ax1.set_title('Position Space Quantum Trajectory')
    plt.colorbar(lc1, ax=ax1, label='time')
    ax1.grid(True)
    
    # Plot momentum space trajectory
    lc2 = LineCollection(segments_pp, cmap='viridis', norm=norm)
    lc2.set_array(times[:-1])
    ax2.add_collection(lc2)
    ax2.set_xlim(px_exp.min()-0.1, px_exp.max()+0.1)
    ax2.set_ylim(py_exp.min()-0.1, py_exp.max()+0.1)
    ax2.set_xlabel('⟨px⟩')
    ax2.set_ylabel('⟨py⟩')
    ax2.set_title('Momentum Space Quantum Trajectory')
    plt.colorbar(lc2, ax=ax2, label='time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Evolve coherent state with paper's initial conditions
    times, x_exp, y_exp, px_exp, py_exp = create_and_evolve_coherent_state(
        x0=2, y0=1, px0=0, py0=0, lambda_val=1/3
    )
    
    # Plot trajectories
    plot_quantum_trajectories(times, x_exp, y_exp, px_exp, py_exp)