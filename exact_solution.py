import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

class ExactGhostQuantumSystem:
    def __init__(self, n_points=100, x_max=10, lambda_coupling=1/3):
        """
        Initialize the quantum system with ghost-normal oscillator interaction.
        Accounts for the indefinite metric of the ghost system.
        
        Parameters:
        n_points (int): Number of grid points for position space
        x_max (float): Maximum position value (grid goes from -x_max to x_max)
        lambda_coupling (float): Coupling constant λ in the interaction potential
        """
        self.n = n_points
        self.lambda_coupling = lambda_coupling
        
        # Setup spatial grid
        self.x_max = x_max
        self.dx = 2 * x_max / (n_points - 1)
        self.x_grid = np.linspace(-x_max, x_max, n_points)
        
        # Setup momentum grid
        self.dp = 2 * np.pi / (2 * x_max)
        self.p_max = self.dp * (n_points // 2)
        self.p_grid = np.fft.fftfreq(n_points, self.dx/(2*np.pi))
        
        # Setup operators and Hamiltonian
        self.setup_operators()
        self.setup_hamiltonian()
        self.setup_metric()
    
    def setup_operators(self):
        """Setup position and momentum operators directly"""
        # Position operators as diagonal matrices
        x_diag = sparse.diags(self.x_grid)
        I = sparse.eye(self.n)
        
        # Full position operators in tensor product space
        self.x = sparse.kron(x_diag, I)
        self.y = sparse.kron(I, x_diag)
        
        # Momentum operator in position space
        # Using periodic boundary conditions for better behavior
        k = 2 * np.pi * np.fft.fftfreq(self.n, self.dx)
        p_diag = sparse.diags(k)
        
        # Full momentum operators in tensor product space
        self.p = sparse.kron(p_diag, I)
        self.p_y = sparse.kron(I, p_diag)
    
    def setup_metric(self):
        """Setup the indefinite metric for the ghost system"""
        # Create metric operator η that gives the correct inner product
        # For ghost system, we need different signs for normal and ghost parts
        I = sparse.eye(self.n)
        self.eta = sparse.kron(I, -I)  # minus sign for ghost part
    
    def setup_hamiltonian(self):
        """Construct the full Hamiltonian in matrix form"""
        # Free Hamiltonian parts with correct signs
        H0_x = 0.5 * (self.p @ self.p + self.x @ self.x)
        H0_y = -0.5 * (self.p_y @ self.p_y + self.y @ self.y)  # ghost part
        
        # Interaction potential from the paper
        x_matrix = self.x.toarray()
        y_matrix = self.y.toarray()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        V_int = self.lambda_coupling * ((x_matrix**2 - y_matrix**2 - 1)**2 + 4*x_matrix**2 + epsilon)**(-0.5)
        
        # Full Hamiltonian
        self.H = H0_x + H0_y + sparse.csr_matrix(V_int)
    
    def compute_eigenstates(self, n_states=20):
        """
        Compute eigenvalues and eigenvectors using the indefinite metric
        
        Returns:
        eigenvalues: Complex array of energy eigenvalues
        eigenvectors: Matrix of eigenvectors
        """
        # Use eigs instead of eigsh since the system isn't positive definite
        # We need to solve the generalized eigenvalue problem H ψ = E η ψ
        eigenvalues, eigenvectors = eigs(self.H, k=n_states, M=self.eta, 
                                       which='SR')
        
        # Sort by real part of eigenvalues
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def compute_norm(self, state):
        """
        Compute the norm using the indefinite metric
        
        Parameters:
        state: State vector to compute norm for
        
        Returns:
        complex norm value (should be real for physical states)
        """
        return state.conj().T @ (self.eta @ state)
    
    def time_evolve_state(self, psi0, times):
        """
        Evolve an initial state through time using the indefinite metric
        
        Parameters:
        psi0: Initial wavefunction
        times: Array of times to evaluate at
        
        Returns:
        x_expect: Expectation values of x
        y_expect: Expectation values of y
        norms: Time evolution of the norm
        """
        # First get eigenstates
        evals, evecs = self.compute_eigenstates()
        
        # Project initial state onto eigenstates with correct metric
        coeffs = evecs.conj().T @ (self.eta @ psi0)
        
        # Time evolution
        x_expect = np.zeros(len(times))
        y_expect = np.zeros(len(times))
        norms = np.zeros(len(times))
        
        for i, t in enumerate(times):
            # Evolve state
            psi_t = evecs @ (coeffs * np.exp(-1j * evals * t))
            
            # Compute expectation values with correct metric
            x_expect[i] = np.real(psi_t.conj().T @ (self.eta @ (self.x @ psi_t)))
            y_expect[i] = np.real(psi_t.conj().T @ (self.eta @ (self.y @ psi_t)))
            norms[i] = np.real(self.compute_norm(psi_t))
        
        return x_expect, y_expect, norms
    
    def analyze_stability(self, n_states=20):
        """
        Analyze stability of the quantum system with indefinite metric
        
        Returns:
        Dictionary containing stability metrics
        """
        # Compute spectrum
        eigenvalues, eigenvectors = self.compute_eigenstates(n_states)
        
        # Compute norms of eigenstates
        norms = np.array([np.real(self.compute_norm(evec)) 
                         for evec in eigenvectors.T])
        
        # Compute expectation values with correct metric
        radius_exp = np.zeros(n_states)
        for i in range(n_states):
            psi = eigenvectors[:, i]
            x2 = np.real(psi.conj().T @ (self.eta @ (self.x @ self.x @ psi)))
            y2 = np.real(psi.conj().T @ (self.eta @ (self.y @ self.y @ psi)))
            radius_exp[i] = np.sqrt(abs(x2 + y2))
        
        return {
            'eigenvalues': eigenvalues,
            'norms': norms,
            'radius_exp': radius_exp
        }

def analyze_quantum_system(lambda_values=[1/3], n_points=100, x_max=10):
    """
    Analyze quantum stability for different coupling strengths
    """
    results = []
    
    for lambda_val in lambda_values:
        print(f"Analyzing λ = {lambda_val}")
        system = ExactGhostQuantumSystem(n_points=n_points, x_max=x_max, 
                                       lambda_coupling=lambda_val)
        
        # Analyze stability
        stability_data = system.analyze_stability()
        
        # Store results
        results.append({
            'lambda': lambda_val,
            'stability_data': stability_data
        })
    
    return results

def plot_results(results):
    """
    Visualize the quantum analysis including indefinite metric effects
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for result in results:
        lambda_val = result['lambda']
        data = result['stability_data']
        
        # Plot complex eigenvalues
        ax1.plot(np.real(data['eigenvalues']), np.imag(data['eigenvalues']), 
                'o', label=f'λ={lambda_val}')
        
        # Plot norms
        ax2.plot(np.real(data['norms']), 'o-', label=f'λ={lambda_val}')
        
        # Plot radius expectation vs energy
        ax3.plot(np.real(data['eigenvalues']), data['radius_exp'], 'o-', 
                label=f'λ={lambda_val}')
        
        # Plot energy spacing
        if len(data['eigenvalues']) > 1:
            spacing = np.diff(np.real(data['eigenvalues']))
            ax4.plot(spacing, 'o-', label=f'λ={lambda_val}')
    
    ax1.set_xlabel('Re(E)')
    ax1.set_ylabel('Im(E)')
    ax1.set_title('Complex Energy Spectrum')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('State index')
    ax2.set_ylabel('Norm')
    ax2.set_title('State Norms')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('Re(E)')
    ax3.set_ylabel('$\\sqrt{|\\langle x^2 + y^2 \\rangle|}$')
    ax3.set_title('Spatial Extent of Eigenstates')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_xlabel('State index')
    ax4.set_ylabel('Energy spacing')
    ax4.set_title('Level Spacing')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    lambda_values = [1/3, 1/4]
    results = analyze_quantum_system(lambda_values, n_points=100, x_max=10)
    plot_results(results)