import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

class GhostQuantumSystem:
    def __init__(self, n_states=50, lambda_coupling=1/3):
        """
        Initialize the quantum system with ghost-normal oscillator interaction.
        
        Parameters:
        n_states (int): Number of basis states to use in computation
        lambda_coupling (float): Coupling constant λ in the interaction potential
        """
        self.n = n_states
        self.lambda_coupling = lambda_coupling
        
        # Create position and momentum operators in matrix form
        self.setup_operators()
        
        # Setup the Hamiltonian
        self.setup_hamiltonian()
    
    def setup_operators(self):
        """Setup position and momentum operators in the harmonic oscillator basis"""
        # Creation operator elements
        diag_elements = np.sqrt(np.arange(1, self.n))
        self.a_dag = sparse.diags(diag_elements, -1, shape=(self.n, self.n))
        self.a = self.a_dag.T
        
        # Position and momentum operators
        self.x = (self.a + self.a_dag) / np.sqrt(2)
        self.p = 1j * (self.a_dag - self.a) / np.sqrt(2)
        
        # For the ghost oscillator
        self.y = (self.a + self.a_dag) / np.sqrt(2)
        self.p_y = 1j * (self.a_dag - self.a) / np.sqrt(2)
    
    def interaction_potential(self, x, y):
        """
        Compute the interaction potential V_I from the paper.
        V_I = λ[(x² - y² - 1)² + 4x²]^(-1/2)
        """
        return self.lambda_coupling * ((x**2 - y**2 - 1)**2 + 4*x**2)**(-0.5)
    
    def setup_hamiltonian(self):
        """Setup the full Hamiltonian including interaction"""
        # Free Hamiltonian parts
        H0_x = 0.5 * (self.p.dot(self.p) + self.x.dot(self.x))
        H0_y = -0.5 * (self.p_y.dot(self.p_y) + self.y.dot(self.y))
        
        # Interaction part (truncated series expansion)
        x_matrix = self.x.toarray()
        y_matrix = self.y.toarray()
        
        # Expand interaction potential around origin
        V_int = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                # Using perturbative expansion of interaction
                x_val = x_matrix[i, j]
                y_val = y_matrix[i, j]
                if abs(x_val) < 1e-10 and abs(y_val) < 1e-10:
                    continue
                V_int[i, j] = self.interaction_potential(x_val, y_val)
        
        self.H = H0_x + H0_y + sparse.csr_matrix(V_int)
    
    def compute_spectrum(self, n_eigenvalues=10):
        """
        Compute the lowest n_eigenvalues eigenvalues and eigenvectors
        """
        eigenvalues, eigenvectors = eigsh(self.H, k=n_eigenvalues, which='SA')
        return eigenvalues, eigenvectors
    
    def compute_perturbative_correction(self, state_idx, order=2):
        """
        Compute perturbative correction to energy up to given order
        """
        E0 = (state_idx + 0.5)  # Unperturbed energy
        
        # First order correction
        if order >= 1:
            E1 = self.lambda_coupling * np.real(
                np.sum(self.x.dot(self.x).diagonal()) -
                np.sum(self.y.dot(self.y).diagonal())
            )
        
        # Second order correction (if requested)
        E2 = 0
        if order >= 2:
            # Implement second order correction...
            pass
        
        return E0 + E1 + E2 if order >= 2 else E0 + E1 if order >= 1 else E0

def analyze_system(lambda_values=[1/3], n_states=50, n_eigenvalues=10):
    """
    Analyze the quantum system for different coupling constants
    """
    results = []
    
    for lambda_val in lambda_values:
        system = GhostQuantumSystem(n_states=n_states, lambda_coupling=lambda_val)
        
        # Compute exact spectrum
        eigenvalues, eigenvectors = system.compute_spectrum(n_eigenvalues)
        
        # Compute perturbative corrections
        pert_energies = [
            system.compute_perturbative_correction(i, order=2)
            for i in range(n_eigenvalues)
        ]
        
        results.append({
            'lambda': lambda_val,
            'exact_energies': eigenvalues,
            'pert_energies': pert_energies,
            'eigenvectors': eigenvectors
        })
    
    return results

def plot_results(results):
    """
    Plot the results of the analysis
    """
    plt.figure(figsize=(10, 6))
    
    for result in results:
        lambda_val = result['lambda']
        exact = result['exact_energies']
        pert = result['pert_energies']
        
        plt.plot(exact, 'o-', label=f'λ={lambda_val} (exact)')
        plt.plot(pert, 's--', label=f'λ={lambda_val} (perturbative)')
    
    plt.xlabel('State index')
    plt.ylabel('Energy')
    plt.title('Energy Spectrum: Exact vs Perturbative')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    lambda_values = [1/3, 1/4]
    results = analyze_system(lambda_values)
    plot_results(results)