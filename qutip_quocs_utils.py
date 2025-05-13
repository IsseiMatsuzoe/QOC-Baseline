import numpy as np
import qutip as qt
from typing import Union, List, Dict, Tuple, Any
from scipy.linalg import expm  # SciPyのexpm関数を使用

def qobj_to_numpy(op: Union[qt.Qobj, np.ndarray]) -> np.ndarray:
    """
    Convert a QuTiP Qobj to a numpy array if needed
    
    Args:
        op: QuTiP Qobj or numpy array
        
    Returns:
        numpy ndarray representation
    """
    if isinstance(op, qt.Qobj):
        return op.full()
    return op

def numpy_to_qobj(arr: np.ndarray, dims=None) -> qt.Qobj:
    """
    Convert a numpy array to a QuTiP Qobj
    
    Args:
        arr: numpy array
        dims: dimensions of the resulting Qobj (optional)
        
    Returns:
        QuTiP Qobj
    """
    return qt.Qobj(arr, dims=dims)

def qobj_matmul(a: Union[qt.Qobj, np.ndarray], 
                b: Union[qt.Qobj, np.ndarray]) -> np.ndarray:
    """
    Matrix multiplication that works with mixed Qobj and numpy arrays
    
    Args:
        a: First operator (Qobj or ndarray)
        b: Second operator (Qobj or ndarray)
        
    Returns:
        Result as numpy array
    """
    return qobj_to_numpy(a) @ qobj_to_numpy(b)

def get_electron_spin_operators(as_numpy: bool = True) -> Dict[str, Any]:
    """
    Get the electron spin operators (Pauli matrices)
    
    Args:
        as_numpy: If True, return numpy arrays, otherwise return Qobj
        
    Returns:
        Dictionary with spin operators: 'sx', 'sy', 'sz', 'si'
    """
    operators = {
        'sx': qt.sigmax(),
        'sy': qt.sigmay(),
        'sz': qt.sigmaz(),
        'si': qt.identity(2)
    }
    
    if as_numpy:
        for key in operators:
            operators[key] = qobj_to_numpy(operators[key])
    
    return operators

class QutipToNumPyWrapper:
    """
    Wrapper class for handling conversions between QuTiP and NumPy 
    for QuOCS optimization
    """
    
    def __init__(self, initial_state: Union[qt.Qobj, np.ndarray], 
                 target_state: Union[qt.Qobj, np.ndarray],
                 hamiltonian_drift: Union[qt.Qobj, np.ndarray],
                 hamiltonian_controls: List[Union[qt.Qobj, np.ndarray]],
                 is_maximization: bool = True):
        """
        Initialize the wrapper with quantum system components
        
        Args:
            initial_state: Initial quantum state (density matrix)
            target_state: Target quantum state (density matrix)
            hamiltonian_drift: Drift Hamiltonian
            hamiltonian_controls: List of control Hamiltonians
            is_maximization: Whether the optimization is maximization (True) or minimization (False)
        """
        # Convert all inputs to numpy arrays
        self.initial_state = qobj_to_numpy(initial_state)
        self.target_state = qobj_to_numpy(target_state)
        self.drift = qobj_to_numpy(hamiltonian_drift)
        self.controls = [qobj_to_numpy(h) for h in hamiltonian_controls]
        self.is_maximization = is_maximization
        
        # Keep the original QuTiP objects if needed
        self.initial_state_qobj = initial_state if isinstance(initial_state, qt.Qobj) else None
        self.target_state_qobj = target_state if isinstance(target_state, qt.Qobj) else None
        
        # Dimensions
        self.dim = self.initial_state.shape[0]
    
    def evolve_state(self, pulses: List[np.ndarray], dt: float) -> np.ndarray:
        """
        Evolve the initial state according to the given pulses
        
        Args:
            pulses: List of control amplitudes
            dt: Time step
            
        Returns:
            Final state as numpy array
        """
        state = self.initial_state.copy()
        steps = pulses[0].shape[0]
        
        for t in range(steps):
            # Construct total Hamiltonian for this time step
            h_total = self.drift.copy()
            for i, ctrl_ham in enumerate(self.controls):
                if i < len(pulses):
                    h_total += pulses[i][t] * ctrl_ham
            
            # Evolve with this Hamiltonian for dt
            U = self._get_propagator(h_total, dt)
            
            # Apply propagator to state
            if state.ndim == 2:  # Density matrix
                state = U @ state @ U.conj().T
            else:  # State vector
                state = U @ state
        
        return state
    
    def _get_propagator(self, hamiltonian: np.ndarray, dt: float) -> np.ndarray:
        """
        Get the propagator for a given Hamiltonian and time step
        
        Args:
            hamiltonian: Hamiltonian as numpy array
            dt: Time step
            
        Returns:
            Propagator as numpy array
        """
        # U = exp(-i*H*dt)
        return expm(-1j * hamiltonian * dt)
    
    def calculate_fidelity(self, final_state: np.ndarray) -> float:
        """
        Calculate the fidelity between the final state and the target state
        
        Args:
            final_state: Final state after evolution
            
        Returns:
            Fidelity value
        """
        # Convert back to QuTiP for fidelity calculation
        final_qobj = qt.Qobj(final_state)
        target_qobj = self.target_state_qobj if self.target_state_qobj else qt.Qobj(self.target_state)
        
        if final_qobj.isoper and target_qobj.isoper:
            fidelity = qt.fidelity(final_qobj, target_qobj)
        else:
            # For pure states: |<ψ|φ>|²
            fidelity = np.abs(np.vdot(final_state, self.target_state))**2
            
        return fidelity if self.is_maximization else 1.0 - fidelity 