import pickle
import os

def save_neuron_state(neurons, filepath="network_state.pkl"):
    """
    Save neuron network state to pickle file.
    
    Args:
        neurons: List of Neuron objects
        filepath: Path to save file
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(neurons, f)
        print(f"Neuron state saved to {filepath}")
    except Exception as e:
        print(f"Error saving neuron state: {e}")

def load_neuron_state(filepath="network_state.pkl"):
    """
    Load neuron network state from pickle file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        List of Neuron objects or None if file doesn't exist
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                neurons = pickle.load(f)
            print(f"Loaded neuron state from {filepath}")
            return neurons
        except Exception as e:
            print(f"Error loading neuron state: {e}")
            return None
    else:
        print("No saved neuron state found, initializing random network.")
        return None  # Caller should generate fresh neurons