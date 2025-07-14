import pickle
import os

def save_monitor_state(monitor, filepath="monitor_state.pkl"):
    """
    Save monitor state to pickle file.
    
    Args:
        monitor: NeuronEnsembleMonitor object
        filepath: Path to save file
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(monitor, f)
        print(f"Monitor state saved to {filepath}")
    except Exception as e:
        print(f"Error saving monitor state: {e}")

def load_monitor_state(filepath="monitor_state.pkl"):
    """
    Load monitor state from pickle file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        NeuronEnsembleMonitor object or None if file doesn't exist
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                monitor = pickle.load(f)
            print(f"Loaded monitor state from {filepath}")
            return monitor
        except Exception as e:
            print(f"Error loading monitor state: {e}")
            return None
    else:
        print("No saved monitor state found, initializing fresh monitor.")
        return None  # Caller should create new monitor