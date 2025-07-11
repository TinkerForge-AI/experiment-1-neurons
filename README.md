# Bio-Inspired Neural System: Green Detection Experiment

## Project Overview
This project implements a bio-inspired neural network for color detection, specifically focusing on green pixel detection. The system challenges traditional machine learning approaches by incorporating biological principles like neighbor communication, specialty emergence, and goal-driven activation.

## Key Features
- **Custom Neuron Class**: Bio-inspired neurons with specialty values, neighbor communication, and dynamic thresholds
- **Ensemble Monitoring**: Clusters neurons by specialty and tracks activation patterns
- **State Persistence**: Save/load network and monitor states for iterative experiments
- **Comprehensive Logging**: Daily logs with experiment summaries and timestamps
- **Extensible Architecture**: Ready for expansion to more complex scenarios

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Basic Execution
```bash
python experiment_runner.py
```

### Expected Output
The system will:
1. Load or initialize a neural network (10 neurons by default)
2. Process a green pixel input (0, 255, 0)
3. Display ensemble activity summary
4. Save states for next run
5. Log results to `logs/YYYY-MM-DD.json`

## Architecture

### Core Components

#### Neuron (`neuron.py`)
- **Specialty**: Emergent property that develops over time
- **Activation**: Dynamic threshold based on goal alignment
- **Communication**: Excitation/inhibition messages with neighbors
- **Learning**: Adapt specialty based on successful activations

#### NeuronEnsembleMonitor (`neuron_ensemble_monitor.py`)
- **Clustering**: Groups neurons by specialty using K-means
- **Activity Tracking**: Logs activation patterns and correlations
- **Human-Readable Mapping**: Converts specialties to interpretable labels
- **Visualization**: Summary reports and analysis

#### State Management (`state_mgr/`)
- **Persistence**: Save/load network and monitor states
- **Error Handling**: Robust file operations
- **Incremental Learning**: Build on previous experiments

#### Data Loading (`mappings/`)
- **Pixel Processing**: Load and normalize RGB values
- **Component Extraction**: Isolate specific color channels
- **File Support**: Load from .npy files for complex data

## How It Works

1. **Initialization**: Create or load neural network with random specialties
2. **Goal Setting**: All neurons receive the detection goal ("detect_green")
3. **Input Processing**: Green pixel (0, 255, 0) is normalized and passed as signal
4. **Activation**: Neurons evaluate activation based on:
   - Input signal strength
   - Goal alignment with specialty
   - Excitation/inhibition from neighbors
5. **Communication**: Active neurons send messages to neighbors
6. **Analysis**: Monitor clusters neurons and analyzes patterns
7. **Persistence**: Save state and log results for next iteration

## Biological Inspiration

### Human Vision vs. Traditional ML
- **Traditional ML**: Each pixel gets individual weight, processed independently
- **Human Vision**: Contextual processing, attention mechanisms, hierarchical feature extraction
- **This System**: Neurons communicate, form specialties, and use goal-driven activation

### Key Innovations
- **Neighbor Communication**: Neurons influence each other (excitation/inhibition)
- **Dynamic Thresholds**: Activation depends on goal alignment
- **Specialty Emergence**: Neurons develop preferences through experience
- **Context Awareness**: Network state influences individual neuron behavior

## Next Steps

### Immediate Enhancements
- Add more input types (RGB images, different colors)
- Implement attention mechanisms (focus on specific regions)
- Develop hierarchical processing (edge→shape→object detection)

### Advanced Experiments
- **Feedback Loops**: Top-down signals to interpret ambiguous data
- **Sparse Coding**: Encourage sparse, redundant activations
- **Sequential Processing**: Simulate saccades and attention shifts
- **Plasticity**: Dynamic network topology changes

### Research Directions
- Compare performance with traditional CNNs
- Test robustness to noise and occlusion
- Analyze emergence of feature detectors
- Study scaling properties with larger networks

## File Structure
```
experiment_runner.py          # Main execution script
neuron.py                    # Core neuron implementation
neuron_ensemble_monitor.py   # Network monitoring and analysis
requirements.txt             # Python dependencies
logs/                       # Daily experiment logs
state_mgr/                  # State persistence modules
mappings/                   # Data loading and processing
assets/                     # Test data and resources
```

## Logging
- **Daily Files**: `logs/YYYY-MM-DD.json`
- **Append Mode**: Multiple experiments per day
- **Rich Data**: Timestamps, activations, specialties, summaries
- **Analysis Ready**: JSON format for easy processing

## Contributing
This project is designed for iterative experimentation. Key areas for contribution:
- Novel activation functions
- Alternative clustering methods
- Visualization improvements
- Performance benchmarking
- Biological accuracy enhancements

## License
See LICENSE file for details.l System: Green Detection Experiment

## Project Overview
This project scaffolds a simple bio-inspired neural network for the task "detect green" using a custom Neuron class and ensemble monitor.

## How to Run
```bash
python experiment_runner.py
```

## Next Steps
- Add more input types (other colors, multi-pixel images)
- Experiment with specialty learning rules
- Refine mapping methods based on activity logs

## How It Works
Goal is set: “detect green”
Neurons are created: each with a random specialty
Input is loaded: a single green pixel
Goal and input are passed to neurons
Monitor records which neurons fire and their specialties
Human-readable summary and machine log produced

## Saved State and Logging

The system automatically saves the state of the neural network and the monitor after each run. On subsequent runs, it loads the saved state if available, enabling analysis of emergent properties over time. Logs and mappings are also saved for iterative mapping and human review.