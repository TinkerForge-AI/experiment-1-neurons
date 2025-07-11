# Bio-Inspired Neural Experiment: Green Identification

## System Overview
- **Neuron**: Lightweight unit with specialty, firing state, neighbor awareness, and goal alignment.
- **NeuronEnsembleMonitor**: Monitors activity, clusters neurons, maps specialties to human-understandable labels, logs behavior, and aids iterative mapping.

## Mapping Specialties
- **Initial**: Numeric or categorical specialty values.
- **Clustering**: Use k-means or similar to group neurons by specialty.
- **Correlation**: Compare cluster activation to input features (e.g., color).
- **Iterative Human Mapping**: Human researcher assigns labels to clusters/specialties based on observed activation patterns.

## Interactive Tools
- **Visualization**: Plot neuron grid with color-coded specialties and activations.
- **Logging**: Record all activations, clusters, mapping steps for review.
- **Human Summary**: Generate human-readable reports after each experiment.

## Iterative Mapping Strategy
1. Run experiment with input (e.g., green image).
2. Monitor neuron firing and specialty clustering.
3. Correlate cluster activations with input features.
4. Assign human labels to clusters (e.g., "green detector").
5. Refine mapping as more experiments are conducted.

## Future Extensions
- Extend to more colors/features.
- Enable feedback and hierarchical layers.
- Add more sophisticated learning rules for specialties.
- Automate mapping with semi-supervised learning.