# Mapping Neuron Specialties to Human Language

## Methods Used
- **Clustering**: k-means on specialty values to group neurons.
- **Rule-based Mapping**: Assign human label if specialty falls within a threshold for known feature (e.g., green).
- **Correlation**: Analyze neuron activation patterns with respect to input feature statistics.
- **Iterative Dictionary Building**: Use ensemble logs and human review to build-up specialty-to-human-label dictionary.

## Example Mapping
| Specialty Range | Human Label    |
|-----------------|---------------|
| 13.0 - 14.0     | Green Detector|
| 7.0 - 8.0       | Edge Detector |
| ...             | ...           |

## Logging & Review
- Each experiment logs: input label, active neurons, specialty values, cluster assignments.
- Human reviews logs and updates mapping dictionary.