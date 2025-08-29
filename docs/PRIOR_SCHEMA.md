# Prior Schema

The docking prior embeds each VOC class into a fixed-length vector.

## Fields
- `class` *(string)*: VOC class label.
- `embedding` *(list[float])*: length `D` feature vector.

## Dimensions
For this project `D = 56` features per class.

## Example
```json
{
  "acetone": [0.1, 0.2, 0.3],
  "ethanol": [0.0, 0.1, 0.4]
}
```
