# Architecture

High level: data -> docking priors -> teacher training -> student KD -> export -> inference.

## Deployment path

Trained PyTorch models are exported to the interoperable ONNX format using
``python -m src.train.export_onnx``.  From ONNX the recommended conversion path
is:

``PyTorch → ONNX → TFLite → TFLite‑Micro``

* [ONNX](https://onnx.ai/) serves as the interchange format.
* [TensorFlow Lite](https://www.tensorflow.org/lite) provides tooling to
  convert ONNX models to a representation suitable for mobile/embedded use.
* [TFLite Micro](https://www.tensorflow.org/lite/microcontrollers) executes the
  resulting model on bare‑metal microcontrollers.

The repository does not yet contain a C++ inference stub for the microcontroller
target.  **TODO:** add a minimal C++ example wiring up the exported TFLite Micro
model to the hardware once the interface stabilises.
