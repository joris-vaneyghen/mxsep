# Music Source Separation Toolkit (musep)

A flexible, research-oriented toolkit for training music source separation models with PyTorch.


## Key Features Implemented

1. **Modular Configuration**: Hierarchical configuration system with YAML support
2. **Flexible STFT/ISTFT**: Optional STFT modules that can be disabled for training
3. **Dynamic Mix Generation**: Configurable mix generation with augmentations
4. **Multi-backend Training**: Support for GPU, TPU, and distributed training
5. **Monitoring & Debugging**: WandB integration and memory debugging tools
6. **ONNX Export**: Easy model export for production deployment
7. **Docker Support**: Separate containers for data preparation, training, and inference
8. **Package Installation**: Can be installed via pip with optional dependencies

This structure provides a solid foundation for a research-to-production music source separation pipeline that's both flexible and maintainable.