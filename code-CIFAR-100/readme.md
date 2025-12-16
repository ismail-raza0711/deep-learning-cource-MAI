# CIFAR-10 Classification - Correct Implementation

This is a clean, well-structured implementation of CIFAR-100 image classification using PyTorch. It supports both Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures.

## Project Structure

```
.
├── config.py           # Configuration settings
├── models.py           # Model architectures (MLP and CNN)
├── data_loader.py      # Data loading and preprocessing
├── utils.py            # Training and evaluation utilities
├── train.py            # Main training script
├── requirements.txt    # Python dependencies
└── readme.md           # This file
```

## Setup

1. **Install dependencies:**

See *requirements.txt*.

2. **Verify installation:**
```bash
python models.py
python data_loader.py
```

## Usage

### Quick Start

Run training with default settings (CNN model):
```bash
python train.py
```

### Customizing Configuration

Edit `config.py` to change settings.

### Model Architectures

**MLP (Multi-Layer Perceptron):**
- Input: Flattened 32×32×3 images tensors
- Hidden layers: [2048, 1024, 512, 256] (configurable)
- BatchNorm + ReLU + Dropout after each hidden layer
- Output: 100 classes

**CNN (Convolutional Neural Network):**
- 4 convolutional blocks (64→128→256→512)
- Each block: Conv2d layers + BatchNorm2d + ReLU + MaxPool2d + dropout 
- Capacity: Features doubled filter widths across blocks (up to 512 channels) 
            for high-capacity feature extraction.
- Classifier: Uses Global Average Pooling (GAP) followed by a final
               Linear layer to output 100 class logits.

**Training Pipeline Enhancements**
- The system implements high-performance training strategy
- Optimizer: Stochastic Gradient Descent (SGD) with Nesterov Momentum.
- Scheduler: Cosine Annealing Learning Rate Schedule.
- Loss: Cross Entropy Loss with Label Smoothing (0.1).
- Augmentation: Uses aggressive techniques including RandAugment and RandomErasing for
  robust generalization (used when MODEL_TYPE='cnn') (normal augumentation for MODEL_TYPE='mlp').

### Expected Performance

With default settings:

| Model | Validation Acc | Test Acc | Training Time* |
|-------|---------------|----------|----------------|
| MLP   | ~50-55%       | ~50-55%  | ~5 min         |
| CNN   | ~70-72%       | ~70-72%  | ~10 hrs on mac |

*On MAC (M4 pro)



## Output Files

After training, the following files are created in `./checkpoints/`:

- `best_model_cnn.pth` - Best model checkpoint
- `training_history_cnn.png` - Loss and accuracy curves
- `training_history_mlp.png` - Loss and accuracy curves

## Citation

Dataset: [CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

## License

This code is for educational purposes.
