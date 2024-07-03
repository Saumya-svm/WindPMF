## WindPMF

To run : 
`python3 train.py --include_denoiser True model_dir </path/to/save/model>`

## Command Line Arguments

- `--train` (int, default=0): Flag to indicate whether to train the model. Set to `1` to enable training.
- `--eval` (int, default=1): Flag to indicate whether to evaluate the model. Set to `1` to enable evaluation.
- `--opt` (int, default=0): Flag to indicate whether to optimize the model hyperparameters. Set to `1` to enable optimization.
- `--weights_path` (str, default=None): Path to the pre-trained model weights. If not provided, the model will be initialized with random weights.
- `--params_path` (str, default=None): Path to the file containing optimized hyperparameters. If not provided, default parameters will be used.
- `--ini_learning_rate` (float, default=8e-3): Initial learning rate for the optimizer.
- `--decay_rate` (float, default=0.85): Decay rate for the learning rate scheduler.
- `--step_size` (int, default=75): Step size for the learning rate scheduler.
- `--seq_len` (int, default=16): Sequence length for the input data.
- `--numBlocks` (int, default=3): Number of blocks in the model architecture.
- `--numLayers` (int, default=3): Number of layers in each block of the model.
- `--filters` (int, default=64): Number of filters in the convolutional layers.