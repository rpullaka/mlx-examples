# flash_lm 

Train an on-device LM with MLX

### Install

Install dependencies:

```
pip install -r requirements.tx
```

Install MLX on macOS:

```
pip install mlx
```

Or for CUDA:

```
pip install mlx[cuda13]
```

### Run

To run:

```
python pretrian.py
```

### Next Steps 

To customize the model change the default config (`configs/tiny.py`) or
make a new config and use it.

```
python pretrian.py --config my_custom_config.py
```
