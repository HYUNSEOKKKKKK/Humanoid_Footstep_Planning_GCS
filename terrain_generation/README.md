# CVAE terrain generation

Install the integrated environment using the [root README](../README.md).

```bash
# From the repository root; samples the shipped model without training.
python terrain_generation/generate.py --alpha 0.5 --stones 4 --samples 100 --output-dir outputs/new_terrains

# Train a separate model explicitly.
python terrain_generation/train.py --output outputs/checkpoints/my_cvae.pt
```

The benchmark inputs are in `../data/terrains/`; checkpoints are in `checkpoints/`.
`main.py` contains the original CVAE model/training/sampling functions. Direct
execution of that file now delegates to sampling; it does not retrain the model.
The historical standalone environment specification is retained for provenance;
the integrated workflow uses the root `environment.yml`.
