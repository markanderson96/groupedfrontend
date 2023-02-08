# GroupedFrontend

A reimplementation of the grouped filterbanks of [EfficientLEAF](https://github.com/CPJKU/EfficientLEAF), but non-learnable and with the addition of using gammatone filterbanks alongside gabor.
Also includes various filterbank initialisations such as:
- Mel
- Bark
- Linear

## Requirements

Requires:
- PyTorch
- Numpy

## Usage

Can be installed in a conda environment or just included as part of the source.

Import the frontend

``` python
from groupedfrontend.frontend import GroupedFrontend
```

And then add it as a layer to your model.
For example to use:
- 40 filters
- Low frequency cut off of 80Hz
- Sample rate of 16KHz
- No compression
- Bark Scale

``` python
frontend = GroupedFrontend(
    n_filters=40,
    min_freq=80.,
    max_freq=8000.,
    sample_rate=16000,
    compression=None,
    init_filter='bark'
)
```

## Citation

Please cite the EfficientLEAF paper if you use this repository in a publication:
```
@INPROCEEDINGS{2022eleaf,
author={Schl{\"u}ter, Jan and Gutenbrunner, Gerald},
  booktitle={Proceedings of the 30th European Signal Processing Conference (EUSIPCO)},
  title={{EfficientLEAF}: A Faster {LEarnable} Audio Frontend of Questionable Use},
  year=2022,
  month=sep}
```
