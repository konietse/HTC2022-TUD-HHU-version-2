# [Helsinki Tomography Challenge 2022](https://www.fips.fi/HTC2022.php) -  [Technical University Dortmund](https://www.tu-dortmund.de) & [Heinrich Heine University Düsseldorf](https://www.hhu.de/)

Thomas Germer<sup>1</sup>, Jan Robine<sup>2</sup>, Sebastian Konietzny<sup>2</sup>, Stefan Harmeling<sup>2</sup>, Tobias Uelwer<sup>2</sup>

<sup>1</sup> Department of Computer Science, Heinrich Heine University Düsseldorf, Germany

<sup>2</sup> Department of Computer Science, Technical University Dortmund, Germany

This is a submission to the Helsinki Tomography Challenge 2022.

The goal of the challenge is to compute binary segmentations from limited angle sinograms, consisting of two-dimensional X-ray projections of an acrylic disk.

A dataset is available online:

* https://www.fips.fi/HTCdata.php
* https://zenodo.org/record/6984868

# Method Description

Since the dataset is quite small, we generate synthetic data to train a neural network. First, we generate a large number of synthetic images by placing various shapes on top of a disk. We then compute sinograms from those images using the forward operator from the [Helsinki Tomography Toolbox](https://github.com/Diagonalizable/HelTomo) and [Astra Toolbox](https://www.astra-toolbox.com/). Lastly, we train a neural network to directly predict the images from the subsampled sinograms.

# Installation instructions

To install:

```
pip3 install -r requirements.txt
```

# Usage instructions

Use the following command to generate images from subsampled sinograms in the directory `data/limited`.
The images will be written to the directory `output`.

```
python3 main.py data/limited output
```

You can then run the following command to compute the mean score on the subsampled dataset.

```
python3 main_compute_score.py output
```

# Examples

| Angles |         Limited sinogram                    |         Prediction                    | Ground truth |
|:----:|:-----------------------------------:|:------------------------------------:|:------------------------------------:|
| 61° |![](data/limited_png/tb_opening_angle_030_starting_angle_000.png)|![](expected_output/tb_opening_angle_030_starting_angle_000.png)|![](data/htc2022_tb_full_recon_fbp_seg.png)|
| 61° |![](data/limited_png/tc_opening_angle_030_starting_angle_000.png)|![](expected_output/tc_opening_angle_030_starting_angle_000.png)|![](data/htc2022_tc_full_recon_fbp_seg.png)|
| 61° |![](data/limited_png/td_opening_angle_030_starting_angle_000.png)|![](expected_output/td_opening_angle_030_starting_angle_000.png)|![](data/htc2022_td_full_recon_fbp_seg.png)|
| 61° |![](data/limited_png/ta_opening_angle_030_starting_angle_000.png)|![](expected_output/ta_opening_angle_030_starting_angle_000.png)|![](data/htc2022_ta_full_recon_fbp_seg.png)|
| 181° |![](data/limited_png/ta_opening_angle_090_starting_angle_000.png)|![](expected_output/ta_opening_angle_090_starting_angle_000.png)|![](data/htc2022_ta_full_recon_fbp_seg.png)|
| 61°, start 240° |![](data/limited_png/ta_opening_angle_090_starting_angle_240.png)|![](expected_output/ta_opening_angle_090_starting_angle_240.png)|![](data/htc2022_ta_full_recon_fbp_seg.png)|
