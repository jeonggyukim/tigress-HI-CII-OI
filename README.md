# tigress-HI-CII-OI

A minimal set of scripts and examples for performing synthetic HI, OI, and CII observations using TIGRESS simulation snapshots.

## Setting up a conda environment

Below is an example of how you can set up a new conda environment. It assumes that you have already installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) or anaconda on your system.

1. Clone this repo
   ```sh
   git clone https://github.com/jeonggyukim/tigress-HI-CII-OI.git
   ```
3. Create an environment from the env.yml file
   ```sh
   conda update conda # if you haven't already
   conda env create -f path_to_this_repo/env.yml
   ```
4. Activate the tigress-synthetic environment
   ```sh
   conda activate tigress-synthetic
   ```

## Downloading snapshots

Example data can be downloaded here.
```
wget http://tigress-web.princeton.edu/~changgoo/TIGRESS_example_data/R8_8pc_rst.0300.vtk
wget http://tigress-web.princeton.edu/~changgoo/TIGRESS_example_data/R8_2pc_rst-mid.0300.vtk
```

Full data release can be downloaded using Globus (see [TIGRESS public data release](https://princetonuniversity.github.io/astro-tigress/intro.html)).
