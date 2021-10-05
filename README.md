# Vertebrae_segmentation

This is an associated repository of the work **Vertebrae segmentation, identification and localization using a graph optimization and a synergistic sycle**.

Pre-print at [link]().

It includes the proposed pipeline which is able to segment and label vertebrae from CT images of varient resolution, orientation and field of view (a few results shown below). You could test the performance following the steps.  

![visu](visu.png)

## Download the repository

``
git clone https://gitlab.inria.fr/spine/vertebrae_segmentation.git
``

## Set up a virtual environment

Install the link[anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html).

Once you have the conda installed, create a virtual environment using the env.yml file in the repo.

``
conda env create -f env.yml
``

The env name is defined as verse20, you could change it in the env.yml.

Then activate the env by

``
conda activate verse20
``

Now you should hava all the dependencies that the pipeline needs.

## Get the data
