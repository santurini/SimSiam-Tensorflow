# SimSiam Tensorflow2.0 Implementation
This is a simple and clean implementation of a simple Siamese network in order to perform contrastive learning as pretext task and evaluate on an image classification downstream task following the paper: [SimSiam: Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566).

## Architecture

The architecture consists of two branches with shared encoder and predictor but the gradient only flows to the left branch as for the right branch the gradient is stopped.
The encoder usually is a feature extractor to obtain a compact representation of the image through a small mlp; the output is passed to a predictor head (another small mlp) and finally it is computed the similarity between the outputs of the two branches.

<div align="center">
  <img src="https://user-images.githubusercontent.com/91251307/215079798-efccb85b-a52a-4214-8792-5b13cb2af541.png" width="60%"/>
</div>

## Loss Function

The Loss function is the sum of the negative cosine similarity between the representations and the predictions of opposite branches:

$$ Loss = \frac{1}{2} D(p_1, z_2) +  \frac{1}{2} D(p_2, z_1) $$, where

## Pretext Task

The pretext task consists in taking as input two augmentations of the same image and trying to generate embeddings as similar as possible for both augmentations. It is important to pick the correct augmentations in order to learn the correct invariances and obtain better performances in the downstream task.

## Downstream Task

The downstream task is image classification on the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) thta maybe is to simple for assessing the quality of contrastive learning, in the future I will test it on ImageNet.


