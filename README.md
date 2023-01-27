# SimSiam Tensorflow2.0 Implementation 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)![Kaggle](https://img.shields.io/badge/kaggle_notebook-20BEFF?style=for-the-badge&logo=googlecolab&logoColor=white)
This is a simple and clean implementation of a simple Siamese network in order to perform contrastive learning as pretext task and evaluate on an image classification downstream task following the paper: [SimSiam: Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566).

## Architecture

The architecture consists of two branches with shared encoder and predictor but the gradient only flows to the left branch as for the right branch the gradient is stopped.
The encoder usually is a feature extractor to obtain a compact representation of the image through a small mlp; the output is passed to a predictor head (another small mlp) and finally it is computed the similarity between the outputs of the two branches.

<div align="center">
  <img src="https://user-images.githubusercontent.com/91251307/215079798-efccb85b-a52a-4214-8792-5b13cb2af541.png" width="60%"/>
</div>

## Loss Function

The Loss function is the sum of the negative cosine similarity between the representations and the predictions of opposite branches:

$$Loss = \frac{1}{2} D(p_1, z_2) +  \frac{1}{2} D(p_2, z_1) \ \ with \ \ D(p, z) = - \frac{p \cdot z}{||p||_2 \cdot ||z||_2}$$

This is the pseudocode of the paper that makes it even clearer:
```
# f: backbone + projection mlp
# h: prediction mlp
for x in loader: # load a minibatch x with n samples
   x1, x2 = aug(x), aug(x) # random augmentation
   z1, z2 = f(x1), f(x2) # projections, n-by-d
   p1, p2 = h(z1), h(z2) # predictions, n-by-d
   L = D(p1, z2)/2 + D(p2, z1)/2 # loss
   L.backward() # back-propagate
   update(f, h) # SGD update
def D(p, z): # negative cosine similarity
   z = z.detach() # stop gradient
   p = normalize(p, dim=1) # l2-normalize
   z = normalize(z, dim=1) # l2-normalize
   return -(p*z).sum(dim=1).mean()
```

## Pretext Task

The pretext task consists in taking as input two augmentations of the same image and trying to generate embeddings as similar as possible for both augmentations. It is important to pick the correct augmentations in order to learn the correct invariances and obtain better performances in the downstream task.

## Downstream Task

The downstream task is image classification on the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) thta maybe is to simple for assessing the quality of contrastive learning, in the future I will test it on ImageNet.

## Citations
```
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```
