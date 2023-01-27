# SimSiam Tensorflow2.0 Implementation
This is a simple and clean implementation of a simple Siamese network in order to perform contrastive learning as pretext task and evaluate on an image classification downstream task.

## Pretext Task
The pretext task consists in maximizing the similarity between the representations of two augmentations of the same image as explained in the paper: [SimSiam: Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

<div style="text-align: center;">
  <img src="https://user-images.githubusercontent.com/91251307/215079798-efccb85b-a52a-4214-8792-5b13cb2af541.png" width="60%"/><center/>
</div>

