# ARGH-tf

This repo contains some of my thoughts related to solving the ARC dataset ([paper](https://arxiv.org/pdf/1911.01547.pdf) [repo](https://github.com/fchollet/ARC)).

## A toy problem: Finding the "odd one out" vector

We have N vectors. N-1 of them are equal to each other, and the one that's left is different, i.e. inequal to the others. The task is to identify the "odd one out". A normal transformer ([paper](https://arxiv.org/pdf/1706.03762.pdf)) isn't capable of solving it, if the vectors are initialized randomly.

We can make the observation that this is more or less a counting task: Find the vector that only appears once in the data. So we can make a hypothesis: transformers are incapable of counting abstractly.

## Counting

Let's define a toy dataset: `[1, 8, 5, 5, 3, 1, 5]`

Can we count "similar vectors" using something that looks like a transformer? We can just test equality and count them to get a dictionary-like structure like: `[1:2, 8:1, 5:3, 3:1]`, since there are two 1s, one 8, three 5s and one 3. However, that doesn't work with transformers, because the output should always be the same size as the input.

What we *can* do to fix that is: for every vector V_i in N, we count how many vectors *equal to that vector* there are in the dataset. Simplified example:

Dataset is still: `[1, 8, 5, 5, 3, 1, 5]`

Output should be: `[2, 1, 3, 3, 1, 2, 3]`, since there are two 1s, one 8, three 5s and one 3.

How to calculate something like that? We can do a pairwise equality calculation on the dataset, marking 1 if they are equal, and 0 if they are not. We get:

```
[1, 0, 0, 0, 0, 1, 0]
[0, 1, 0, 0, 0, 0, 0]
[0, 0, 1, 1, 0, 0, 1]
[0, 0, 1, 1, 0, 0, 1]
[0, 0, 0, 0, 1, 0, 0]
[1, 0, 0, 0, 0, 1, 0]
[0, 0, 1, 1, 0, 0, 1]
```

We see that the resulting matrix is symmetric, because a==b implies b==a. We can sum-reduce the matrix along one of the dimensions (it doesn't matter which one, since the matrix is symmetric), to get `[2, 1, 3, 3, 1, 2, 3]`, which is exactly what we wanted.

## Differentiable counting

However, the method has a gradient of 0, so it cannot be used in a neural network. We can try, for example, cosine similarity.


## Comparing counting to transformer attention

