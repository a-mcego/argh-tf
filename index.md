# ARGH-tf

This repo contains some of my thoughts related to solving the ARC dataset ([paper](https://arxiv.org/pdf/1911.01547.pdf) [repo](https://github.com/fchollet/ARC)).

## Motivation for this text

A main reason for the existence of ARC is to have tasks that are simple or almost trivial to humans, but very difficult for computers. Here, I demonstrate one specific kind of task that is difficult to transformer networks: counting.

ARC has many tasks that involve counting in some form or another. Therefore, in order to solve ARC, we have to be able to count things. We can make a toy problem to test whether transformers can count.

## A toy problem: Finding the "odd one out" vector

We have N vectors. N-1 of them are equal to each other, and the one that's left is different, i.e. inequal to the others. The task is to identify the "odd one out". This is a trivial problem to humans. But in my tests, a normal transformer ([paper](https://arxiv.org/pdf/1706.03762.pdf)) isn't capable of solving it, if the vectors are initialized randomly before every training step. However, it *can* be solved, quite trivially, if the vectors are initialized once in the beginning of the program.

We can make the observation that this is more or less a counting task: Find the vector that only appears once in the data. So we can make a hypothesis: transformers are incapable of counting abstractly. My hypothesis is that it has something to do with the classification not being linearly separable, because the vectors are all random, and transformers somehow cannot unrandomize them.

## Counting

Let's define a toy dataset: `[1, 8, 5, 5, 3, 1, 5]` Here, N=7, and for simplicity, the vectors are of length 1, i.e. scalars.

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

This also solves the toy problem we defined earlier, as the result will be something like `[N-1, N-1, ... , 1, ... , N-1]`, which is linearly separable.

## Differentiable counting, special case

However, the method has a gradient of 0, so it cannot be used in a neural network. We can try, for example, *cosine similarity*. There are some problems with it that can be fixed, though. First of all, as cosine similarity doesn't make sense in only one dimension, we'll make a new toy dataset. Second of all, we'll start with a simpler version, where all vectors are axis-aligned. The new dataset is:

`[[0,-1],[0,1],[-1,0],[-1,0],[0,1],[1,0],[-1,0]]`

According to the previous algorithm, the output is `[1 2 3 3 2 1 3]`. However, the cosine similarity matrix is

```
[ 1 -1  0  0 -1  0  0]
[-1  1  0  0  1  0  0]
[ 0  0  1  1  0 -1  1]
[ 0  0  1  1  0 -1  1]
[-1  1  0  0  1  0  0]
[ 0  0 -1 -1  0  1 -1]
[ 0  0  1  1  0 -1  1]
```
, and if we sum it up like before, we get `[-1, 1, 2, 2, 1, -2, 2]`, which is not what we want. Why? We see that vectors that are opposites of each other (like `[1,0]` and `[-1,0]`) actually produce `-1`, which means they *remove* one from the count. But we want to count them as a `0` instead, because we just don't want to count them. One way to fix this is to put the matrix through a `max(x,0)` function, also known as ReLU, before summing it up. The matrix becomes

```
[ 1  0  0  0  0  0  0]
[ 0  1  0  0  1  0  0]
[ 0  0  1  1  0  0  1]
[ 0  0  1  1  0  0  1]
[ 0  1  0  0  1  0  0]
[ 0  0  0  0  0  1  0]
[ 0  0  1  1  0  0  1]
```
and the sum is now `[1, 2, 3, 3, 2, 1, 3]`, exactly what we want.

## Differentiable counting, general case

Now we allow any vectors, not just ones on the axes. New dataset:

`[[0,1],[1,1],[-1,1],[-1,1],[-1,1],[1,0.9],[1,0.9]]`

The result should be

`[1, 1, 3, 3, 3, 2, 2]`

As `cosine_similarity(V1,V2) == dot_product(V1/len(V1), V2/len(V2))` , we can normalize the dataset first, then use the dot product. The normalized dataset is (approximately):
`[[0.000,1.000],[0.707,0.707],[-0.707,0.707],[-0.707,0.707],[-0.707,0.707],[0.743,0.669],[0.743,0.669]]

The pairwise dot product, after `max(x,0)`, is approximately:
```
[1.    0.707 0.707 0.707 0.707 0.669 0.669]
[0.707 1.    0.    0.    0.    0.999 0.999]
[0.707 0.    1.    1.    1.    0.    0.   ]
[0.707 0.    1.    1.    1.    0.    0.   ]
[0.707 0.    1.    1.    1.    0.    0.   ]
[0.669 0.999 0.    0.    0.    1.    1.   ]
[0.669 0.999 0.    0.    0.    1.    1.   ]
```

The result is still a symmetric matrix, since `cosine_similarity(a,b) == cosine_similarity(b,a)` , and we can sum it up just like before. The result is `[5.166 3.704 3.707 3.707 3.707 3.668 3.668]`. 

Remember, the result should have been`[1, 1, 3, 3, 3, 2, 2]`. The problem is that many of the vectors are close to each other, and it's really not obvious whether they should be counted as "partly the same vector" or "a totally different vector". We *could* solve it by being more strict with the dot product results, by for example dropping values below 0.5. That sounds very hacky, and I have a better idea.

## Counting multidimensional vectors

In machine learning, the vectors that are used usually have lots of dimensions. Even the *small* version of GPT-2 ([paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) uses a vector dimension of *768*. 

My argument is that if two vectors of that size are almost equal, maybe they *should* be counted together. I also did an experiment: I choose 6 random vectors and make a matrix such that the first vector appears once, the second appears twice, etc. The result should be

`[1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 6]`

and the run ended with

`[1.383 2.351 2.351 3.193 3.193 3.193 4.267 4.267 4.267 4.267 5.231 5.231 5.231 5.231 5.231 6.322 6.322 6.322 6.322 6.322 6.322]` 

,which is very close to the real result. We could remove some of the small numbers from the similarity matrix, just like we remove the negative numbers. That actually gives exact results, but also seems like it wouldn't have a non-zero derivative.

## Comparing counting to transformer attention

To be continued...
