## ARGH-tf

This repo contains some of my thoughts related to solving the ARC dataset ([paper](https://arxiv.org/pdf/1911.01547.pdf) [repo](https://github.com/fchollet/ARC)).

# A toy problem: Finding the "odd one out" vector

We have N vectors. N-1 of them are equal to each other, and the one that's left is different, i.e. inequal to the others. The task is to identify the "odd one out". A normal transformer ([paper](https://arxiv.org/pdf/1706.03762.pdf)) isn't capable of solving it, if the vectors are initialized randomly.

We can make the observation that the task is more or less a counting task: Find the vector that only appears once in the data. So we can make a hypothesis: transformers are incapable of counting abstractly.

# Counting

1, 2, 3, ...
