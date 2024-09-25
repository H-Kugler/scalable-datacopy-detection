# Detecting Data-Copying Using Random Linear Projections
Repository for the Master Thesis in the Theory of Machine Learning Group at the University of Tuebingen.

## Prior Work on Data-Copying

A classical basic problem in unsupervised learning is generating artifical samples from a data distribution. In the simplest formulation of a setting, a learner has access to n i.i.d samples, S, from probability distribution p, and outputs a distribution, q with the goal that q = p.
One common pitfall that learners encounter is data-copying, in which q is constructed by overfitting to the samples from S, and effectively outputs near-copies of its training data. A generative model that simply memorizes its training data and outputs a training example at random (making q the uniform distribution over S) would be the prototypical example of a “data-copier.” Data- copying is undesirable for a whole host of reasons, not the least of which is that data-copiers exhibit poor generalization. As a consequence, detecting and prevent data-copying is an important step for building good generative models.
Prior work has provided a litany of tests and definitions that attempt to detect data-copying in practical generative models, and to formalize precise criteria for what constitutes egregious data-copying. For example, Meehan et al. (2020) provides a simple 3-sample based method that was able to demonstrate data-copying in a wide variety of cases. However, as demonstrated in Bhattacharjee et al. (2023), their methods are nevertheless unable to detect data-copying in cases where it is localized to only parts of the training data.
On the other hand, despite the theoretical guarnatees their methods provide, the approach provided in [BDC23] can require prohibitively large amounts of data and computation for detecting data-copying in high dimensional data distributions, making it impractical for most practical cases.
This motivates the natural question: is there a method for detecting data-copying that is both theoretically founded but also practically implementable? This project is centered around working towards this question.


## Approach

At a technical level, this project builds on the idea that if q egregiously copies dataset S, then the same is likely to hold for $\phi(q)$ and $\phi(S)$ where $\phi$ is any continuous map. After all, points that appear identical will remain so after being transformed by any well-behaved function.
Thus, rather than determining if q copies S, it suffices to determine if $\phi(q)$ copies $\phi(S)$ for some map $\phi$. To this end, we seek to answer the following specific questions:

1. Can we leverage the observation about $\phi$ to create an efficient data-copying detector by taking $\phi$ to be a random linear projection?
2. Is our algorithm able to succeed on practical datasets?
3. Under what conditions is our algorithm guarnateed to recover data-copying.


To investigate this, we will loosely proceed with the following steps:
1. We will first gather a working list of generative models along with their training data. Starting with the cases considered in [MCD20] is a good starting point.
2. We will develop implementations of prior work to serve as a baseline.
3. We will implement several variations of the algorithm outlined above and evaluate it on datasets above.
4. Finally, we will potentially include a theory component in which we investigate assumptions under which our algorithm enjoys provable guarantees.


