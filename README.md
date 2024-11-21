# XVAE

Emulating complex climate models via integrating variational autoencoder
and spatial extremes

> Department of Statistics, University of Missouri

### Max-infinitely divisible processes

Our model is based on the max-infinitely divisible process proposed by
Bopp et al. (2021)[[1]](#1) and allows for both short-range asymptotic
independence and dependence along with long-range asymptotic
independence, which can be specified as follows:

<p align="center">

<img src="https://latex.codecogs.com/svg.image?X_t(\textbf{s})=\epsilon_t(\textbf{s})Y_t(\textbf{s}),&amp;space;"/>

</p>

where $X_t(s)$ is a spatio-temporal output from a simulator (e.g.,
high-resolution climate model), $\epsilon_t(s)$ is a white noise process
with independent $1/\alpha$-Fréchet marginal distribution, and $Y_t(s)$
is described by a low-rank representation:

<p align="center">

<img src="https://latex.codecogs.com/svg.image?Y_t(\textbf{s})=\left(\sum_{k=1}^K&amp;space;\omega_k(\textbf{s},&amp;space;r_k)^{1/\alpha}Z_{kt}\right)^\alpha.&amp;space;"/>

</p>

Here, we use compactly-supported Wendland basis functions
$\omega_k(s, r_k)$, $k=1,\ldots,K$, which are centered at $K$
pre-specified knots. The latent variables are lighter-tailed,
exponentially tilted, positive-stable variables:

<p align="center">

<img src="https://latex.codecogs.com/svg.image?Z_{kt}\sim&amp;space;H(\alpha,\alpha,\theta_k),\;&amp;space;k=1,\ldots,&amp;space;K,&amp;space;"/>

</p>

in which $\alpha\in (0,1)$ is the concentration parameter, and larger
values of the tail index $\theta_k\geq 0$ induce lighter-tailed
distributions of $Z_{kt}$.

### User instructions

The users can follow the data analysis shown in our manuscript
[link](https://arxiv.org/abs/2307.08079) to learn the implementation of
the XVAE. Right now, the XVAE is implemented in R with different
scripts. We wish to translate everything into python in the near future
and deliver a more user-friendly package. For the time being, the users
can simply run through the following scripts: - Initialization.R -
XVAE_stochastic_gradient_descent.R - XVAE_results_summary.R

Next, we demonstrate how to train an XVAE using the dataset simulated from Model III in the Simulation Study of Zhang et al. [[2]](#2). The steps to run an XVAE can be generally applied to any spatial input.

#### 1. Initialization.R
First, we need to make sure the file `utils.R` and  is under your working directory so all the utility functions can be loaded:
``` ruby
source("utils.R")
load("example_X.RData")
```

Assume the input data `X` is appropriately marginally transformed that has rows representing different locations and columns representing different times, and `stations` stores the 2-D coordinates of each location.

Then, we apply Algorithm 1 in the Supplementary Material of Zhang et al. [[2]](#2) to select data-driven knots based on high values (e.g., `thresh_p=0.95`) in the input data matrix and also determine the radius for the Wendland basis functions:
``` ruby
knots <- data_driven_knots(X, stations, 0.95, echo=TRUE)
r <- calc_radius(knots, stations)
```

With the knot locations, we can calculate the Wendland basis function values and row-standardize so they sum up to 1 for each location
``` ruby
W <- wendland(eucD,r=r)
W <- sweep(W, 1, rowSums(W), FUN="/")
dim(W)
```
Here, the dimensions of `W` should be $n_s\times k$, and we can do some bookkeeping of the dimensions:
``` ruby
k = nrow(knots)
n.s <- nrow(stations)
n.t <- ncol(X)
stations <- data.frame(stations)
knots <- data.frame(knots)
```

_(Optional) Visualize knots and the coverage of Wendland basis functions_
``` ruby
visualize_knots(knots, stations, r, W)
```
![plot_knots](www/knots.png)

### (Conditional) Variational autoencoder

![plot1](www/Extremes_CVAE.png)

## References

<a id="1">[1]</a> Gregory P Bopp, Benjamin A Shaby, and Raphaël Huser. A
hierarchical max-infinitely divisible spatial model for extreme
precipitation. *Journal of the American Statistical Association*,
116(533):93–106, 2021.

<a id="2">[2]</a> Zhang, Likun, Xiaoyu Ma, Christopher K. Wikle, and
Raphaël Huser. "Flexible and efficient spatial extremes emulation via
variational autoencoders." arXiv preprint arXiv:2307.08079 (2024).
