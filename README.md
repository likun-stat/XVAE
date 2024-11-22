# XVAE Tutorial 

Emulating complex climate models via integrating variational autoencoder
and spatial extremes

> Department of Statistics, University of Missouri

## Introduction
This tutorial provides step-by-step instructions for implementing the XVAE model, designed to emulate 
high-resolution climate models by integrating spatial extreme value theory with a variational autoencoder (VAE). 
The methodology follows the max-infinitely divisible process proposed by Bopp et al. (2021)[[1]](#1) and is implemented using `R`. 
A Python package is planned for future development.

## Implementation Guide

### Requirements
1. **Dependencies**: Install `R` libraries including `torch`, `dplyr`, `VGAM` and any required visualization libraries such as `ggplot`.
2. The users can follow the demonstration shown below to learn the implementation of
the XVAE. We wish to translate everything into python in the near future
and deliver a more user-friendly package. For the time being, the users
can simply download the following scripts: 
- utils.R 
- Initializing_XVAE.R
- XVAE_training_loop.R


## Step-by-Step Instructions

Next, we demonstrate how to train an XVAE using the dataset simulated from Model III in the Simulation Study of Zhang et al. [[2]](#2). The steps to run an XVAE can be generally applied to any spatial input.

### 1. Generate data-driven knots

First, we need to make sure the file `utils.R` is under your working directory so all the utility functions can be loaded:
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

Using the knot locations, we calculate the Wendland basis function values and row-standardize them so that they sum to 1 at each location:
``` ruby
eucD <- rdist(stations,as.matrix(knots))
W <- wendland(eucD,r=r)
W <- sweep(W, 1, rowSums(W), FUN="/")
dim(W)  # Verify dimensions: `n.s` × `k`
```
Here, the dimensions of `W` should be $n_s\times k$, and we can summarize the dimensional bookkeeping as follows:
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

### 2. Initial values for latent variables

In this section, we initialize the latent expPS variables via solving a linear system using QR decomposition:
``` ruby
# Setting up the Fréchet white noise process
alpha = 0.5; tau <- 0.1; m <- 0.85
W_alpha <- W^(1/alpha)
Z_approx <- array(NA, dim=c(k, n.t))
for (iter in 1:n.t){
  if(iter %% 10 == 0 ) cat('Finding good initial Z_t for time', iter, '\n')
  Z_approx[,iter] <- relu(qr.solve(a=W_alpha, b=X[,iter]))
}

# Compute approximations
Y_star <- (W_alpha)%*%(Z_approx)
Y_approx <- Y_star - relu(Y_star-X) 
```

### 3. Define VAE Weights and Encoder-Decoder Initialization
Now we initialize the weights and biases for the encoder in the VAE and define them as `torch` tensor:
``` ruby
## -------------------- Initializing VAE --------------------
source("Initializing_XVAE.R")
```

### 4. Training the VAE
Next, we configure the learning rate, activation functions, and other network parameters. These parameters might _require tuning_ based on dataset complexity and model performance.
``` ruby
learning_rate <- -1e-15; alpha_v <- 0.9
lrelu <- nn_leaky_relu(-0.01)
nEpoch = 10000
```
### Training Loop

The main training process, where the VAE optimizes the ELBO (Evidence Lower Bound):
``` ruby
source("XVAE_training_loop.R")
```

### 5. Post-Processing and Results

Now with the trained XVAE weights and biases in the global environment, we emulate the spatial input using the folowing function:
``` ruby
output <- emulate_from_trained_XVAE()
```
Here, `output` is a list with 
- `emulations`: A matrix of simulated values for spatial inputs.
-  `theta_est`: A matrix of estimated parameters from the decoder.

#### $\chi_d(u)$ comparison

Now we empirically estimate $\chi_d(u),\; u\in (0,1),$ for both the original spatial input and the emulated dataset at $d\approx 1.5$:
``` ruby
chi_plot(X,stations, output$emulations, distance=1.5, legend = TRUE, ylab = expression(atop(bold("Simulated input data from Model III"), chi[u])))
```
![plot_chi](www/chi_plot.png)
Summarize the results using the provided script:

## Max-infinitely divisible processes

Our model is based on the max-infinitely divisible process proposed by
Bopp et al. (2021)[[1]](#1) and allows for both short-range asymptotic
independence and dependence along with long-range asymptotic
independence, which can be specified as follows:

<p align="center">

<img src="https://latex.codecogs.com/svg.image?X_t(\textbf{s})=\epsilon_t(\textbf{s})Y_t(\textbf{s}),&amp;space;"/>

</p>

where 
- $X_t(s)$ is a spatio-temporal output from a simulator (e.g.,
high-resolution climate model), 
- $\epsilon_t(s)$ is a white noise process
with independent $1/\alpha$-Fréchet marginal distribution, 
- $Y_t(s)$ is described by a low-rank representation:

<p align="center">

<img src="https://latex.codecogs.com/svg.image?Y_t(\textbf{s})=\left(\sum_{k=1}^K&amp;space;\omega_k(\textbf{s},&amp;space;r_k)^{1/\alpha}Z_{kt}\right)^\alpha.&amp;space;"/>

</p>

with:
- $\omega_k(s, r_k)$ compactly-supported Wendland basis functions
, $k=1,\ldots,K$, which are centered at $K$
pre-specified knots. 
- $Z_{kt}\sim \text{expPS}(\alpha,\theta_k)$: Exponentially tilted, positive-stable variables,  governed by $\alpha\in (0,1)$ and a tail index 
$\theta_k\geq 0$.

## (Conditional) Variational autoencoder

![plot1](www/Extremes_CVAE.png)

## References

<a id="1">[1]</a> Gregory P Bopp, Benjamin A Shaby, and Raphaël Huser. A
hierarchical max-infinitely divisible spatial model for extreme
precipitation. *Journal of the American Statistical Association*,
116(533):93–106, 2021.

<a id="2">[2]</a> Zhang, Likun, Xiaoyu Ma, Christopher K. Wikle, and
Raphaël Huser. "Flexible and efficient spatial extremes emulation via
variational autoencoders." arXiv preprint arXiv:2307.08079 (2024).
