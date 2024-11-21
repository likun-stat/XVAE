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
2. The users can follow the data analysis shown in our manuscript
[link](https://arxiv.org/abs/2307.08079) to learn the implementation of
the XVAE. We wish to translate everything into python in the near future
and deliver a more user-friendly package. For the time being, the users
can simply run through the following scripts: 
- Initialization.R 
- XVAE_stochastic_gradient_descent.R 
- XVAE_results_summary.R


## Step-by-Step Instructions

Next, we demonstrate how to train an XVAE using the dataset simulated from Model III in the Simulation Study of Zhang et al. [[2]](#2). The steps to run an XVAE can be generally applied to any spatial input.

### 1. Generate data-driven knots

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

Using the knot locations, we calculate the Wendland basis function values and row-standardize them so that they sum to 1 at each location:
``` ruby
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
source("Initializing_VAE")
```

### 4. Training the VAE
Next, we configure the learning rate, activation functions, and other network parameters. These parameters might _require tuning_ based on dataset complexity and model performance.
``` ruby
learning_rate <- -1e-13; alpha_v <- 0.9
lrelu <- nn_leaky_relu(-0.01)
nEpoch = 80000
```
### Training Loop

The main training process, where the VAE optimizes the ELBO (Evidence Lower Bound):
``` ruby
old_loss <- -Inf
for (t in 1:nEpoch) {
  # Adapt learning rate and momentum at regular intervals
  if(round(log2(t))%%4 == 0) { learning_rate <- 2 * learning_rate; alpha_v <- 0.95*alpha_v}
  
  ### -------- Forward pass --------
  # Generate random noise for reparameterization trick
  Epsilon <- t(0.1+abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
  Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
  Epsilon <- torch_tensor(Epsilon,dtype=torch_float())
  Epsilon_prime <- torch_tensor(Epsilon_prime,dtype=torch_float())
  
  
 ### -------- Encoder for Primary Latent Space --------
  h <- w_1$mm(X_tensor)$add(b_1)$relu()
  h_1 <- w_2$mm(h)$add(b_2)$relu()
  sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
  mu <- w_4$mm(h_1)$add(b_4)$relu()
  
  ### -------- Encoder for Auxiliary Latent Space --------
  # Similar encoding process for auxiliary representation `v_t_prime`
  h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
  h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()
  
  ### -------- Activation via Laplace transformation --------
  h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
  h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
  h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
  theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
  
  sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
  mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)
  
  ### -------- Re-parameterization trick --------
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  
  ### -------- Decoder --------
  # Decode auxiliary latent variables to reconstruct input
  l <- w_5$mm(v_t_prime)$add(b_5)$relu()
  l_1 <- w_6$mm(l)$add(b_6)$relu()
  theta_t <- w_7$mm(l_1)$add(b_7)$relu()
  
  # Apply leaky ReLU activation for final output
  y_star <- lrelu(W_alpha_tensor$mm(v_t)$add(b_8))
  
  
  ### -------- Evidence Lower Bound (ELBO) -------- 
  # Compute reconstruction loss (Part 1)
  standardized <- X_tensor$divide(y_star)$sub(m)
  leak <- as_array(sum(standardized<0))
  if(leak>0 & leak<=175) standardized$abs_()
  leak2 <- as_array(sum(standardized==0))
  if(leak2>0 & leak2<=120) standardized$add_(1e-07)
  part1 <- -2 * standardized$log()$sum() - y_star$log()$sum() - tau*standardized$pow(-1)$sum() # + n.s*n.t*log(tau)
  
  # Compute latent space regularization (Part 2)
  V_t <- v_t$view(c(k*n.t,1))
  Theta_t <- theta_t$view(c(k*n.t,1))
  part_log_v1  <- V_t$pow(-const)$mm(Zolo_vec) 
  part_log_v2  <- (-V_t$pow(-const1)$mm(Zolo_vec))$exp()
  part_log_v3 <- Theta_t$pow(alpha)-Theta_t$mul(V_t)
  part2 <- (part_log_v1$mul(part_log_v2)$mean(dim=2)$log()+part_log_v3$view(k*n.t)$add(const3))$sum()
  if(as_array(part2) == -Inf) {
    part2_tmp <- part_log_v1$log()[,1] + (-V_t$pow(-const1)$mm(Zolo_vec))[,1]
    part2 <- (part2_tmp+part_log_v3$view(k*n.t)$add(const3))$sum()
  }
  
  # Compute Gaussian prior penalty (Part 3)
  part3 = Epsilon$pow(2)$sum()/2 + Epsilon_prime$pow(2)$sum()/2 + sigma_sq_vec$log()$sum() + sigma_sq_vec_prime$log()$sum()
  
  # Aggregate ELBO
  res <- part1 + part2 + part3
  loss <- (res/(n.s*n.t))
  
  #Log and terminate early on NaN or diverging loss
  if(!is.finite(loss$item())) break
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   ELBO: ", loss$item(), "\n") # we want to maximize

  ### -------- Backpropagation --------
  # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
  loss$backward()

  ### -------- Update Parameters --------
  # Perform parameter updates with momentum
  
  # Wrap in with_no_grad() because this is a part we DON'T 
  # want to record for automatic gradient computation
  with_no_grad({
    old_loss <- loss$item()
    w_1_velocity <- alpha_v*w_1_velocity - learning_rate*w_1$grad
    w_1$add_(w_1_velocity)
    w_2_velocity <- alpha_v*w_2_velocity - learning_rate*w_2$grad
    w_2$add_(w_2_velocity)
    w_3_velocity <- alpha_v*w_3_velocity - learning_rate*w_3$grad
    w_3$add_(w_3_velocity)
    w_4_velocity <- alpha_v*w_4_velocity - learning_rate*w_4$grad
    w_4$add_(w_4_velocity)
    b_1_velocity <- alpha_v*b_1_velocity - learning_rate*b_1$grad
    b_1$add_(b_1_velocity)
    b_2_velocity <- alpha_v*b_2_velocity - learning_rate*b_2$grad
    b_2$add_(b_2_velocity)
    b_3_velocity <- alpha_v*b_3_velocity - learning_rate*b_3$grad
    b_3$add_(b_3_velocity)
    b_4_velocity <- alpha_v*b_4_velocity - learning_rate*b_4$grad
    b_4$add_(b_4_velocity)
    w_1_prime_velocity <- alpha_v*w_1_prime_velocity - learning_rate*w_1_prime$grad
    w_1_prime$add_(w_1_prime_velocity)
    w_2_prime_velocity <- alpha_v*w_2_prime_velocity - learning_rate*w_2_prime$grad
    w_2_prime$add_(w_2_prime_velocity)
    w_3_prime_velocity <- alpha_v*w_3_prime_velocity - learning_rate*w_3_prime$grad
    w_3_prime$add_(w_3_prime_velocity)
    w_4_prime_velocity <- alpha_v*w_4_prime_velocity - learning_rate*w_4_prime$grad
    w_4_prime$add_(w_4_prime_velocity)
    b_1_prime_velocity <- alpha_v*b_1_prime_velocity - learning_rate*b_1_prime$grad
    b_1_prime$add_(b_1_prime_velocity)
    b_2_prime_velocity <- alpha_v*b_2_prime_velocity - learning_rate*b_2_prime$grad
    b_2_prime$add_(b_2_prime_velocity)
    b_3_prime_velocity <- alpha_v*b_3_prime_velocity - learning_rate*b_3_prime$grad
    b_3_prime$add_(b_3_prime_velocity)
    b_4_prime_velocity <- alpha_v*b_4_prime_velocity - learning_rate*b_4_prime$grad
    b_4_prime$add_(b_4_prime_velocity)
    
    w_5_velocity <- alpha_v*w_5_velocity - learning_rate*w_5$grad
    w_5$add_(w_5_velocity)
    w_6_velocity <- alpha_v*w_6_velocity - learning_rate*w_6$grad
    w_6$add_(w_6_velocity)
    w_7_velocity <- alpha_v*w_7_velocity - learning_rate*w_7$grad
    w_7$add_(w_7_velocity)
    b_5_velocity <- alpha_v*b_5_velocity - learning_rate*b_5$grad
    b_5$add_(b_5_velocity)
    b_6_velocity <- alpha_v*b_6_velocity - learning_rate*b_6$grad
    b_6$add_(b_6_velocity)
    b_7_velocity <- alpha_v*b_7_velocity - learning_rate*b_7$grad
    b_7$add_(b_7_velocity)
    b_8_velocity <- alpha_v*b_8_velocity - learning_rate*b_8$grad
    b_8$add_(b_8_velocity)
    
    # Zero gradients after every pass, as they'd accumulate otherwise
    w_1$grad$zero_()
    w_2$grad$zero_()
    w_3$grad$zero_()
    w_4$grad$zero_()
    b_1$grad$zero_()
    b_2$grad$zero_()
    b_3$grad$zero_()
    b_4$grad$zero_()
    w_1_prime$grad$zero_()
    w_2_prime$grad$zero_()
    w_3_prime$grad$zero_()
    w_4_prime$grad$zero_()
    b_1_prime$grad$zero_()
    b_2_prime$grad$zero_()
    b_3_prime$grad$zero_()
    b_4_prime$grad$zero_()
    w_5$grad$zero_()
    w_6$grad$zero_()
    w_7$grad$zero_()
    b_5$grad$zero_()
    b_6$grad$zero_()
    b_7$grad$zero_()
    b_8$grad$zero_()
  })
}
```

### 5. Post-Processing and Results

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
