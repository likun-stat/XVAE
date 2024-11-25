library(dplyr)
library(fields)
library(VGAM)
library(torch)

#' ReLU Activation Function
#'
#' Implements the Rectified Linear Unit (ReLU) activation function, which is widely used in neural networks.
#' This function outputs the input `x` if it is positive; otherwise, it returns `0`.
#'
#' @param x A numeric vector, matrix, or array to apply the ReLU activation.
#' @return A numeric object of the same dimensions as `x`, with all negative values replaced by `0`.
#' @examples
#' relu(c(-3, 0, 2))  # Returns c(0, 0, 2)
#' @export
relu <- function(x) {
  return(pmax(0, x))
  # Alternative: return(log(1 + exp(x))) for smoother approximation
}

#' Leaky ReLU Activation Function
#'
#' Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
#' This function outputs the input `x` if it is positive, and a small slope 
#' times `x` (specified by `slope`) if `x` is negative.
#'
#' @param x A numeric matrix or array where the activation function is applied column-wise.
#' @param slope A numeric value specifying the slope for negative inputs. Default values are typically small (e.g., 0.01).
#' @return A numeric object of the same dimensions as `x`, with modified values based on the Leaky ReLU activation.
#' @examples
#' leaky_relu(matrix(c(-1, 2, -3, 4), nrow = 2), slope = 0.01)
#' # Returns a matrix with values adjusted for Leaky ReLU
#' @export
leaky_relu <- function(x, slope) {
  res <- apply(x, 2, function(x) return(pmax(0, x) + slope * pmin(0, x)))
  return(res)
  # Alternative: return(log(1 + exp(x))) for a smoother approximation
}



#' Wendland Radial Basis Function
#'
#' This function evaluates the Wendland radial basis function with parameters \code{s = 2} and \code{k = 1}.
#' It is compactly supported, meaning the function value is zero for distances greater than \code{r}.
#'
#' @param d A vector or matrix of nonnegative distances.
#' @param r A nonnegative scalar representing the radius of influence.
#' 
#' @return A vector or matrix of Wendland function values, where the function is zero for distances greater than \code{r}.
#'
#' @details The Wendland radial basis function is commonly used in spatial statistics and machine learning for interpolation and smoothing.
#' This implementation assumes parameters \code{s = 2} and \code{k = 1}.
#' 
#' @examples
#' # Example: Compute Wendland function values for a vector of distances
#' distances <- c(0, 0.5, 1, 1.5, 2)
#' radius <- 1
#' wendland(distances, radius)
#'
#' @export
wendland <- function(d, r) {
  if (any(d < 0)) 
    stop("d must be nonnegative") # Ensure distances are valid
  
  return((1 - d/r)^2 * (d < r))
}




#' Calculate Total Within-Cluster Sum of Squares (WSS) for k-Means Clustering
#'
#' This function computes the total WSS for a specified number of clusters (`k`) 
#' on a given dataset. It is commonly used to evaluate clustering performance 
#' and select an optimal number of clusters.
#'
#' @param k Integer. The number of clusters for the k-means algorithm.
#' @param df Data frame or matrix. The dataset to be clustered, where rows are observations 
#'           and columns are features.
#' @return Numeric. The total within-cluster sum of squares (WSS) for the specified `k`.
#'
#' @examples
#' wss(3, iris[, -5])
#' @export

wss <- function(k, df) {
  kmeans(df, k, nstart = 10)$tot.withinss
}




#' Initialize the XVAE Model Components
#'
#' This function initializes the encoder and decoder components of an XVAE,
#' including weights, biases, and tensors for the forward pass, reparameterization trick, and likelihood evaluation.
#'
#' @details 
#' The initialization process consists of the following steps:
#' 1. **Encoder Initialization**:
#'    - Converts input matrices (`X` and `W`) to PyTorch tensors.
#'    - Computes initial weights for the encoder using a least squares approximation.
#'    - Initializes biases and weights for all layers of the encoder.
#'    - Implements forward passes through the encoder network with ReLU activation.
#'    - Computes latent variables (`mu` and `sigma_sq`) and uses the reparameterization trick to sample latent variables.
#'
#' 2. **Decoder Initialization**:
#'    - Mirrors the structure of the encoder with matching weights and biases for reconstruction.
#'    - Sets up additional layers with identity matrices and small biases for robust decoding.
#'
#' 3. **Momentum Initialization**:
#'    - Initializes momentum tensors (`velocity`) for all weights and biases to support gradient updates.
#'
#' 4. **Likelihood Evaluation**:
#'    - Precomputes constants and tensors for evaluating the explicit power stable (expPS) likelihood using Zolotarev's approximation.
#'
#' @return None. The function uses global assignment (`<<-`) to initialize variables in the global environment.
#' These include:
#' - `w_1, w_2, ..., w_7`: Weight tensors for encoder and decoder layers.
#' - `b_1, b_2, ..., b_8`: Bias tensors for encoder and decoder layers.
#' - `v_t`: Latent variable samples from the reparameterization trick.
#' - `y_approx`: Reconstructed output of the XVAE.
#' - Precomputed tensors (`Zolo_vec`, `W_alpha_tensor`) for likelihood evaluation.
#'
#' @note 
#' - This function relies on PyTorch tensors for computations, requiring the `torch` package in R.
#' - Global assignments (`<<-`) are used extensively, which may impact modularity and debugging.
#'
#' @examples
#' # Ensure X, W, and Z_approx are defined in the global environment
#' XVAE_initialization()
#'
#' @seealso [torch_tensor()], [relu()], [exp()]
#' @import torch
#' @export

XVAE_initialization <- function(alpha=0.5){
  ## -------------------- Initializing Encoder --------------------
  # Convert data matrices X and W to PyTorch tensors with float dtype.
  X_tensor <<- torch_tensor(X, dtype=torch_float())
  W_tensor <<- torch_tensor(W, dtype=torch_float())
  
  # Compute initial weights for the encoder's first layer using a least squares solution.
  tmp <<- qr.solve(a=t(X), b=t(Z_approx))
  w_1 <<- t(tmp)
  w_1 <<- torch_tensor(w_1, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize biases for the first layer of the encoder with zeros.
  b_1 <<- matrix(rep(0, k), ncol=1)
  b_1 <<- torch_tensor(b_1, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize weights and biases for additional layers of the encoder.
  # Layer 2: weights initialized as identity matrix, biases slightly positive.
  w_2 <<- diag(k)
  w_2 <<- torch_tensor(w_2, dtype=torch_float(), requires_grad = TRUE)
  b_2 <<- matrix(rep(0.00001, k), ncol=1)
  b_2 <<- torch_tensor(b_2, dtype=torch_float(), requires_grad = TRUE)
  
  # Layer 3: weights initialized to zero, biases to a negative value (-15).
  w_3 <<- 0 * diag(k)
  w_3 <<- torch_tensor(w_3, dtype=torch_float(), requires_grad = TRUE)
  b_3 <<- matrix(rep(-15, k), ncol=1)
  b_3 <<- torch_tensor(b_3, dtype=torch_float(), requires_grad = TRUE)
  
  # Layer 4: weights initialized as identity matrix, biases as zeros.
  w_4 <<- diag(k)
  w_4 <<- torch_tensor(w_4, dtype=torch_float(), requires_grad = TRUE)
  b_4 <<- matrix(rep(0, k), ncol=1)
  b_4 <<- torch_tensor(b_4, dtype=torch_float(), requires_grad = TRUE)
  
  # Forward pass through the encoder network with ReLU activation.
  h <<- w_1$mm(X_tensor)$add(b_1)$relu()
  h_1 <<- w_2$mm(h)$add(b_2)$relu()
  
  # Compute latent variables: mean (mu) and variance (sigma^2).
  sigma_sq_vec <<- w_3$mm(h_1)$add(b_3)$exp()
  mu <<- w_4$mm(h_1)$add(b_4)$relu()
  
  ## -------------------- Re-parameterization trick --------------------
  # Introduce random noise (Epsilon) for the VAE re-parameterization trick.
  Epsilon <<- matrix(abs(rnorm(k * n.t)) + 0.1, nrow=k)
  Epsilon <<- torch_tensor(Epsilon, dtype=torch_float())
  
  # Generate latent variable samples (v_t) using mu and sigma^2.
  v_t <<- mu + sqrt(sigma_sq_vec) * Epsilon
  
  # Adjust latent samples using weight matrix W and apply ReLU to correct for residuals.
  b_8 <<- as_array((W_tensor^(1/alpha))) %*% as_array(v_t) - X
  b_8 <<- array(-relu(b_8), dim=dim(b_8))
  b_8 <<- torch_tensor(b_8, dtype=torch_float(), requires_grad = TRUE)
  
  # Reconstructed output from the encoder.
  y_approx <<- (W_tensor^(1/alpha))$mm(v_t) + b_8
  
  ## -------------------- Initializing Decoder --------------------
  # Initialize weights and biases for the decoder, mirroring the encoder structure.
  w_1_prime <<- as_array(w_1)
  w_1_prime <<- torch_tensor(w_1_prime, dtype=torch_float(), requires_grad = TRUE)
  w_2_prime <<- matrix(diag(k), nrow=k)
  w_2_prime <<- torch_tensor(w_2_prime, dtype=torch_float(), requires_grad = TRUE)
  w_3_prime <<- matrix(rep(0, k * k), nrow=k)
  w_3_prime <<- torch_tensor(w_3_prime, dtype=torch_float(), requires_grad = TRUE)
  w_4_prime <<- matrix(diag(k), nrow=k)
  w_4_prime <<- torch_tensor(w_4_prime, dtype=torch_float(), requires_grad = TRUE)
  
  b_1_prime <<- matrix(rep(0, k), ncol=1)
  b_1_prime <<- torch_tensor(b_1_prime, dtype=torch_float(), requires_grad = TRUE)
  b_2_prime <<- matrix(rep(0.05, k), ncol=1)
  b_2_prime <<- torch_tensor(b_2_prime, dtype=torch_float(), requires_grad = TRUE)
  b_3_prime <<- matrix(rep(-10, k), ncol=1)
  b_3_prime <<- torch_tensor(b_3_prime, dtype=torch_float(), requires_grad = TRUE)
  b_4_prime <<- matrix(rep(0, k), ncol=1)
  b_4_prime <<- torch_tensor(b_4_prime, dtype=torch_float(), requires_grad = TRUE)
  
  # Additional weights for the decoder, initialized as identity matrices.
  w_5 <<- diag(k)
  w_5 <<- torch_tensor(w_5, dtype=torch_float(), requires_grad = TRUE)
  w_6 <<- diag(k)
  w_6 <<- torch_tensor(w_6, dtype=torch_float(), requires_grad = TRUE)
  w_7 <<- diag(k)
  w_7 <<- torch_tensor(w_7, dtype=torch_float(), requires_grad = TRUE)
  
  # Biases for the additional decoder layers, initialized with small positive values.
  b_5 <<- matrix(rep(1e-6, k), ncol=1)
  b_5 <<- torch_tensor(b_5, dtype=torch_float(), requires_grad = TRUE)
  b_6 <<- matrix(rep(1e-6, k), ncol=1)
  b_6 <<- torch_tensor(b_6, dtype=torch_float(), requires_grad = TRUE)
  b_7 <<- matrix(rep(1e-6, k), ncol=1)
  b_7 <<- torch_tensor(b_7, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize momentum tensors (velocities) for gradient updates.
  w_1_velocity <<- torch_zeros(w_1$size())
  w_2_velocity <<- torch_zeros(w_2$size())
  w_3_velocity <<- torch_zeros(w_3$size())
  w_4_velocity <<- torch_zeros(w_4$size())
  b_1_velocity <<- torch_zeros(b_1$size())
  b_2_velocity <<- torch_zeros(b_2$size())
  b_3_velocity <<- torch_zeros(b_3$size())
  b_4_velocity <<- torch_zeros(b_4$size())
  w_1_prime_velocity <<- torch_zeros(w_1_prime$size())
  w_2_prime_velocity <<- torch_zeros(w_2_prime$size())
  w_3_prime_velocity <<- torch_zeros(w_3_prime$size())
  w_4_prime_velocity <<- torch_zeros(w_4_prime$size())
  b_1_prime_velocity <<- torch_zeros(b_1_prime$size())
  b_2_prime_velocity <<- torch_zeros(b_2_prime$size())
  b_3_prime_velocity <<- torch_zeros(b_3_prime$size())
  b_4_prime_velocity <<- torch_zeros(b_4_prime$size())
  
  w_5_velocity <<- torch_zeros(w_5$size())
  w_6_velocity <<- torch_zeros(w_6$size())
  w_7_velocity <<- torch_zeros(w_7$size())
  b_5_velocity <<- torch_zeros(b_5$size())
  b_6_velocity <<- torch_zeros(b_6$size())
  b_7_velocity <<- torch_zeros(b_7$size())
  b_8_velocity <<- torch_zeros(b_8$size())
  
  ## -------------------- For evaluating expPS likelihood --------------------
  # Precompute terms for likelihood evaluation using Zolotarev's approximation.
  n_zolo <<- 1e3
  vec <<- Zolo_A(pi*seq(1/2,n_zolo-1/2,1)/n_zolo, alpha)
  Zolo_vec <<- torch_tensor(matrix(vec, nrow=1,ncol=n_zolo), dtype=torch_float(), requires_grad = FALSE) 
  Zolo_vec_double <<- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
  const <<- 1/(1-alpha); const1 <<- 1/(1-alpha)-1; const3 <<- log(const1)
  W_alpha_tensor <<- W_tensor$pow(1/alpha)
}



#' XVAE Training Loop
#'
#' This function performs training for XVAE.
#' It implements forward passes, evidence lower bound (ELBO) computation, and 
#' parameter updates using backpropagation.
#'
#' @details
#' The `XVAE_training_loop` function performs the following key steps for training:
#' 
#' - **Learning Rate and Momentum Adjustment**: Dynamically adjusts the learning rate and momentum
#'   at regular intervals based on the logarithmic scale of the epoch count.
#' 
#' - **Forward Pass**:
#'   - Random noise is generated for the reparameterization trick.
#'   - Encoders are applied to generate latent space representations for primary and auxiliary spaces.
#'   - Decoding is performed to reconstruct the input data.
#' 
#' - **ELBO Computation**:
#'   - Reconstruction loss: Measures the error in reconstructing input data.
#'   - Latent space regularization: Enforces structure in the learned latent representations.
#'   - Gaussian prior penalty: Regularizes the latent variables using a Gaussian prior.
#' 
#' - **Backpropagation**:
#'   - Gradients of the ELBO loss with respect to all parameters are computed.
#' 
#' - **Parameter Updates**:
#'   - Parameters are updated using momentum-based gradient descent.
#'   - Gradients are reset after each update.
#' 
#' @note
#' The function modifies several global variables using the `<<-` operator. These include parameters
#' (e.g., weights and biases) and intermediate variables used during the training process.
#' 
#' @examples
#' # Initialize required global variables before running the function
#' XVAE_training_loop()
#' 
#' @seealso
#' [torch](https://torch.mlverse.org) for tensor operations, 
#' [mvtnorm::rmvnorm()] for generating multivariate normal noise.
#' @import torch
#' @export

XVAE_training_loop <- function(nEpoch, learning_rate, alpha_v){
  ## -------------------- Initializing Encoder --------------------
  # Convert data matrices X and W to PyTorch tensors with float dtype.
  X_tensor <<- torch_tensor(X, dtype=torch_float())
  W_tensor <<- torch_tensor(W, dtype=torch_float())
  
  # Compute initial weights for the encoder's first layer using a least squares solution.
  tmp <<- qr.solve(a=t(X), b=t(Z_approx))
  w_1 <<- t(tmp)
  w_1 <<- torch_tensor(w_1, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize biases for the first layer of the encoder with zeros.
  b_1 <<- matrix(rep(0, k), ncol=1)
  b_1 <<- torch_tensor(b_1, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize weights and biases for additional layers of the encoder.
  # Layer 2: weights initialized as identity matrix, biases slightly positive.
  w_2 <<- diag(k)
  w_2 <<- torch_tensor(w_2, dtype=torch_float(), requires_grad = TRUE)
  b_2 <<- matrix(rep(0.00001, k), ncol=1)
  b_2 <<- torch_tensor(b_2, dtype=torch_float(), requires_grad = TRUE)
  
  # Layer 3: weights initialized to zero, biases to a negative value (-15).
  w_3 <<- 0 * diag(k)
  w_3 <<- torch_tensor(w_3, dtype=torch_float(), requires_grad = TRUE)
  b_3 <<- matrix(rep(-15, k), ncol=1)
  b_3 <<- torch_tensor(b_3, dtype=torch_float(), requires_grad = TRUE)
  
  # Layer 4: weights initialized as identity matrix, biases as zeros.
  w_4 <<- diag(k)
  w_4 <<- torch_tensor(w_4, dtype=torch_float(), requires_grad = TRUE)
  b_4 <<- matrix(rep(0, k), ncol=1)
  b_4 <<- torch_tensor(b_4, dtype=torch_float(), requires_grad = TRUE)
  
  # Forward pass through the encoder network with ReLU activation.
  h <<- w_1$mm(X_tensor)$add(b_1)$relu()
  h_1 <<- w_2$mm(h)$add(b_2)$relu()
  
  # Compute latent variables: mean (mu) and variance (sigma^2).
  sigma_sq_vec <<- w_3$mm(h_1)$add(b_3)$exp()
  mu <<- w_4$mm(h_1)$add(b_4)$relu()
  
  ## -------------------- Re-parameterization trick --------------------
  # Introduce random noise (Epsilon) for the VAE re-parameterization trick.
  Epsilon <<- matrix(abs(rnorm(k * n.t)) + 0.1, nrow=k)
  Epsilon <<- torch_tensor(Epsilon, dtype=torch_float())
  
  # Generate latent variable samples (v_t) using mu and sigma^2.
  v_t <<- mu + sqrt(sigma_sq_vec) * Epsilon
  
  # Adjust latent samples using weight matrix W and apply ReLU to correct for residuals.
  b_8 <<- as_array((W_tensor^(1/alpha))) %*% as_array(v_t) - X
  b_8 <<- array(-relu(b_8), dim=dim(b_8))
  b_8 <<- torch_tensor(b_8, dtype=torch_float(), requires_grad = TRUE)
  
  # Reconstructed output from the encoder.
  y_approx <<- (W_tensor^(1/alpha))$mm(v_t) + b_8
  
  ## -------------------- Initializing Decoder --------------------
  # Initialize weights and biases for the decoder, mirroring the encoder structure.
  w_1_prime <<- as_array(w_1)
  w_1_prime <<- torch_tensor(w_1_prime, dtype=torch_float(), requires_grad = TRUE)
  w_2_prime <<- matrix(diag(k), nrow=k)
  w_2_prime <<- torch_tensor(w_2_prime, dtype=torch_float(), requires_grad = TRUE)
  w_3_prime <<- matrix(rep(0, k * k), nrow=k)
  w_3_prime <<- torch_tensor(w_3_prime, dtype=torch_float(), requires_grad = TRUE)
  w_4_prime <<- matrix(diag(k), nrow=k)
  w_4_prime <<- torch_tensor(w_4_prime, dtype=torch_float(), requires_grad = TRUE)
  
  b_1_prime <<- matrix(rep(0, k), ncol=1)
  b_1_prime <<- torch_tensor(b_1_prime, dtype=torch_float(), requires_grad = TRUE)
  b_2_prime <<- matrix(rep(0.05, k), ncol=1)
  b_2_prime <<- torch_tensor(b_2_prime, dtype=torch_float(), requires_grad = TRUE)
  b_3_prime <<- matrix(rep(-10, k), ncol=1)
  b_3_prime <<- torch_tensor(b_3_prime, dtype=torch_float(), requires_grad = TRUE)
  b_4_prime <<- matrix(rep(0, k), ncol=1)
  b_4_prime <<- torch_tensor(b_4_prime, dtype=torch_float(), requires_grad = TRUE)
  
  # Additional weights for the decoder, initialized as identity matrices.
  w_5 <<- diag(k)
  w_5 <<- torch_tensor(w_5, dtype=torch_float(), requires_grad = TRUE)
  w_6 <<- diag(k)
  w_6 <<- torch_tensor(w_6, dtype=torch_float(), requires_grad = TRUE)
  w_7 <<- diag(k)
  w_7 <<- torch_tensor(w_7, dtype=torch_float(), requires_grad = TRUE)
  
  # Biases for the additional decoder layers, initialized with small positive values.
  b_5 <<- matrix(rep(1e-6, k), ncol=1)
  b_5 <<- torch_tensor(b_5, dtype=torch_float(), requires_grad = TRUE)
  b_6 <<- matrix(rep(1e-6, k), ncol=1)
  b_6 <<- torch_tensor(b_6, dtype=torch_float(), requires_grad = TRUE)
  b_7 <<- matrix(rep(1e-6, k), ncol=1)
  b_7 <<- torch_tensor(b_7, dtype=torch_float(), requires_grad = TRUE)
  
  # Initialize momentum tensors (velocities) for gradient updates.
  w_1_velocity <<- torch_zeros(w_1$size())
  w_2_velocity <<- torch_zeros(w_2$size())
  w_3_velocity <<- torch_zeros(w_3$size())
  w_4_velocity <<- torch_zeros(w_4$size())
  b_1_velocity <<- torch_zeros(b_1$size())
  b_2_velocity <<- torch_zeros(b_2$size())
  b_3_velocity <<- torch_zeros(b_3$size())
  b_4_velocity <<- torch_zeros(b_4$size())
  w_1_prime_velocity <<- torch_zeros(w_1_prime$size())
  w_2_prime_velocity <<- torch_zeros(w_2_prime$size())
  w_3_prime_velocity <<- torch_zeros(w_3_prime$size())
  w_4_prime_velocity <<- torch_zeros(w_4_prime$size())
  b_1_prime_velocity <<- torch_zeros(b_1_prime$size())
  b_2_prime_velocity <<- torch_zeros(b_2_prime$size())
  b_3_prime_velocity <<- torch_zeros(b_3_prime$size())
  b_4_prime_velocity <<- torch_zeros(b_4_prime$size())
  
  w_5_velocity <<- torch_zeros(w_5$size())
  w_6_velocity <<- torch_zeros(w_6$size())
  w_7_velocity <<- torch_zeros(w_7$size())
  b_5_velocity <<- torch_zeros(b_5$size())
  b_6_velocity <<- torch_zeros(b_6$size())
  b_7_velocity <<- torch_zeros(b_7$size())
  b_8_velocity <<- torch_zeros(b_8$size())
  
  ## -------------------- For evaluating expPS likelihood --------------------
  # Precompute terms for likelihood evaluation using Zolotarev's approximation.
  n_zolo <<- 1e3
  vec <<- Zolo_A(pi*seq(1/2,n_zolo-1/2,1)/n_zolo, alpha)
  Zolo_vec <<- torch_tensor(matrix(vec, nrow=1,ncol=n_zolo), dtype=torch_float(), requires_grad = FALSE) 
  Zolo_vec_double <<- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
  const <<- 1/(1-alpha); const1 <<- 1/(1-alpha)-1; const3 <<- log(const1)
  W_alpha_tensor <<- W_tensor$pow(1/alpha)
  
  old_loss <- -Inf
  current_interval <- 0
  for (t in 1:nEpoch) {
    # Adapt learning rate and momentum at regular intervals
    tmp_interval <- round(log2(t))
    if(tmp_interval%%4 == 0 & tmp_interval>current_interval) { learning_rate <- 2 * learning_rate; alpha_v <- 0.99*alpha_v; current_interval <- tmp_interval}
    
    ### -------- Forward pass --------
    # Generate random noise for reparameterization trick
    Epsilon <<- t(0.1+abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
    Epsilon_prime <<- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
    Epsilon <<- torch_tensor(Epsilon,dtype=torch_float())
    Epsilon_prime <<- torch_tensor(Epsilon_prime,dtype=torch_float())
    
    
    ### -------- Encoder for Primary Latent Space --------
    h <<- w_1$mm(X_tensor)$add(b_1)$relu()
    h_1 <<- w_2$mm(h)$add(b_2)$relu()
    sigma_sq_vec <<- w_3$mm(h_1)$add(b_3)$exp()
    mu <<- w_4$mm(h_1)$add(b_4)$relu()
    
    ### -------- Encoder for Auxiliary Latent Space --------
    # Similar encoding process for auxiliary representation `v_t_prime`
    h_prime <<- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
    h_1_prime <<- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()
    
    ### -------- Activation via Laplace transformation --------
    h_1_prime_laplace <<- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
    h_1_prime_t <<- h_1_prime_laplace$log()$multiply(-1)
    h_1_prime_to_theta <<- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
    theta_propagate <<- h_1_prime_to_theta$expand(c(k,n.t))
    
    sigma_sq_vec_prime <<- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
    mu_prime <<- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)
    
    ### -------- Re-parameterization trick --------
    v_t <<- mu + sqrt(sigma_sq_vec)*Epsilon
    v_t_prime <<- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
    
    
    ### -------- Decoder --------
    # Decode auxiliary latent variables to reconstruct input
    l <<- w_5$mm(v_t_prime)$add(b_5)$relu()
    l_1 <<- w_6$mm(l)$add(b_6)$relu()
    theta_t <<- w_7$mm(l_1)$add(b_7)$relu()
    
    # Apply leaky ReLU activation for final output
    y_star <<- lrelu(W_alpha_tensor$mm(v_t)$add(b_8))
    
    
    ### -------- Evidence Lower Bound (ELBO) -------- 
    # Compute reconstruction loss (Part 1)
    standardized <<- X_tensor$divide(y_star)$sub(m)
    leak <<- as_array(sum(standardized<0))
    if(leak>0 & leak<=200) standardized$abs_()
    leak2 <<- as_array(sum(standardized==0))
    if(leak2>0 & leak2<=120) standardized$add_(1e-07)
    part1 <<- -2 * standardized$log()$sum() - y_star$log()$sum() - tau*standardized$pow(-1)$sum() # + n.s*n.t*log(tau)
    
    # Compute latent space regularization (Part 2)
    V_t <<- v_t$view(c(k*n.t,1))
    Theta_t <<- theta_t$view(c(k*n.t,1))
    part_log_v1  <<- V_t$pow(-const)$mm(Zolo_vec) 
    part_log_v2  <<- (-V_t$pow(-const1)$mm(Zolo_vec))$exp()
    part_log_v3 <<- Theta_t$pow(alpha)-Theta_t$mul(V_t)
    part2 <<- (part_log_v1$mul(part_log_v2)$mean(dim=2)$log()+part_log_v3$view(k*n.t)$add(const3))$sum()
    if(as_array(part2) == -Inf) {
      part2_tmp <<- part_log_v1$log()[,1] + (-V_t$pow(-const1)$mm(Zolo_vec))[,1]
      part2 <<- (part2_tmp+part_log_v3$view(k*n.t)$add(const3))$sum()
    }
    
    # Compute Gaussian prior penalty (Part 3)
    part3 = Epsilon$pow(2)$sum()/2 + Epsilon_prime$pow(2)$sum()/2 + sigma_sq_vec$log()$sum() + sigma_sq_vec_prime$log()$sum()
    
    # Aggregate ELBO
    res <<- part1 + part2 + part3
    loss <<- (res/(n.s*n.t))
    
    #Log and terminate early on NaN or diverging loss
    if(!is.finite(loss$item())) break
    if (t %% 100 == 0)
      cat("Epoch: ", t, "   ELBO: ", loss$item(), "\n") # we want to maximize
    
    ### -------- Backpropagation --------
    # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
    loss$backward()
    
    ### -------- Update Parameters --------
    # Perform parameter updates with momentum
    
    # Wrap in with_no_grad() because this is a part we DON'T 
    # want to record for automatic gradient computation
    with_no_grad({
      old_loss <<- loss$item()
      w_1_velocity <<- alpha_v*w_1_velocity - learning_rate*w_1$grad
      w_1$add_(w_1_velocity)
      w_2_velocity <<- alpha_v*w_2_velocity - learning_rate*w_2$grad
      w_2$add_(w_2_velocity)
      w_3_velocity <<- alpha_v*w_3_velocity - learning_rate*w_3$grad
      w_3$add_(w_3_velocity)
      w_4_velocity <<- alpha_v*w_4_velocity - learning_rate*w_4$grad
      w_4$add_(w_4_velocity)
      b_1_velocity <<- alpha_v*b_1_velocity - learning_rate*b_1$grad
      b_1$add_(b_1_velocity)
      b_2_velocity <<- alpha_v*b_2_velocity - learning_rate*b_2$grad
      b_2$add_(b_2_velocity)
      b_3_velocity <<- alpha_v*b_3_velocity - learning_rate*b_3$grad
      b_3$add_(b_3_velocity)
      b_4_velocity <<- alpha_v*b_4_velocity - learning_rate*b_4$grad
      b_4$add_(b_4_velocity)
      w_1_prime_velocity <<- alpha_v*w_1_prime_velocity - learning_rate*w_1_prime$grad
      w_1_prime$add_(w_1_prime_velocity)
      w_2_prime_velocity <<- alpha_v*w_2_prime_velocity - learning_rate*w_2_prime$grad
      w_2_prime$add_(w_2_prime_velocity)
      w_3_prime_velocity <<- alpha_v*w_3_prime_velocity - learning_rate*w_3_prime$grad
      w_3_prime$add_(w_3_prime_velocity)
      w_4_prime_velocity <<- alpha_v*w_4_prime_velocity - learning_rate*w_4_prime$grad
      w_4_prime$add_(w_4_prime_velocity)
      b_1_prime_velocity <<- alpha_v*b_1_prime_velocity - learning_rate*b_1_prime$grad
      b_1_prime$add_(b_1_prime_velocity)
      b_2_prime_velocity <<- alpha_v*b_2_prime_velocity - learning_rate*b_2_prime$grad
      b_2_prime$add_(b_2_prime_velocity)
      b_3_prime_velocity <<- alpha_v*b_3_prime_velocity - learning_rate*b_3_prime$grad
      b_3_prime$add_(b_3_prime_velocity)
      b_4_prime_velocity <<- alpha_v*b_4_prime_velocity - learning_rate*b_4_prime$grad
      b_4_prime$add_(b_4_prime_velocity)
      
      w_5_velocity <<- alpha_v*w_5_velocity - learning_rate*w_5$grad
      w_5$add_(w_5_velocity)
      w_6_velocity <<- alpha_v*w_6_velocity - learning_rate*w_6$grad
      w_6$add_(w_6_velocity)
      w_7_velocity <<- alpha_v*w_7_velocity - learning_rate*w_7$grad
      w_7$add_(w_7_velocity)
      b_5_velocity <<- alpha_v*b_5_velocity - learning_rate*b_5$grad
      b_5$add_(b_5_velocity)
      b_6_velocity <<- alpha_v*b_6_velocity - learning_rate*b_6$grad
      b_6$add_(b_6_velocity)
      b_7_velocity <<- alpha_v*b_7_velocity - learning_rate*b_7$grad
      b_7$add_(b_7_velocity)
      b_8_velocity <<- alpha_v*b_8_velocity - learning_rate*b_8$grad
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
}




#' Derive Data-Driven Knots for Spatio-Temporal Models
#'
#' This function identifies data-driven knots (spatial locations) based on high values in 
#' the input data matrix and performs clustering to determine knot locations. Knots are 
#' used as the basis for constructing spatial basis functions in subsequent modeling steps.
#'
#' @param X Matrix. Observed data, where rows correspond to spatial sites and columns to time replicates.
#' @param stations Data frame or matrix. Spatial coordinates of observation sites, where rows correspond to sites.
#' @param threshold_p Numeric. The quantile level (e.g., 0.96) used to threshold high values in `X`.
#' @param echo Logical. If `TRUE`, progress messages are printed during execution.
#' @param start.knot.num Integer. Initial number of knots. If `NULL`, it defaults to approximately 
#'                       `5 * log(nrow(stations))`.
#' @return Matrix. A refined set of knot locations, where rows are coordinates.
#'
#' @details
#' 1. High values in `X` exceeding the specified quantile are identified for each time replicate.
#' 2. A clustering algorithm (`k-means`) is applied to the spatial locations of these high values 
#'    to identify clusters, with the number of clusters determined dynamically based on the WSS.
#' 3. The center within each cluster is chosen as a knot.
#' 4. Additional coarse grid knots are generated, and all knot candidates are refined to ensure 
#'    adequate spacing by eliminating close points.
#'
#' @examples
#' # Load example data
#' data(example_X.RData)
#' knots <- data_driven_knots(X, stations, threshold_p = 0.96)
#' @export
data_driven_knots <- function(X, stations, threshold_p, echo=FALSE, start.knot.num = NULL){
  # Set the default number of knots if not specified
  if (is.null(start.knot.num)) 
    start.knot.num <- round(5 * log(nrow(stations)))
  
  # Define the threshold based on the quantile of X
  threshold <- quantile(X, probs = threshold_p)
  data.knots <- data.frame(x = NA, y = NA)
  
  # Loop through each time replicate
  for (iter.t in 1:ncol(X)) {
    # Identify sites exceeding the threshold
    where.exceed <- which(X[, iter.t] > threshold)
    
    # Only proceed if there are enough exceedances
    if (length(where.exceed) > 10) {
      # Determine the range of cluster counts to test
      tmp.min <- min(15, length(where.exceed) - 1)
      k.values <- 1:tmp.min
      tmp_df <- stations[where.exceed, ]
      tmp.obs <- X[where.exceed, iter.t]
      
      # Compute WSS for potential cluster counts
      wss_values <- unlist(lapply(k.values, wss, df = tmp_df))
      
      # Identify the optimal number of clusters
      n.clusters <- which(wss_values / wss_values[1] < 0.15)[1]
      
      # Perform k-means clustering with the optimal number of clusters
      res <- kmeans(tmp_df, n.clusters, nstart = 10)
      
      # Select the point with the highest value within each cluster as a knot
      for (tmp.iter in 1:n.clusters) {
        where.max <- which.max(tmp.obs[res$cluster == tmp.iter])
        data.knots <- rbind(data.knots, tmp_df[where.max, ])
      }
    }
  }
  
  # Remove the placeholder first row
  data.knots <- data.knots[-1, ]
  
  # Refine knots by clustering them again
  res <- kmeans(data.knots, start.knot.num, nstart = 10)
  
  # Generate additional coarse grid knot candidates
  coarse.grid.length <- round(sqrt(start.knot.num))
  knot_candidates <- as.matrix(expand.grid(
    x = seq(min(stations[, 1]), max(stations[, 1]), length.out = coarse.grid.length),
    y = seq(min(stations[, 2]), max(stations[, 2]), length.out = coarse.grid.length)
  ))
  
  # Combine cluster centers and coarse grid knots
  knots <- rbind(knot_candidates, res$centers)
  rownames(knots) <- NULL
  
  # Calculate pairwise distances and eliminate close knots
  distances <- fields::rdist(knots)
  min.gap <- max(distances) / 30
  where.close <- which(distances < min.gap & distances > 0, arr.ind = TRUE)
  eliminate <- c()
  if (nrow(where.close) > 0) {
    for (tmp.iter in 1:nrow(where.close)) {
      tmp_row <- where.close[tmp.iter, ]
      if (tmp_row[1] > tmp_row[2]) eliminate <- c(eliminate, tmp_row[2])
    }
  }
  
  # Remove eliminated knots
  if (length(eliminate) > 0) 
    knots <- knots[-eliminate, ]
  
  if(echo){
    plot(data.knots, pch=20, col=res$cluster)
    points(knots, pch='+', col='red')
  }
  return(knots)
}





#' Calculate Radius to Ensure Coverage by Knots
#'
#' This function calculates the radius required to ensure that all station locations 
#' are covered by at least one radial basis function centered at the knot locations.
#'
#' @param knots A matrix or dataframe of coordinates representing knot locations.
#' @param stations A matrix or dataframe of coordinates representing station locations.
#' 
#' @return A numeric scalar representing the radius required to ensure coverage.
#'
#' @details The function computes the Euclidean distance between every station and all knots, 
#' determines the nearest knot for each station, and calculates the radius as twice the maximum 
#' of these minimum distances. This ensures that every station is within the radius of influence 
#' of at least one knot.
#' 
#' @examples
#' # Example: Calculate radius to cover all stations with knots
#' knots <- matrix(c(0, 0, 1, 1), ncol = 2)       # Knot coordinates
#' stations <- matrix(c(0.5, 0.5, 2, 2), ncol = 2) # Station coordinates
#' calc_radius(knots, stations)
#'
#' @export
calc_radius <- function(knots, stations) {
  # Compute the Euclidean distance between each station and all knots
  eucD <- rdist(stations, as.matrix(knots)) 
  
  # Find the minimum distance from each station to the nearest knot
  nearest_knot_dist <- apply(eucD, 1, min) # Find the minimum distance for each station
  
  #  Determine the radius, ensuring all stations are covered by one or more knots
  radius <- 2 * max(nearest_knot_dist)
  
  return(radius)
}


#' Visualize Knots and Coverage of Wendland Basis Functions
#'
#' This function visualizes the spatial coverage of Wendland basis functions for selected knots.
#' It highlights the knots, their coverage regions, and stations influenced by selected knots.
#'
#' @param knots A dataframe or matrix with coordinates of knots (must have columns `x` and `y`).
#' @param stations A dataframe or matrix with coordinates of stations (must have columns `x` and `y`).
#' @param r A numeric scalar specifying the radius of influence for the Wendland basis functions.
#' @param W A matrix of weights from the Wendland basis functions. Rows correspond to stations, and 
#' columns correspond to knots.
#' @param select A numeric vector of indices specifying the knots whose coverage regions will be 
#' visualized. Defaults to `c(1, 12, 10)`.
#'
#' @return A `ggplot` object that visualizes:
#' \itemize{
#'   \item Knots as red '+' markers.
#'   \item Coverage regions as circles around the knots.
#'   \item Stations influenced by selected knots in different colors (e.g., blue, green, yellow).
#'   \item Stations not covered by any knot (with `NA` weights) in black.
#' }
#'
#' @details The function generates circular paths representing the coverage of each knot. It then 
#' visualizes the stations influenced by specific knots with a customizable color scheme.
#'
#' @examples
#' # Example data
#' knots <- data.frame(x = runif(20, 0, 10), y = runif(20, 0, 10))
#' stations <- data.frame(x = runif(100, 0, 10), y = runif(100, 0, 10))
#' r <- 2
#' W <- matrix(runif(2000, 0, 1), nrow = 100, ncol = 20)
#' 
#' # Visualize the coverage
#' visualize_knots(knots, stations, r, W, select = c(1, 5, 10))
#'
#' @export
visualize_knots <- function(knots, stations, r, W, select = c(1, 12, 10)) {
  # Step 1: Generate circular paths for each knot's coverage
  # Create the first circle based on the first knot
  dat <- cbind(circleFun(unlist(knots[1, ]), diameter = r * 2, npoints = 100), group = 1)
  colnames(knots) <- c('x', 'y')
  
  # Add circles for the remaining knots
  for (iter in 2:nrow(knots)) {
    dat <- rbind(dat, cbind(circleFun(unlist(knots[iter, ]), diameter = r * 2, npoints = 100), group = iter))
  }
  
  # Step 2: Load ggplot2 library for visualization
  library(ggplot2)
  
  # Step 3: Create the plot
  fig <- ggplot(knots) +
    # Plot knots as red '+' symbols
    geom_point(aes(x = x, y = y), shape = '+', size = 6, color = 'red') +
    # Plot circular coverage regions
    geom_path(data = dat, aes(x = x, y = y, group = group)) +
    # Plot stations influenced by the first selected knot in blue
    geom_point(
      data = stations[which(W[, select[1]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("blue", 0.3)
    ) +
    # Plot stations influenced by the second selected knot in green
    geom_point(
      data = stations[which(W[, select[2]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("green", 0.3)
    ) +
    # Plot stations influenced by the third selected knot in yellow
    geom_point(
      data = stations[which(W[, select[3]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("yellow", 0.3)
    ) +
    # Plot stations with no coverage (NA weights) in black
    geom_point(
      data = stations[which(apply(W, 1, function(x) any(is.na(x)))), ],
      aes(x = x, y = y),
      colour = "black"
    )
  
  # Step 4: Return the ggplot object
  fig
}


#' Sinc Function
#'
#' Implements the sinc function, defined as sin(x) / x. A special case is handled when x = 0,
#' where the function returns 1 (using the limit of sinc(x) as x approaches 0).
#'
#' @param x A numeric value or vector.
#' @return The result of sin(x) / x, or an error if both numerator and denominator are zero.
#' @examples
#' sinc(0)  # Returns 1 (limit case)
#' sinc(1)  # Returns sin(1)/1
sinc <- function(x) {
  if (sin(x)==0 & x==0)
    stop("Num&Denom of sinc() are both 0")
  else return(sin(x)/x)
}

#' Zolo_A Function
#'
#' Implements a mathematical transformation based on sine functions. This is used to model
#' specific stochastic processes and is parameterized by `u` and `alpha`.
#'
#' @param u A numeric vector or value. The input for the sine functions.
#' @param alpha A numeric value, usually between 0 and 1, controlling the weight in the transformation.
#' @return The result of the transformation applied to `u` and `alpha`.
#' @examples
#' Zolo_A(1, alpha = 0.7)  # Returns the transformed value
#' @export
Zolo_A <- function(u,alpha=0.7){
  y = ((sin(alpha*u)^alpha*sin(u-u*alpha)^(1-alpha))/(sin(u)))^(1/(1-alpha))
  return(y)
}

#' f_H Function
#'
#' Computes a function related to the expPS distribution, parameterized by `alpha`, `theta`, and `x`.
#' The function includes two main cases based on the value of `alpha`. When `alpha == 0.5`, a specific form 
#' of the function is calculated using `gamma` and `log`. Otherwise, numerical methods (integration) are used.
#'
#' @param x A numeric value for which the function is evaluated.
#' @param alpha A numeric value controlling the shape of the function.
#' @param theta A numeric value influencing the transformation.
#' @param n The number of points for integration when `alpha != 0.5`.
#' @param log Logical. If TRUE, the result is returned on the log scale.
#' @return The computed value of the function `f_H`.
#' @examples
#' f_H(1.0, alpha = 0.7, theta = 0.02)  # Returns the computed value
#' f_H_integrand <- function(u, x){
#'   A <- (sin(pi*u/2)/sin(pi*u))^2
#'   return(x^{-2}*A*exp(-A/x))
#' }
#' integrate(f_H_integrand, lower=0, upper=1, x=x)
#' @export
f_H <- function(x,alpha=0.7,theta=0.02, n=1e3, log=TRUE){
  if(alpha==0.5){
    alpha=1/2; beta = 1/4 # alpha = shape; beta = rate
    f_x <- alpha*log(beta) -log(gamma(alpha)) - (alpha+1)*log(x) - beta/x
    y = theta^alpha+f_x-theta*x
  } else{
    u <- seq(1/2,n-1/2,1)/n
    tmp <- Zolo_A(pi*u, alpha)
    const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1
    H <- const1*x^{-const}*tmp*exp(-x^(-const1)*tmp)
    y = theta^alpha+log(mean(H))-theta*x
  }
  if(log) return(y) else return(exp(y))
}



#' Double Rejection Sampler
#'
#' This function implements a rejection sampling method to generate expPS variables parameterized
#' by `alpha` and `theta`. Based on Devroye (2009), it involves multiple stages of sampling
#' from auxiliary distributions and applying acceptance/rejection criteria.
#'
#' @param theta A numeric parameter that influences the distribution.
#' @param alpha A numeric parameter controlling the shape of the distribution.
#' @return A random sample from the distribution.
#' @examples
#' double_rejection_sampler(theta = 0.5, alpha = 0.7)  # Returns a sample from the distribution
#' @export
double_rejection_sampler = function(theta=theta,alpha=alpha){
  if(theta!=0){
    ## set up
    gam = theta^alpha * alpha * (1-alpha)
    xi = (2+sqrt(pi/2))*(sqrt(2*gam))/pi + 1
    psi = exp(-gam*(pi^2)/8)*(2+sqrt(pi/2))*sqrt(gam*pi)/pi
    w_1 = xi*sqrt(pi/(2*gam))
    w_2 = 2*psi*sqrt(pi)
    w_3 = xi*pi
    b = (1-alpha)/alpha
    
    # Initial state
    X = 1
    second_condition = 1
    E = 0.5
    while (!(X >= 0 & second_condition <= E)) {
      ## First generate U with density proportional to g**
      # Initial state
      U = 1; Z = 2
      # Start
      while (!(U<pi & Z<=1)) {
        V = runif(1);W_prime = runif(1)
        if(gam >= 1){
          if(V < (w_1/(w_1+w_2))){
            U <- abs(rnorm(1))/sqrt(gam)
          } else {
            U <- pi*(1-W_prime^2)
          }
        } else {
          if(V < (w_3/(w_3+w_2))){
            U = pi*W_prime
          } else {
            U = pi*(1-W_prime^2)
          }
        }
        W = runif(1)
        zeta = sinc(U)/((sinc(alpha*U)^alpha)*(sinc((1-alpha)*U))^(1-alpha))
        zeta = sqrt(zeta)
        phi = (sqrt(gam)+alpha*zeta)^(1/alpha)
        z = phi/(phi-(sqrt(gam))^(1/alpha))
        
        part = xi*exp((-gam*U^2)/2)*ifelse(U>=0 & gam>=1,1,0) + psi/(sqrt(pi-U))*ifelse(U<pi & U>0,1,0) + xi*ifelse(U>=0 & U<=pi & gam<1,1,0)
        num = pi*exp((-theta^alpha)*(1-zeta^(-2)))*part
        denom = (1+sqrt(pi/2))*sqrt(gam)/zeta + z
        pho = num/denom
        Z = W*pho
      }
      
      # Generate X with density proportional to g(x,U)
      
      # set up of constants
      a = Zolo_A(u = U,alpha = alpha)
      m = (b*theta/a)^alpha
      delta = sqrt(m*alpha/a)
      a_1 = delta*sqrt(pi/2)
      a_2 = delta
      a_3 = z/a
      s = a_1+a_2+a_3
      
      V_prime = runif(1)
      N_prime = rnorm(1)
      E_prime = rexp(1)
      if(V_prime < a_1/s){
        X = m-delta*abs(N_prime)
      } else if (V_prime < a_2/s){
        X = runif(1,m,m+delta)
      } else {
        X = m+delta+E_prime*a_3
      }
      E = -log(Z)
      second_condition = a*(X-m) + theta*(X^(-b)-m^(-b)) - (N_prime^2/2)*ifelse(X<m,1,0) - E_prime*ifelse(X>(m+delta),1,0)
    }
    return(1/X^b)
  } else {
    U = runif(1,0,pi)
    E_prime = rexp(1)
    X = E_prime/Zolo_A(U)
    res = (1/X)^((1-alpha)/alpha)
    return(res)
  }
}

#' Single Rejection Sampler
#'
#' A simpler rejection sampling method for generating samples from distributions controlled by `alpha` and `theta`.
#' It uses inverse gamma and stable distribution sampling depending on the value of `alpha`.
#'
#' @param theta A numeric parameter influencing the sampling process.
#' @param alpha A numeric value, typically between 0 and 1
#' @export
single_rejection_sampler = function(theta=theta, alpha=1/2){
  if(alpha==1/2) {
    X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
    V <- runif(1)
    while(V> exp(-theta*X)){
      X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
      V <- runif(1)
    }
  }else{
    gamma <- cos(pi*alpha/2)^{1/alpha}
    X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
    V <- runif(1)
    while(V> exp(-theta*X)){
      X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
      V <- runif(1)
    }
  }
  
  return(X)
}

#' expPS Density Calculation
#'
#' Use functions in FMStable to calculate H density, but using numerical integration implicitly. 
#' This function leverages the FMStable package to evaluate the stable distribution.
#'
#' @param x Numeric vector of values at which the density is to be calculated.
#' @param alpha Stability parameter of the stable distribution (0 < alpha < 2).
#' @param delta Scale parameter for the stable distribution.
#' @param theta Exponential decay parameter.
#'
#' @return A numeric vector containing the computed density values at each input value of \code{x}.
#'
#' @examples
#' # Example usage of H_density
#' x <- seq(0.1, 10, by = 0.1)
#' alpha <- 0.7
#' delta <- 1
#' theta <- 0.5
#' densities <- H_density(x, alpha = alpha, delta = delta, theta = theta)
#' #'
#' @importFrom FMStable dEstable setParam
#' @export
H_density <- function(x,alpha=alpha,delta=delta,theta=theta){
  gamma <- ((delta/alpha)*cos(pi*alpha/2))^{1/alpha}
  xtmp <- x/((delta/alpha)^{1/alpha})
  res1 <- FMStable::dEstable(xtmp,FMStable::setParam(alpha=alpha, location=0, logscale=log(gamma),pm="S1"))
  res2 <- res1*exp(-theta*x)/(((delta/alpha)^{1/alpha})*exp((-delta*theta^alpha)/alpha))
  return(res2)
}


#' Marginal Distribution Function for Dependence Measure
#'
#' Computes the marginal distribution function to calculate the dependence measure
#' based on Equation 10 in Bopp (2020).
#'
#' @param x Numeric vector of values at which the marginal distribution is evaluated.
#' @param L Numeric parameter related to the dependence structure.
#' @param theta Numeric vector of parameter values for the dependence structure.
#' @param alpha Stability parameter of the stable distribution (0 < alpha < 2).
#' @param k_l Numeric parameter related to the scaling in the dependence measure.
#'
#' @return A numeric vector containing the computed marginal distribution values for each input \code{x}.
#'
#' @examples
#' # Example usage of marginal_thetavec
#' x <- seq(0.1, 10, by = 0.1)
#' L <- 1
#' theta <- 0.5
#' alpha <- 0.7
#' k_l <- 2
#' result <- marginal_thetavec(x, L = L, theta = theta, alpha = alpha, k_l = k_l)
#'
#' @export
marginal_thetavec <- function(x,L=L,theta=theta,alpha=alpha,k_l=k_l){
  y <- sapply(x, function(z) exp(sum(theta^alpha - (theta+(k_l/z)^(1/alpha))^alpha)))
  return(y)
}

#################################################################################
##  --------------------------- v_t initial values   ----------------------------
#################################################################################
full_cond <- function(v_t, W_alpha, X_t, tau, m){
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  if(any(tmp<=0)) return(-1e5)
  return(-mean(log(Y_t)) + 2*mean(log(tmp))- tau*mean(tmp))
}

gradient_full_cond <-  function(v_t, W_alpha, X_t, tau, m){
  res <- rep(NA, ncol(W_alpha))
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  for (iter in 1:ncol(W_alpha)){
    tmp2 <- X_t*W_alpha[,iter]/Y_t^2
    res[iter] <- mean(2*tmp*tmp2 - W_alpha[,iter]/Y_t- tau*tmp^2*tmp2)
  }
  return(res)
}

full_cond_logscale <- function(log_v_t, W_alpha, X_t, tau, m){
  v_t <- exp(log_v_t)
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  if(any(tmp<=0)) return(-1e5)
  return(-mean(log(Y_t)) + 2*mean(log(tmp))- tau*mean(tmp))
}

gradient_full_cond_logscale <-  function(log_v_t, W_alpha, X_t, tau, m){
  v_t <- exp(log_v_t)
  res <- rep(NA, ncol(W_alpha))
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  for (iter in 1:ncol(W_alpha)){
    tmp2 <- X_t*W_alpha[,iter]/Y_t^2
    res[iter] <- mean(2*tmp*tmp2 - W_alpha[,iter]/Y_t- tau*tmp^2*tmp2)
  }
  return(res*v_t)
}

# tmp =as_array(v_t[,1])
# res <- optim(par=tmp, full_cond, W_alpha=W_alpha, X_t=X_t, gr=gradient_full_cond, method='CG',
#       control = list(fnscale=-1, trace=1, maxit=20000))



#' Emulate Spatial Input Using Trained Weights and Biases from VAE
#'
#' This function emulates spatial inputs using trained weights and biases from the XVAE. Note weights and biases
#' have to be in the global environment. 
#' It encodes input data, applies transformations, and decodes latent variables to generate simulations.
#'
#' @details
#' The function implements the following steps:
#' 1. **Encoder for \( v_t \)**: Encodes spatial inputs into a latent representation.
#' 2. **Encoder for \( v_t' \)**: Encodes another input for transformation.
#' 3. **Laplace Transformation**: Activates latent variables using a Laplace transformation.
#' 4. **Re-parameterization Trick**: Generates stochastic latent variables.
#' 5. **Decoder**: Decodes latent variables to produce outputs.
#' 
#' The final outputs are simulated spatial data (`emulations`) and an estimated parameter (`theta_est`).
#'
#' @return A list with:
#' \itemize{
#'   \item `emulations`: A matrix of simulated values for spatial inputs.
#'   \item `theta_est`: A matrix of estimated parameters from the decoder.
#' }
#'
#' @export
emulate_from_trained_XVAE <- function(){
  
  ### -------- Encoder for v_t --------
  h <- w_1$mm(X_tensor)$add(b_1)$relu()
  h_1 <- w_2$mm(h)$add(b_2)$relu()
  sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
  mu <- w_4$mm(h_1)$add(b_4)$relu()
  
  ### -------- Encoder for v_t_prime --------
  h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
  h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()
  
  ### -------- Activation via Laplace transformation --------
  h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
  h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
  h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
  theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
  
  sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
  mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)
  
  
  Epsilon <- t(abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
  Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
  
  Epsilon <- torch_tensor(Epsilon, dtype=torch_float())
  Epsilon_prime <- torch_tensor(Epsilon_prime, dtype=torch_float())
  
  ### -------- Re-parameterization trick --------
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  ### -------- Decoder --------
  l <- w_5$mm(v_t_prime)$add(b_5)$relu()
  l_1 <- w_6$mm(l)$add(b_6)$relu()
  theta_t <- w_7$mm(l_1)$add(b_7)$relu()
  
  y_star <- lrelu(W_alpha_tensor$mm(v_t)$add(b_8))
  
  ##Decoder
  station_Simulations_All <- matrix(rfrechet(n.s*n.t, shape=1, location = m, scale = tau), nrow=n.s) * as_array((y_star))
  theta_sim <- as_array(theta_t)
  return(list(emulations = station_Simulations_All, theta_est = theta_sim))
}




#################################################################################
##  ------------------------ Plot pretty spatial maps ---------------------------
#################################################################################
#' Generate Points of a Circle
#'
#' Creates a data frame containing \code{npoints} evenly spaced points around a circle 
#' given its center and diameter.
#'
#' @param center Numeric vector of length 2, specifying the \code{(x, y)} coordinates of the circle's center. Default is \code{c(0, 0)}.
#' @param diameter Numeric value specifying the diameter of the circle. Default is \code{1}.
#' @param npoints Integer specifying the number of points to generate along the circle. Default is \code{100}.
#'
#' @return A data frame with columns \code{x} and \code{y}, containing the coordinates of the points on the circle.
#'
#' @examples
#' # Generate points for a circle of diameter 2 centered at (1, 1)
#' circle <- circleFun(center = c(1, 1), diameter = 2, npoints = 50)
#' plot(circle, type = "l", asp = 1)
#'
#' @export
circleFun <- function(center = c(0,0),diameter = 1, npoints = 100){
  r = diameter / 2
  tt <- seq(0,2*pi,length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

#' Great Circle Distance Between Two Points
#'
#' Computes the great circle distance (in kilometers) between two points on the Earth's surface 
#' given their longitude and latitude in degrees.
#'
#' @param x Numeric vector of length 2, containing the longitude and latitude of the first point (in degrees).
#' @param y Numeric vector of length 2, containing the longitude and latitude of the second point (in degrees).
#'
#' @return A numeric value representing the great circle distance (in kilometers) between the two points.
#'
#' @examples
#' # Distance between Paris (48.8566, 2.3522) and New York (40.7128, -74.0060)
#' distance <- circ_dist(c(2.3522, 48.8566), c(-74.0060, 40.7128))
#' print(distance)
#'
#' @export
circ_dist <- function(x, y){
  x <- x*pi/180
  y <- y*pi/180
  6371*acos(sin(x[2])*sin(y[2])+cos(x[2])*cos(y[2])*cos(x[1]-y[1]))
}

#' Great Circle Distance Matrix
#'
#' Computes a matrix of great circle distances (in kilometers) between two sets of points 
#' given their longitudes and latitudes in degrees.
#'
#' @param X Numeric matrix with 2 columns, where each row represents a point's longitude and latitude (in degrees).
#' @param Y Numeric matrix with 2 columns, where each row represents a point's longitude and latitude (in degrees).
#'
#' @return A numeric matrix where the element at position \code{[i, j]} contains the great circle distance (in kilometers) between 
#' the \code{i}-th point in \code{X} and the \code{j}-th point in \code{Y}.
#'
#' @examples
#' # Compute distances between two sets of locations
#' X <- matrix(c(2.3522, 48.8566, -74.0060, 40.7128), ncol = 2, byrow = TRUE) # Paris and New York
#' Y <- matrix(c(139.6917, 35.6895, -0.1278, 51.5074), ncol = 2, byrow = TRUE) # Tokyo and London
#' distances <- circ_dist_mat(X, Y)
#' print(distances)
#'
#' @export
circ_dist_mat <- function(X, Y){
  X <- as.matrix(X); Y <- as.matrix(Y)
  Output <- matrix(NA, nrow=nrow(X), ncol=nrow(Y))
  X <- X*pi/180
  Y <- Y*pi/180
  tmp1_X <- sin(X[,2]); tmp2_X<-cos(X[,2])
  tmp1_Y <- sin(Y[,2]); tmp2_Y<-cos(Y[,2])
  for(i in 1:nrow(X)){
    for(j in 1:nrow(Y)){
      Output[i,j] <- acos(tmp1_X[i]*tmp1_Y[j]+tmp2_X[i]*tmp2_Y[j]*cos(X[i,1]-Y[j,1]))
    }
  }
  return(6371*Output)
}


#' Create a Spatial Map over Contiguous United States with Discrete Data Ranges
#'
#' This function generates spatial maps using ggplot2, either with colored points or raster layers, for visualizing spatial data.
#'
#' @param stations A data frame with columns `x` and `y`, representing spatial coordinates of points to plot.
#' @param var Optional numeric vector to map values to colors for points or raster layers.
#' @param pal Color palette for the map (default is `RColorBrewer::brewer.pal(9,"OrRd")`).
#' @param title Title of the plot.
#' @param legend.name Name of the legend for the variable.
#' @param show.legend Logical; whether to display the legend (default is `TRUE`).
#' @param show.color Logical; whether to use color mapping (default is `TRUE` if `var` is provided).
#' @param show.axis.y Logical; whether to show the y-axis labels (default is `TRUE`).
#' @param xlab, ylab Labels for the x- and y-axes (default: `'x'` and `'y'`).
#' @param brks.round Number of decimal places to round breakpoints (default: 2).
#' @param tight.brks Logical; whether to use tighter breakpoints based on quartiles (default: `FALSE`).
#' @param conus_fill Fill color for the map background (default: `"white"`).
#' @param border.wd Width of point borders (default: `0.2`).
#' @param pt.size Size of points in the plot (default: `3`).
#' @param shp Shape of points in the plot (default: `16`).
#' @param range Range of the variable values to map (default is the range of `var`).
#' @param q25, q75 Optional quartiles for tight breakpoints.
#' @param raster Logical; whether to display a raster map instead of points (default: `TRUE`).
#' @param aspect_ratio Aspect ratio of the plot (default: `1`).
#' 
#' @return A ggplot object representing the spatial map.
#' @examples
#' stations <- data.frame(x = runif(100), y = runif(100), var = rnorm(100))
#' spatial_map(stations, var = stations$var, title = "Example Spatial Map")
#' @import ggplot2
#' @importFrom ggh4x force_panelsizes
#' @export
spatial_map <- function(stations, var=NULL, pal=RColorBrewer::brewer.pal(9,"OrRd"), 
                        title='spatial map', legend.name='val', show.legend=TRUE, show.color=TRUE, show.axis.y = TRUE,
                        xlab='x', ylab='y',
                        brks.round = 2, tight.brks = FALSE,
                        conus_fill = "white", 
                        border.wd = 0.2, pt.size = 3, shp = 16,
                        range=NULL, q25=NULL, q75=NULL, raster=TRUE, aspect_ratio =1){
  require(ggplot2);require(ggh4x)
  if(colnames(stations)[1]!='x') colnames(stations)=c('x', 'y')
  if(is.null(var)) show.color=FALSE
  if(!show.color){
    plt0 <- ggplot(stations) +
      geom_point(size = pt.size, shape = shp, aes( x = x, y = y), na.rm = TRUE) +
      ylab('y') + xlab('x') +
      theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
            plot.title = element_text(hjust = 0.5, size=14),
            legend.text=element_text(size=12), legend.title=element_text(size=13),
            axis.text=element_text(size=13), axis.title.y=element_text(size=14), 
            axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
      ggtitle(title) +
      ggh4x::force_panelsizes(rows = unit(3.75, "in"),
                       cols = unit(3.75, "in"))
    return(plt0)
  }
  
  if(is.null(range)) range =  range(var, na.rm=TRUE)
  nugget = 10^{-brks.round}
  brks = round(seq(range[1]-nugget, range[2]+nugget, length.out = length(pal)+1), brks.round)
  if(tight.brks) {
    if(is.null(q25)) q25 = quantile(var, p=0.25)
    if(is.null(q75)) q75 = quantile(var, p=0.75)
    brks = round(c(range[1]-nugget, seq(q25, q75, length.out = length(pal)-1), range[2]+nugget), brks.round)
  }
  color.use <- cut(var, brks, include.lowest = TRUE)
  col.labels <- c(paste("<", brks[2]), levels(color.use)[2:(length(brks)-2)],
                  paste(">", brks[(length(brks)-1)]))
  
  sc <- scale_color_manual( values = pal, name = legend.name, drop = FALSE, na.translate=FALSE,
                            labels = col.labels ) 
  if(show.legend) gd <- guides(color = guide_legend( reverse = TRUE, override.aes = list(size=5) )) else gd <- guides(color = "none")
  
  if(!raster) {
    if(show.axis.y){
      plt1 <- ggplot(stations) +
        geom_point(size = pt.size, shape = shp, aes( x = x, y = y, color = color.use ), na.rm = TRUE) +
        ylab(ylab) + xlab(xlab)  + sc + gd +
        theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
              plot.title = element_text(hjust = 0.5, size=14),
              legend.text=element_text(size=12), legend.title=element_text(size=13),
              axis.text=element_text(size=13), axis.title.y=element_text(size=14), 
              axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
        ggtitle(title) +
        ggh4x::force_panelsizes(rows = unit(3.75, "in"),
                         cols = unit(3.75, "in"))}else{
                           plt1 <- ggplot(stations) +
                             geom_point(size = pt.size, shape = shp, aes( x = x, y = y, color = color.use ), na.rm = TRUE) +
                             ylab(ylab) + xlab(xlab)  + sc + gd +
                             theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
                                   plot.title = element_text(hjust = 0.5, size=14),
                                   legend.text=element_text(size=12), legend.title=element_text(size=13),
                                   axis.text=element_text(size=13),
                                   axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)),
                                   axis.text.y=element_blank(), 
                                   axis.ticks.y=element_blank(),
                                   axis.title.y = element_blank())+
                             ggtitle(title) +
                             ggh4x::force_panelsizes(rows = unit(3.75, "in"),
                                              cols = unit(3.75, "in"))
                         }
  }else{
    plt1 <- ggplot(stations) +
      geom_raster(aes( x = x, y = y, fill = var), na.rm = TRUE) +
      ylab(ylab) + xlab(xlab)  + scale_fill_gradientn(name=legend.name, colors = pal, limits = range) + gd +
      theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
            plot.title = element_text(hjust = 0.5, size=14),
            legend.text=element_text(size=12), legend.title=element_text(size=13),
            axis.text=element_text(size=13), 
            axis.title=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
      ggtitle(title) +
      ggh4x::force_panelsizes(rows = unit(3.75, "in"),
                       cols = unit(3.75/aspect_ratio, "in"))
  }
  return(plt1)
}





#' Chi-Plot for Spatial Dependence
#'
#' This function generates a chi-plot for evaluating spatial dependence in simulated and emulated data at specified distances.
#'
#' @param X A matrix of simulated values at spatial locations.
#' @param stations A data frame with spatial coordinates (`x`, `y`) of the stations.
#' @param emulation A matrix of emulated values at spatial locations.
#' @param distance Distance for selecting pairs of spatial points for comparison.
#' @param tol Tolerance for distance selection (default: `0.001`).
#' @param u_vec A vector of quantile thresholds for chi calculations (default: `c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001))`).
#' @param L Number of levels for marginal transformation (default: `25`).
#' @param ylab Label for the y-axis (default: `expression(chi[u])`).
#' @param uniform Logical; whether to use uniform marginals (default: `FALSE`).
#' @param legend Logical; whether to display a legend (default: `TRUE`).
#' @param show.axis.y Logical; whether to display the y-axis labels (default: `TRUE`).
#' @param duplicated Logical; whether to retain duplicate pairs of points (default: `TRUE`).
#' 
#' @return A ggplot object representing the chi-plot.
#' @examples
#' # Example data
#' X <- matrix(rnorm(1000), ncol=10)
#' emulation <- matrix(rnorm(1000), ncol=10)
#' stations <- data.frame(x = runif(100), y = runif(100))
#' chi_plot(X, stations, emulation, distance=0.5)
#' @import ggplot2
#' @import fields
#' @importFrom ggh4x force_panelsizes
#' @export
chi_plot <- function(X, stations, emulation, distance, tol=0.001,
                     u_vec=c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001)),
                     L=25, ylab=expression(chi[u]), uniform=FALSE, legend = TRUE, show.axis.y = TRUE, duplicated = TRUE){
  require(fields)
  require(ggplot2)
  d <- distance
  upper <- d+tol
  lower <- d-tol
  
  spatial_loc <- matrix(NA,nrow = nrow(X),ncol = 6)
  for (i in 1:(nrow(X)-1)) {
    tmp_dist <- rdist(matrix(stations[i,],nrow=1),stations[(i+1):nrow(X),])
    cand <- which(tmp_dist < upper 
                  & tmp_dist > lower)
    if(length(cand)!=0) spatial_loc[i,1:6] <- cand[1:6]
  }
  
  pairs <- matrix(NA,ncol = 2,nrow = length(spatial_loc))
  for (i in 1:nrow(spatial_loc)) {
    for (j in 1:ncol(spatial_loc)) {
      pairs[(i-1)*ncol(spatial_loc)+j,1] <- i
      pairs[(i-1)*ncol(spatial_loc)+j,2] <- i+spatial_loc[i,j]
    }
  }
  pairs <- data.frame(na.omit(pairs))
  if(!duplicated)  {
    pairs <- pairs[which(!duplicated(pairs[,1]))+1,]
    wh.sub <- floor(seq(1, nrow(pairs), length.out=10))
    pairs <- pairs[wh.sub,] 
    pairs <- data.frame(na.omit(pairs))}
  
  k1 <- ncol(X)
  k2 <- ncol(emulation)
  
  sim_pairs <- matrix(NA,nrow = ncol(X)*nrow(pairs),ncol = 4)
  emu_pairs <- matrix(NA,nrow = ncol(emulation)*nrow(pairs),ncol = 4)
  
  for (i in 1:nrow(pairs)) {
    sim_pairs[((i-1)*k1+1):(i*k1),1] <- X[pairs[i,1],]
    sim_pairs[((i-1)*k1+1):(i*k1),2] <- marginal_thetavec(x=X[pairs[i,1],],L=L,theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,1],])
    if(uniform) {
      sim_pairs[((i-1)*k1+1):(i*k1),2] <- X[pairs[i,1],]
    }
    sim_pairs[((i-1)*k1+1):(i*k1),3] <- X[pairs[i,2],]
    sim_pairs[((i-1)*k1+1):(i*k1),4] <- marginal_thetavec(x=X[pairs[i,2],],L=L,theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,2],])
    if(uniform) {
      sim_pairs[((i-1)*k1+1):(i*k1),4] <- X[pairs[i,2],]
    }
    
    emu_pairs[((i-1)*k2+1):(i*k2),1] <- emulation[pairs[i,1],]
    emu_pairs[((i-1)*k2+1):(i*k2),2] <- marginal_thetavec(x=emulation[pairs[i,1],],L=L,
                                                          theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,1],])
    if(uniform) {
      emu_pairs[((i-1)*k1+1):(i*k1),2] <- emulation[pairs[i,1],]
    }
    
    emu_pairs[((i-1)*k2+1):(i*k2),3] <- emulation[pairs[i,2],]
    emu_pairs[((i-1)*k2+1):(i*k2),4] <- marginal_thetavec(x=emulation[pairs[i,2],],L=L,
                                                          theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,2],])
    if(uniform) {
      emu_pairs[((i-1)*k1+1):(i*k1),4] <- emulation[pairs[i,2],]
    }
  }
  
  sim_pairs <- sim_pairs[,c(2,4)]
  emu_pairs <- emu_pairs[,c(2,4)]
  
  Min_sim <- apply(sim_pairs, 1, min)
  all_sim <- as.vector(sim_pairs)
  Min_emu <- apply(emu_pairs, 1, min)
  all_emu <- as.vector(emu_pairs)
  
  
  EmpIntv_sim <- matrix(NA, nrow = length(u_vec), ncol=3)
  EmpIntv_emu <- matrix(NA, nrow = length(u_vec), ncol=3)
  
  for(i in 1:length(u_vec)){
    p_tmp1_sim <- mean(Min_sim>u_vec[i])
    p_tmp2_sim <- mean(all_sim>u_vec[i])
    if(p_tmp1_sim==0|p_tmp2_sim==0){
      EmpIntv_sim[i,]<-c(-2,2,0)
    } else{
      var_sim <- (1/p_tmp1_sim-1)/length(Min_sim) + (1/p_tmp2_sim-1)/length(all_sim)
      EmpIntv_sim[i,]<-c(exp(log(p_tmp1_sim/p_tmp2_sim)) - qnorm(0.975)*sqrt(var_sim),
                         exp(log(p_tmp1_sim/p_tmp2_sim)) - qnorm(0.025)*sqrt(var_sim), p_tmp1_sim/p_tmp2_sim)
    }
    
    p_tmp1_emu <- mean(Min_emu>u_vec[i])
    p_tmp2_emu <- mean(all_emu>u_vec[i])
    if(p_tmp1_emu==0|p_tmp2_emu==0){
      EmpIntv_emu[i,]<-c(-2,2,0)
    } else{
      var_emu <- (1/p_tmp1_emu-1)/length(Min_emu) + (1/p_tmp2_emu-1)/length(all_emu)
      EmpIntv_emu[i,]<-c(exp(log(p_tmp1_emu/p_tmp2_emu)) - qnorm(0.975)*sqrt(var_emu),
                         exp(log(p_tmp1_emu/p_tmp2_emu)) - qnorm(0.025)*sqrt(var_emu), p_tmp1_emu/p_tmp2_emu)
    }
  }
  dat <- data.frame(x=u_vec,truth=EmpIntv_sim[,3],truth_upper=EmpIntv_sim[,2],truth_lower=EmpIntv_sim[,1],
                    emu=EmpIntv_emu[,3],emu_upper=EmpIntv_emu[,2],emu_lower=EmpIntv_emu[,1])
  dat[dat>1] <- 1
  dat[dat<0] <- 0
  plt <- ggplot(dat,aes(x=x,y=truth)) +
    geom_line(aes(color="Truth"),linewidth=1) +
    geom_line(aes(y=emu,color="Emulation"),linewidth=1) +
    scale_color_manual(values=c('red', 'black')) + 
    geom_ribbon(data=dat,aes(ymin=emu_lower,ymax=emu_upper),alpha=0.2,fill="red") +
    geom_ribbon(data=dat,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.4,fill="black") +
    labs(colour="Type") +
    ylab(ylab) + xlab("Quantile") + 
    ggtitle(paste("Distance in (", lower, ", ", upper, ")", sep=''))+
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0,1)) + 
    ggh4x::force_panelsizes(rows = unit(3.05, "in"),
                     cols = unit(3.05, "in"))
  
  if(!legend) plt <- plt + guides(color="none")
  if(!show.axis.y) plt<- plt + theme( axis.text.y=element_blank(), 
                                      axis.ticks.y=element_blank(),
                                      axis.title.y = element_blank())
  
  return(plt)
}



#' Compare Average Radius of Exceedance (ARE) using different models
#'
#' This function calculates the Average Radius of Exceedance (ARE) for different 
#' models based on a regular grid of spatial data in their uniform scale (i.e., copulas). 
#'
#' @param stations A matrix or data frame containing the coordinates of points 
#'   in the regular spatial grid. Must be a regular grid.
#' @param center An optional index specifying the center of the spatial domain. 
#'   If NULL, the center is determined as the point with the minimum mean distance 
#'   to all other points.
#' @param U1 A matrix of simulated data in their uniform scalerepresenting the "Truth" model.  
#'   Rows correspond to spatial locations, and columns correspond to simulations.
#' @param U2 An optional matrix of data representing the emulations from a model.
#' @param U3 An optional matrix of data representing the emulations from another model.
#' @param u_vec A numeric vector of quantile thresholds used for calculating exceedances. 
#'   Defaults to a sequence of values between 0 and 0.998.
#' @param names A character vector of length three specifying labels for the models. 
#'   Defaults to c("Truth", "XVAE", "extGAN").
#'
#' @return A ggplot object showing the ARE comparison for the three models. The 
#'   plot includes lines for mean ARE and shaded areas for 95% confidence intervals.
#'
#' @details The function calculates the ARE for the specified models at each 
#'   quantile threshold in `u_vec`. The ARE is computed as the average spatial 
#'   area where the model exceeds the threshold at the center of the domain.
#'
#' @examples
#' # Example usage:
#' data(copulas)
#' ARE_comparison(stations_grid, U1=U_sim_grid, U2=U_xvae_grid, U3=U_gan_grid, names =c("Truth", "XVAE", "extGAN"))
#' @import ggplot2
#' @import fields
#' @importFrom ggh4x force_panelsizes
#' @import tidyterra
#' @export
ARE_comparison <- function(stations, center=NULL, U1, U2=NULL, U3=NULL, u_vec=NULL, names =c("Truth", "XVAE", "extGAN")){
  tidyterra::is_regular_grid(stations, digits = 6) #stop("The spatial input must be on a regular grid.")
  if(is.null(center)){
    center <- which.min(apply(rdist(stations),1,mean))
  }
  unit_area <- diff(unique(stations[,1]))[1] * diff(unique(stations[,2]))[1]
  total_area <- unit_area * nrow(stations)
  if(is.null(u_vec)) u_vec <- c(seq(0,0.98,0.01),seq(0.9801,0.998,0.0001))
  
  Emp_ARE_ori <- array(NA, c(length(u_vec), 3))
  for(iter in 1:length(u_vec)){
    u_tmp <- u_vec[iter]
    where.exceed <- which(U1[center, ]>u_tmp) ## center of the spatial domain
    if(length(where.exceed)>0){
      tmp_AE <- rep(NA, length(where.exceed))
      for(col in 1:length(where.exceed)){
        tmp_AE[col] <- sum(U1[,where.exceed[col]]>u_tmp)*unit_area
      }
      
      Emp_ARE_ori[iter,] <- c(mean(tmp_AE), quantile(tmp_AE, p=c(0.025, 0.975)))
    }else{
      Emp_ARE_ori[iter,] <- c(0,0,total_area)
    }
  }
  
  
  Emp_ARE_XVAE <- array(NA, c(length(u_vec), 3))
  if(!is.null(U2)){
    for(iter in 1:length(u_vec)){
      u_tmp <- u_vec[iter]
      where.exceed <- which(U2[center, ]>u_tmp) ## center of the spatial domain
      if(length(where.exceed)>0){
        tmp_AE <- rep(NA, length(where.exceed))
        for(col in 1:length(where.exceed)){
          tmp_AE[col] <- sum(U2[,where.exceed[col]]>u_tmp)*unit_area
        }
        
        Emp_ARE_XVAE[iter,] <- c(mean(tmp_AE), quantile(tmp_AE, p=c(0.025, 0.975)))
      }else{
        Emp_ARE_XVAE[iter,] <- c(0,0,total_area)
      }
    }
  }
  
  
  if(!is.null(U2)){
    Emp_ARE_extGAN <- array(NA, c(length(u_vec), 3))
    for(iter in 1:length(u_vec)){
      u_tmp <- u_vec[iter]
      where.exceed <- which(U3[center, ]>u_tmp) ## center of the spatial domain
      if(length(where.exceed)>0){
        tmp_AE <- rep(NA, length(where.exceed))
        for(col in 1:length(where.exceed)){
          tmp_AE[col] <- sum(U3[,where.exceed[col]]>u_tmp)*unit_area
        }
        
        Emp_ARE_extGAN[iter,] <- c(mean(tmp_AE), quantile(tmp_AE, p=c(0.025, 0.975)))
      }else{
        Emp_ARE_extGAN[iter,] <- c(0,0,total_area)
      }
    }
  }
  
  # dat <- data.frame(x = u_vec, 
  #                   truth = sqrt(Emp_ARE_ori[,1]/pi), truth_lower = sqrt(Emp_ARE_ori[,2]/pi), truth_upper=sqrt(Emp_ARE_ori[,3]/pi), 
  #                   emu = sqrt(Emp_ARE_XVAE[,1]/pi), emu_lower = sqrt(Emp_ARE_XVAE[,2]/pi), emu_upper=sqrt(Emp_ARE_XVAE[,3]/pi),
  #                   gan = sqrt(Emp_ARE_extGAN[,1]/pi), gan_lower = sqrt(Emp_ARE_extGAN[,2]/pi), gan_upper=sqrt(Emp_ARE_extGAN[,3]/pi))
  dat <- data.frame(x = u_vec, truth = sqrt(Emp_ARE_ori[,1]/pi), truth_lower = sqrt(Emp_ARE_ori[,1]/pi)+(sqrt(Emp_ARE_ori[,2]/pi)-sqrt(Emp_ARE_ori[,1]/pi))/2, 
                    truth_upper=sqrt(Emp_ARE_ori[,1]/pi)+(sqrt(Emp_ARE_ori[,3]/pi)-sqrt(Emp_ARE_ori[,1]/pi))/2, emu = sqrt(Emp_ARE_XVAE[,1]/pi), 
                    emu_lower = sqrt(Emp_ARE_XVAE[,1]/pi)+(sqrt(Emp_ARE_XVAE[,2]/pi)-sqrt(Emp_ARE_XVAE[,1]/pi))/2, 
                    emu_upper=sqrt(Emp_ARE_XVAE[,1]/pi)+(sqrt(Emp_ARE_XVAE[,3]/pi)-sqrt(Emp_ARE_XVAE[,1]/pi))/2,
                    gan = sqrt(Emp_ARE_extGAN[,1]/pi), 
                    gan_lower = sqrt(Emp_ARE_extGAN[,1]/pi)+(sqrt(Emp_ARE_extGAN[,2]/pi)-sqrt(Emp_ARE_extGAN[,1]/pi))/2, 
                    gan_upper=sqrt(Emp_ARE_extGAN[,1]/pi)+(sqrt(Emp_ARE_extGAN[,3]/pi)-sqrt(Emp_ARE_extGAN[,1]/pi))/2)
  
  plt <- ggplot(dat,aes(x=x,y=gan)) +
    geom_line(aes(color=names[3]),linewidth=1) +
    geom_line(aes(y=truth, color=names[1]),linewidth=1) +
    geom_line(aes(y=emu, color=names[2]),linewidth=1) +
    scale_color_manual(values=c('#0674cf', 'red', 'black')) + 
    geom_ribbon(data=dat,aes(ymin=gan_lower,ymax=gan_upper),alpha=0.2,fill="#4190d1") +
    geom_ribbon(data=dat,aes(ymin=emu_lower,ymax=emu_upper),alpha=0.2,fill="red") +
    geom_ribbon(data=dat,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.2,fill="black") +
    labs(colour="Type") +
    ylab("Average radius of exceedance (ARE)") + xlab("Quantile") + 
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0), limits=c(0, 13)) + 
    ggh4x::force_panelsizes(rows = unit(3.05, "in"),
                            cols = unit(3.05, "in")) + theme(plot.title = element_text(face = 'bold'))
  
  legend <- TRUE; show.axis.y <- TRUE
  if(!legend) plt <- plt + guides(color="none")
  if(!show.axis.y) plt<- plt + theme( axis.text.y=element_blank(), 
                                      axis.ticks.y=element_blank(),
                                      axis.title.y = element_blank())
  plt
}


