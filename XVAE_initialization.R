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
#'
#' @export

XVAE_initialization <- function(){
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
  n <<- 1e3
  vec <<- Zolo_A(pi*seq(1/2,n-1/2,1)/n, alpha)
  Zolo_vec <<- torch_tensor(matrix(vec, nrow=1,ncol=n), dtype=torch_float(), requires_grad = FALSE) 
  Zolo_vec_double <<- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
  const <<- 1/(1-alpha); const1 <<- 1/(1-alpha)-1; const3 <<- log(const1)
  W_alpha_tensor <<- W_tensor$pow(1/alpha)
}
