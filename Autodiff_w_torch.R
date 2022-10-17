library(torch)

#### Start ####
# setwd("C:/Users/liaoy/OneDrive - University of Missouri/VAE Project")
source("utils.R")

#### Simulation ####
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))
knot <- expand.grid(c(1,3,5,7,9),c(1,3,5,7,9))

k = nrow(knot)
n.s <- nrow(stations)
n.t <- 100 # n.t <- 500

eucD <- rdist(stations,as.matrix(knot))

W <- wendland(eucD,r=3)
W <- sweep(W, 1, rowSums(W), FUN="/")
W_tensor <- torch_tensor(W,requires_grad = TRUE)

set.seed(123)
theta_sim <- runif(k,0,2)
theta_sim[c(8,12,13,14,18)] = 0

alpha = 0.5
V <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)

for (iter in 1:n.t) {
  for (i in 1:k) {
    V[i,iter] <- double_rejection_sampler(theta = theta_sim[i],alpha = alpha)
  }
  X[,iter] <- rfrechet(n.s,shape=(1/alpha)) * (rowSums(V[,iter]*(W^(1/alpha))))^alpha
}

X_tensor <- torch_tensor(t(X),requires_grad=TRUE)

##### Phi #####
# phi start values
w_1 <- torch_tensor(torch_normal(0,0.001, size = c(n.s,k)),requires_grad = TRUE)
w_2 <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_3 <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_4 <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
b_1 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_2 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_3 <- torch_tensor(torch_normal(0,0.001, size = c(1,k)),requires_grad = TRUE)
b_4 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
nn_init_uniform_(b_4,10,20)

w_1_prime <- torch_tensor(torch_normal(0,0.001, size = c(n.s,k)),requires_grad = TRUE)
w_2_prime <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_3_prime <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_4_prime <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
b_1_prime <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_2_prime <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_3_prime <- torch_tensor(torch_normal(0,0.001, size = c(1,k)),requires_grad = TRUE)
b_4_prime <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
nn_init_uniform_(b_4_prime,10,20)

w_5 <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_6 <- torch_tensor(torch_normal(0,0.001, size = c(k,k)),requires_grad = TRUE)
w_7 <- torch_tensor(torch_normal(0,0.001, size = c(1,n.t)),requires_grad = TRUE)
b_5 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_6 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)
b_7 <- torch_tensor(torch_normal(0,1, size = c(1,k)),requires_grad = TRUE)

Epsilon <- (mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
Epsilon_prime <- (mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))

Epsilon <- torch_tensor(Epsilon,requires_grad=TRUE)
Epsilon_prime <- torch_tensor(Epsilon_prime,requires_grad=TRUE)

###### Loss and Optimization ######

### network parameters ---------------------------------------------------------

learning_rate <- 1e-4
niter = 200

for (t in 1:niter) {
  ### -------- Forward pass --------
  
  ## Encoder for v_t
  h_1 <- X_tensor$mm(w_1)$add(b_1)$relu()$mm(w_2)$add(b_2)$relu()
  sigma_sq_vec <- h_1$mm(w_3)$add(b_3)$exp()
  mu <- h_1$mm(w_4)$add(b_4)
  ## Encoder for v_t_prime
  h_1_prime <- X_tensor$mm(w_1_prime)$add(b_1_prime)$relu()$mm(w_2_prime)$add(b_2_prime)$relu()
  sigma_sq_vec_prime <- h_1_prime$mm(w_3_prime)$add(b_3_prime)$exp()
  mu_prime <- h_1_prime$mm(w_4_prime)$add(b_4_prime)
  ## re-parameterization trick
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  # Decoder
  l_1 <- v_t_prime$mm(w_5)$add(b_5)$relu()$mm(w_6)$add(b_6)$relu()
  theta_t <- w_7$mm(l_1)$add(b_7)$relu()
  
  y <- (W_tensor^(1/alpha))$mm(v_t$transpose(1,2))
  
  log_v <- torch_zeros(c(n.t,k))
  for (i in 1:n.t) {
    for (j in 1:k) {
      n <- 1e3
      tmp = (alpha/(1-alpha)) * v_t[i,j]^{-1/(1-alpha)} * Zolo_A(pi*seq(1,n-1,1)/n) * exp(-v_t[i,j]^(-alpha/(1-alpha)) * Zolo_A(pi*seq(1,n-1,1)/n))
      log_v[i,j] = (1/(exp(theta_t[1,j]^alpha))) * tmp$mean() * exp(-theta_t[1,j]*v_t[i,j])
    }
  }
  
  part1 = (n.s*log(1/alpha)*n.t + (X_tensor^(-1/alpha-1)*(y$transpose(1,2)))$log()$sum() - (X_tensor^(-1/alpha-1)*(y$transpose(1,2)))$sum())
  part2 = -log_v$sum()                                                                        # p(v_t|theta_t), p(theta_t)
  part3 = -(v_t-mu)$pow(2)$div(sigma_sq_vec)$sum()/2 - (v_t_prime-mu_prime)$pow(2)$div(sigma_sq_vec_prime)$sum()/2
  part4 = -Epsilon$pow(2)$sum()/2 - Epsilon_prime$pow(2)$sum()/2 - sigma_sq_vec$log()$sum()/2 - sigma_sq_vec_prime$log()$sum()/2
  res <- part1 + part2 - part3 - part4
  
  ### -------- compute loss -------- 
  loss <- res/(n.s*n.t)
  if (t %% 1 == 0)
    cat("Epoch: ", t, "   ELBO: ", loss$item(), "\n")
  if (as.numeric(torch_isnan(loss))==1) break
  ### -------- Backpropagation --------
  
  # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T 
  # want to record for automatic gradient computation
  with_no_grad({
    w_1 <- w_1$sub_(learning_rate * w_1$grad)
    w_2 <- w_2$sub_(learning_rate * w_2$grad)
    w_3 <- w_3$sub_(learning_rate * w_3$grad)
    w_4 <- w_4$sub_(learning_rate * w_4$grad)
    b_1 <- b_1$sub_(learning_rate * b_1$grad)
    b_2 <- b_2$sub_(learning_rate * b_2$grad)
    b_3 <- b_3$sub_(learning_rate * b_3$grad)
    b_4 <- b_4$sub_(learning_rate * b_4$grad)
    w_1_prime <- w_1_prime$sub_(learning_rate * w_1_prime$grad)
    w_2_prime <- w_2_prime$sub_(learning_rate * w_2_prime$grad)
    w_3_prime <- w_3_prime$sub_(learning_rate * w_3_prime$grad)
    w_4_prime <- w_4_prime$sub_(learning_rate * w_4_prime$grad)
    b_1_prime <- b_1_prime$sub_(learning_rate * b_1_prime$grad)
    b_2_prime <- b_2_prime$sub_(learning_rate * b_2_prime$grad)
    b_3_prime <- b_3_prime$sub_(learning_rate * b_3_prime$grad)
    b_4_prime <- b_4_prime$sub_(learning_rate * b_4_prime$grad)
    
    w_5 <- w_5$sub_(learning_rate * w_5$grad)
    w_6 <- w_6$sub_(learning_rate * w_6$grad)
    w_7 <- w_7$sub_(learning_rate * w_7$grad)
    b_5 <- b_5$sub_(learning_rate * b_5$grad)
    b_6 <- b_6$sub_(learning_rate * b_6$grad)
    b_7 <- b_7$sub_(learning_rate * b_7$grad)
    
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
  })
  
}
