library(torch)

setwd("~/Desktop/Turbulence/")
source("~/Desktop/GEV-GP_VAE/extCVAE/utils.R")
Theta <- c(rep(0, 250), #seq(0,0.001, length.out=100), 
           seq(0.0011,1.5, length.out=550)) #0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.9, 1.1, 1.5)
k <- 400
n.t <- length(Theta)


Z <- matrix(NA, nrow=k, ncol=length(Theta))

for (iter in 1:length(Theta)) {
  for (i in 1:k) {
    Z[i,iter] <- single_rejection_sampler(theta = Theta[iter])
  }
}


h_1_prime_laplace <- colMeans(exp(Z*(-0.2)))
h_1_prime_t <- -log(h_1_prime_laplace)
h_1_prime_to_theta <- (0.2-h_1_prime_t^2)^2/(4*h_1_prime_t^2)
plot(Theta, h_1_prime_to_theta)
abline(a=0, b=1, col='red', lty=2)
sum((Theta- h_1_prime_to_theta)^2)


w_1 <- 0.1*diag(k)
w_1 <- torch_tensor(w_1,dtype=torch_float(),requires_grad = TRUE)
b_1 <- 0.025*matrix(1:k, ncol=1) 
b_1 <- torch_tensor(b_1,dtype=torch_float(),requires_grad = TRUE)

w_2 <- cbind(diag(floor(2*k/3)), matrix(0, nrow=floor(2*k/3), ncol=k-floor(2*k/3)))
w_2 <- torch_tensor(w_2,dtype=torch_float(),requires_grad = TRUE)
b_2 <- matrix(rep(0.00001,floor(2*k/3)), ncol=1)
b_2 <- torch_tensor(b_2,dtype=torch_float(),requires_grad = TRUE)

w_3 <- cbind(diag(floor(k/3)), matrix(0, nrow=floor(k/3), ncol=floor(2*k/3)-floor(k/3)))
w_3 <- torch_tensor(w_3,dtype=torch_float(),requires_grad = TRUE)
b_3 <- matrix(rep(0.00001,floor(k/3)), ncol=1)
b_3 <- torch_tensor(b_3,dtype=torch_float(),requires_grad = TRUE)


w_4 <- cbind(-0.01, matrix(-0.01, nrow=1, ncol=floor(k/3)-1))
w_4 <- torch_tensor(w_4,dtype=torch_float(),requires_grad = TRUE)
b_4 <- matrix(rep(2.3,1), ncol=1) 
b_4 <- torch_tensor(b_4,dtype=torch_float(),requires_grad = TRUE)


w_1_velocity <- torch_zeros(w_1$size())
w_2_velocity <- torch_zeros(w_2$size())
w_3_velocity <- torch_zeros(w_3$size())
w_4_velocity <- torch_zeros(w_4$size())
b_1_velocity <- torch_zeros(b_1$size())
b_2_velocity <- torch_zeros(b_2$size())
b_3_velocity <- torch_zeros(b_3$size())
b_4_velocity <- torch_zeros(b_4$size())

Z_tensor <- torch_tensor(log(Z),dtype=torch_float())
Theta_log <- log(Theta+2)
lrelu <- nn_leaky_relu(-0.1)

niter <- 100000
learning_rate <- 1e-5
alpha_v <- 0.8
for (t in 1:niter) {
  if(t==30000) { learning_rate <- 1e-4; alpha_v <- 0.8}
  M <- sample(n.t, 150)
  h_1 <- w_1$mm(Z_tensor[, M])$add(b_1)$relu()
  h_2 <- w_2$mm(h_1)$add(b_2)$relu()
  h_3 <- w_3$mm(h_2)$add(b_3)$relu()
  h_4 <- lrelu(w_4$mm(h_3)$add(b_4))
  
  
  
  ### -------- compute loss -------- 
  loss <- h_4$add(2)$log()$subtract(Theta_log[M])$pow(2)$sum()
  if (t %% 100 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n") # we want to maximize
  if (as.numeric(torch_isnan(loss))==1) break
  ### -------- Backpropagation --------
  # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
  loss$backward()
  # w_1$grad$argmax(dim=2)
  
  ### -------- Update weights -------- 
  
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

    
    # Zero gradients after every pass, as they'd accumulate otherwise
    w_1$grad$zero_()
    w_2$grad$zero_()
    w_3$grad$zero_()
    w_4$grad$zero_()
    b_1$grad$zero_()
    b_2$grad$zero_()
    b_3$grad$zero_()
    b_4$grad$zero_()
  })
  
}




h_1 <- w_1$mm(Z_tensor)$add(b_1)$relu()
h_2 <- w_2$mm(h_1)$add(b_2)$relu()
h_3 <- w_3$mm(h_2)$add(b_3)$relu()
h_4 <- lrelu(w_4$mm(h_3)$add(b_4))
plot(Theta, as_array(h_4))
abline(a=0, b=1, col='red', lty=2)


Theta_est <- rep(NA, 3000)
true_theta <- 0.1
for(iter in 1:length(Theta_est)){
  Z_test <- rep(NA, k)
  for (i in 1:k) {
    Z_test[i] <- single_rejection_sampler(theta = true_theta)
  }
  Z_test_tensor <- torch_tensor(matrix(log(Z_test),ncol=1),dtype=torch_float())
  
  h_1 <- w_1$mm(Z_test_tensor)$add(b_1)$relu()
  h_2 <- w_2$mm(h_1)$add(b_2)$relu()
  h_3 <- w_3$mm(h_2)$add(b_3)$relu()
  Theta_est[iter] <- as_array(lrelu(w_4$mm(h_3)$add(b_4)))
}

plot(density(Theta_est, from=0),  main=bquote(theta == .(true_theta)))
abline(v=true_theta, col='red', lty=2)


