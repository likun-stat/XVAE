## ------------------ Results from the initialization code ------------------
## "./initial_values.RData" generated from the R file "./XVAE/Initialization.R"
load("./initial_values.RData")


## -------------- Define initial weights in torch tensor format ---------------
X_tensor <- torch_tensor(X,dtype=torch_float())
W_tensor <- torch_tensor(W,dtype=torch_float())
tmp <- qr.solve(a=t(X), b=t(Z_approx))
w_1 <- t(tmp)
w_1 <- torch_tensor(w_1,dtype=torch_float(),requires_grad = TRUE)
b_1 <- matrix(rep(0,k), ncol=1) 
b_1 <- torch_tensor(b_1,dtype=torch_float(),requires_grad = TRUE)

w_2 <- diag(k)
w_2 <- torch_tensor(w_2,dtype=torch_float(),requires_grad = TRUE)
b_2 <- matrix(rep(0.00001,k), ncol=1)
b_2 <- torch_tensor(b_2,dtype=torch_float(),requires_grad = TRUE)

w_4 <- diag(k)
w_4 <- torch_tensor(w_4,dtype=torch_float(),requires_grad = TRUE)
b_4 <- matrix(rep(0,k), ncol=1) 
b_4 <- torch_tensor(b_4,dtype=torch_float(),requires_grad = TRUE)

w_3 <- 0*diag(k)
w_3 <- torch_tensor(w_3,dtype=torch_float(),requires_grad = TRUE)
b_3 <- matrix(rep(-15,k), ncol=1)
b_3 <- torch_tensor(b_3,dtype=torch_float(),requires_grad = TRUE)


h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()

## -------------------- Re-parameterization trick --------------------
Epsilon <- matrix(abs(rnorm(k*n.t))+0.1, nrow=k)
Epsilon <- torch_tensor(Epsilon,dtype=torch_float())
v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
b_8 <- as_array((W_tensor^(1/alpha)))%*%as_array(v_t)-X
b_8 <- array(-relu(b_8), dim=dim(b_8))
b_8 <- torch_tensor(b_8,dtype=torch_float(), requires_grad = TRUE)


y_approx <-  (W_tensor^(1/alpha))$mm(v_t) + b_8


## -------------------- Add in the Frechet noise --------------------
Epsilon_frechet_indep <- matrix(rfrechet(n.s*n.t, location = m, shape=1, scale = tau), nrow=n.s)
X_approx <- matrix(NA, nrow=n.s, ncol=n.t)
for (iter in 1:n.t){
  X_approx[,iter] <- as_array(Epsilon_frechet_indep[,iter]* y_approx[,iter])
}


## -------------------- Visualize the X_approx --------------------
# ind <- 3
# tmp_range <- range(c(log(as_array(y_approx[,ind]))), log(X[,ind]))
# ggplot(stations) + geom_raster(aes(x=lon, y=lat, fill=log(X[,ind]))) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA, limits=tmp_range) +
#   xlim(29, 48) + ylim(9, 34)
# 
# ggplot(stations) + geom_raster(aes(x=lon, y=lat, fill=log(as_array(y_approx[,ind])))) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = paste0('Smooth replicate #', ind), na.value = NA, limits=tmp_range) +
#   xlim(29, 48) + ylim(9, 34)
# 
# ggplot(stations) + geom_raster(aes(x=lon, y=lat, fill=log(X_approx[,ind]))) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = paste0('Approx replicate #', ind), na.value = NA, limits=tmp_range) +
#   xlim(29, 48) + ylim(9, 34)


## -------------------- w_prime and b_prime for theta --------------------
w_1_prime <- as_array(w_1) #matrix(rnorm(k*n.s,0,0.001), nrow=k)
w_1_prime <- torch_tensor(w_1_prime,dtype=torch_float(),requires_grad = TRUE)
w_2_prime <- matrix(diag(k), nrow=k)
w_2_prime <- torch_tensor(w_2_prime,dtype=torch_float(),requires_grad = TRUE)
w_3_prime <- matrix(rep(0,k*k), nrow=k)
w_3_prime <- torch_tensor(w_3_prime,dtype=torch_float(),requires_grad = TRUE)
w_4_prime <- matrix(diag(k), nrow=k)
w_4_prime <- torch_tensor(w_4_prime,dtype=torch_float(),requires_grad = TRUE)
b_1_prime <- matrix(rep(0,k), ncol=1) #matrix(rnorm(k), ncol=1)
b_1_prime <- torch_tensor(b_1_prime,dtype=torch_float(),requires_grad = TRUE)
b_2_prime <- matrix(rep(0.05,k), ncol=1)
b_2_prime <- torch_tensor(b_2_prime,dtype=torch_float(),requires_grad = TRUE)
b_3_prime <- matrix(rep(-10,k), ncol=1)
b_3_prime <- torch_tensor(b_3_prime,dtype=torch_float(),requires_grad = TRUE)
b_4_prime <- matrix(rep(0,k), ncol=1)
b_4_prime <- torch_tensor(b_4_prime,dtype=torch_float(),requires_grad = TRUE)

w_5 <- diag(k) #matrix(rnorm(k*k,0,0.001), nrow=k)
w_5 <- torch_tensor(w_5,dtype=torch_float(),requires_grad = TRUE)
w_6 <- diag(k) #matrix(rnorm(k*k,0,0.001), nrow=k)
w_6 <- torch_tensor(w_6,dtype=torch_float(),requires_grad = TRUE)
w_7 <- diag(k) #matrix(rnorm(k*k,0,0.0001), nrow=k)
w_7 <- torch_tensor(w_7,dtype=torch_float(),requires_grad = TRUE)
b_5 <- matrix(rep(1e-6,k), ncol=1) #matrix(rnorm(k), ncol=1)
b_5 <- torch_tensor(b_5,dtype=torch_float(),requires_grad = TRUE)
b_6 <- matrix(rep(1e-6,k), ncol=1) #matrix(rnorm(k), ncol=1)
b_6 <- torch_tensor(b_6,dtype=torch_float(),requires_grad = TRUE)
b_7 <- matrix(rep(1e-6,k), ncol=1) #matrix(rnorm(k,0,0.0001), ncol=1)
b_7 <- torch_tensor(b_7,dtype=torch_float(),requires_grad = TRUE)

w_1_velocity <- torch_zeros(w_1$size())
w_2_velocity <- torch_zeros(w_2$size())
w_3_velocity <- torch_zeros(w_3$size())
w_4_velocity <- torch_zeros(w_4$size())
b_1_velocity <- torch_zeros(b_1$size())
b_2_velocity <- torch_zeros(b_2$size())
b_3_velocity <- torch_zeros(b_3$size())
b_4_velocity <- torch_zeros(b_4$size())
w_1_prime_velocity <- torch_zeros(w_1_prime$size())
w_2_prime_velocity <- torch_zeros(w_2_prime$size())
w_3_prime_velocity <- torch_zeros(w_3_prime$size())
w_4_prime_velocity <- torch_zeros(w_4_prime$size())
b_1_prime_velocity <- torch_zeros(b_1_prime$size())
b_2_prime_velocity <- torch_zeros(b_2_prime$size())
b_3_prime_velocity <- torch_zeros(b_3_prime$size())
b_4_prime_velocity <- torch_zeros(b_4_prime$size())

w_5_velocity <- torch_zeros(w_5$size())
w_6_velocity <- torch_zeros(w_6$size())
w_7_velocity <- torch_zeros(w_7$size())
b_5_velocity <- torch_zeros(b_5$size())
b_6_velocity <- torch_zeros(b_6$size())
b_7_velocity <- torch_zeros(b_7$size())

b_8_velocity <- torch_zeros(b_8$size())

Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
Epsilon_prime <- torch_tensor(Epsilon_prime,dtype=torch_float())

###### ---------------------------------------------------------------------- ######
###### --------------------   Loss and Optimization ------------------------- ######
###### ---------------------------------------------------------------------- ######

###### ----------------------  network parameters --------------------------- ######
learning_rate <- -1e-17
alpha_v <- 0.95
lrelu <- nn_leaky_relu(-0.01)

niter = 20000
n <- 1e3
# only depends on alpha; we are not updating alpha for now.
vec <- Zolo_A(pi*seq(1/2,n-1/2,1)/n, alpha)
Zolo_vec <- torch_tensor(matrix(vec, nrow=1,ncol=n), dtype=torch_float(), requires_grad = FALSE) 
Zolo_vec_double <- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1; const3 <- log(const1)
W_alpha_tensor <- W_tensor$pow(1/alpha)
old_loss <- -Inf

# start_time <- Sys.time()
for (t in 1:niter) {
  if(t==1000) { learning_rate <- -5e-17; alpha_v <- 0.9}
  if(t==5000) { learning_rate <- -1e-16; alpha_v <- 0.7}
  if(t==10000) { learning_rate <- -5e-16; alpha_v <- 0.5}
  if(t==20000) { learning_rate <- -1e-15; alpha_v <- 0.5}
  if(t==80000) { learning_rate <- -1e-14; alpha_v <- 0.4}
  if(t==100000) { learning_rate <- -5e-14; alpha_v <- 0.4}
  if(t==120000) { learning_rate <- -1e-13; alpha_v <- 0.4}
  if(t==140000) { learning_rate <- -1e-12; alpha_v <- 0.4}
  if(t==300000) { learning_rate <- -1e-11; alpha_v <- 0.4}

  ### -------- Forward pass --------
  Epsilon <- t(abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
  Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))

  Epsilon <- torch_tensor(Epsilon,dtype=torch_float())
  Epsilon_prime <- torch_tensor(Epsilon_prime,dtype=torch_float())

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
  
  ### -------- Re-parameterization trick --------
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  
  ### -------- Decoder --------
  l <- w_5$mm(v_t_prime)$add(b_5)$relu()
  l_1 <- w_6$mm(l)$add(b_6)$relu()
  theta_t <- w_7$mm(l_1)$add(b_7)$relu()
  
  y_star <- lrelu(W_alpha_tensor$mm(v_t)$add(b_8))
  
  
  ### -------- ELBO -------- 
  ### Part 1
  standardized <- X_tensor$divide(y_star)$sub(m)
  leak <- as_array(sum(standardized<0))
  if(leak>0 & leak<45) standardized$abs_()
  part1 <- -2 * standardized$log()$sum() - y_star$log()$sum() - tau*standardized$pow(-1)$sum() # + n.s*n.t*log(tau)
  
  ### Part 2
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
  
  ### Part 3
  part3 = Epsilon$pow(2)$sum()/2 + Epsilon_prime$pow(2)$sum()/2 + sigma_sq_vec$log()$sum() + sigma_sq_vec_prime$log()$sum()
  res <- part1 + part2 + part3
  
  ### -------- compute loss -------- 
  loss <- (res/(n.s*n.t))
  if(!is.finite(loss$item())) break
  if (t %% 1 == 0)
    cat("Epoch: ", t, "   ELBO: ", loss$item(), "\n") # we want to maximize
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
# end_time <- Sys.time()
# end_time - start_time

# loss - part3/(n.s*n.t)
# plot(theta_sim[,1], as_array(theta_t[,3]), xlab=expression(paste('true ',theta)), ylab=expression(paste('CVAE ',theta)))
# abline(a = 0, b = 1, col='orange', lty=2)

save.image("./XVAE_Red_Sea_SST_results.RData")