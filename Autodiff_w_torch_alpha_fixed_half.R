library(torch)

setwd('~/Desktop/GEV-GP_VAE/extCVAE/')
source("utils.R")

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Simulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))
knot <- expand.grid(x=c(1,3,5,7,9),y=c(1,3,5,7,9))
plot(stations)
points(knot, pch="+", col='red', cex=2)

k = nrow(knot)
n.s <- nrow(stations)
n.t <- 100 # n.t <- 500

eucD <- rdist(stations,as.matrix(knot))

W <- wendland(eucD,r=3)
dim(W)
W <- sweep(W, 1, rowSums(W), FUN="/")
points(stations[W[,1]>0,], pch=20, col='blue')
points(stations[W[,25]>0,], pch=20, col='green')
points(stations[W[,17]>0,], pch=20, col='orange')

set.seed(12)
theta_sim <- (sin(knot$x/2)*cos(knot$y/2)+1)/50
theta_sim[theta_sim < 0.005] <- 0
theta_sim <- matrix(rep(theta_sim, n.t), ncol=n.t)
fields::image.plot(c(1,3,5,7,9), c(1,3,5,7,9), matrix(theta_sim[,1],5,5), col=terrain.colors(25))


alpha = 0.5
Z <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)
Epsilon_frechet <- matrix(NA, nrow=n.s, ncol=n.t)
for (iter in 1:n.t) {
  for (i in 1:k) {
    Z[i,iter] <- single_rejection_sampler(theta = theta_sim[i,iter])
  }
  # X[,iter] <- rfrechet(n.s,shape=(1/alpha)) * (rowSums(V[,iter]*(W^(1/alpha))))^alpha
  Epsilon_frechet[,iter] <- rfrechet(n.s,shape=1)
  X[,iter] <-  Epsilon_frechet[,iter]* ((W^(1/alpha))%*%Z[,iter])
}


ind=1
spatial_map(stations, var=X[,ind], tight.brks = TRUE, title=paste0('Time replicate #', ind))

y_true_star <-  (W^(1/alpha))%*%Z
log_v <- 0
for (i in 1:n.t) {
  for (j in 1:k) {
    log_v = log_v+f_H(Z[j,i],alpha = alpha,theta = theta_sim[j,i], log=TRUE)
  }
}

part1 = (sum((-2)*log(X)+log(y_true_star)) + (-sum(X^(-1)*y_true_star))) # p(X_t=x_t|v_t)
part2 = log_v  

(part1+part2)/(n.s*n.t) # -1.952435



###### ---------------------------------------------------------------------- ######
###### ----------------------- First initial guess -------------------------- ######
###### ---------------------------------------------------------------------- ######
Epsilon_frechet_indep <- matrix(rfrechet(n.s*n.t,shape=1), nrow=n.s)
W_alpha <- W^(1/alpha)
average_epsilon <- rowMeans(Epsilon_frechet_indep) # average over time
w_1 <- solve(t(W_alpha)%*%W_alpha)%*%t(W_alpha)%*%diag(1/average_epsilon)
Z_approx <- w_1%*%X
# Z_approx-Z

b_1 <- matrix(rep(0,k), ncol=1) 
w_2 <- diag(k)
b_2 <- matrix(rep(0.00001,k), ncol=1)
w_4 <- diag(k)
b_4 <- matrix(rep(0,k), ncol=1) 
w_3 <- 0*diag(k)
b_3 <- matrix(rep(0.01,k), ncol=1)

## Change to torch tensor
X_tensor <- torch_tensor(X,dtype=torch_float())
W_tensor <- torch_tensor(W,dtype=torch_float())

w_1 <- torch_tensor(w_1,dtype=torch_float(),requires_grad = TRUE)
w_2 <- torch_tensor(w_2,dtype=torch_float(),requires_grad = TRUE)
w_3 <- torch_tensor(w_3,dtype=torch_float(),requires_grad = TRUE)
w_4 <- torch_tensor(w_4,dtype=torch_float(),requires_grad = TRUE)
b_1 <- torch_tensor(b_1,dtype=torch_float(),requires_grad = TRUE)
b_2 <- torch_tensor(b_2,dtype=torch_float(),requires_grad = TRUE)
b_3 <- torch_tensor(b_3,dtype=torch_float(),requires_grad = TRUE)
b_4 <- torch_tensor(b_4,dtype=torch_float(),requires_grad = TRUE)

h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()
Epsilon <- matrix(abs(rnorm(k*n.t))+0.1, nrow=k)
Epsilon <- torch_tensor(Epsilon,dtype=torch_float())
v_t <- mu + sqrt(sigma_sq_vec)*Epsilon




###### ---------------------------------------------------------------------- ######
###### ---------------------- Second initial guess -------------------------- ######
###### ---------------------------------------------------------------------- ######
better_Z_approx <- matrix(data=NA, nrow=nrow(Z_approx), ncol=ncol(Z_approx))
for(iter in 1:n.t){
  cat('Finding good initial Z_t for time', iter, '\n')
  tmp = as_array(v_t[,iter])
  while(any(tmp>1e4)){
    cat('-- Bad initial Z_t from matrix projection', iter, '\n')
    tmp = as_array(v_t[,iter+1])
  }
  res <- optim(par=tmp, full_cond, W_alpha=W_alpha, X_t=X[,iter], gr=gradient_full_cond, method='CG',
               control = list(fnscale=-1, trace=0, maxit=10000))
  better_Z_approx[, iter] <- res$par
}

# Z = w_1 %*% X or Z^T = X^T %*% w_1^T
tmp <- qr.solve(a=t(X), b=t(better_Z_approx))
w_1 <- t(tmp)
w_1 <- torch_tensor(w_1,dtype=torch_float(),requires_grad = TRUE)

b_3 <- matrix(rep(-3,k), ncol=1)
b_3 <- torch_tensor(b_3,dtype=torch_float(),requires_grad = TRUE)

h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()
v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
y_approx <-  (W_tensor^(1/alpha))$mm(v_t)

X_approx <- matrix(NA, nrow=n.s, ncol=n.t)
for (iter in 1:n.t){
  X_approx[,iter] <- as_array(Epsilon_frechet[,iter]* y_approx[,iter])
  # cat(iter, as_array(mean(h)),as_array(mean(b_1)),b_1$storage()$data_ptr(),'\n')
}

spatial_map(stations, var=X_approx[,ind], tight.brks = TRUE, 
            title=paste0('Approx replicate #', ind), range = range(X_approx[,ind]),
            q25 = quantile(X[,ind], probs = 0.25), q75=quantile(X[,ind], probs = 0.75))
part1 = sum((-2)*log(X)+log(y_approx)) + (-sum(X^(-1)*y_approx)) # p(X_t=x_t|v_t)
(part1+part2)/(n.s*n.t) # -2.17531


## w_prime and b_prime for theta
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

Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
Epsilon_prime <- torch_tensor(Epsilon_prime,dtype=torch_float())

###### ---------------------------------------------------------------------- ######
###### --------------------   Loss and Optimization ------------------------- ######
###### ---------------------------------------------------------------------- ######

###### ----------------------  network parameters --------------------------- ######
learning_rate <- -1e-9
alpha_v <- 0.9

niter = 400000
n <- 1e3
# only depends on alpha; we are not updating alpha for now.
vec <- Zolo_A(pi*seq(1/2,n-1/2,1)/n, alpha)
Zolo_vec <- torch_tensor(matrix(vec, nrow=1,ncol=n), dtype=torch_float(), requires_grad = FALSE) 
Zolo_vec_double <- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1; const3 <- log(const1)
const4 <- -2; const5 <- -1
old_loss <- -Inf

for (t in 1:niter) {
  if(t==1000) { learning_rate <- -5e-9; alpha_v <- 0.9}
  if(t==5000) { learning_rate <- -1e-8; alpha_v <- 0.7}
  if(t==10000) { learning_rate <- -2e-8; alpha_v <- 0.5}
  if(t==20000) { learning_rate <- -2e-8; alpha_v <- 0.5}
  if(t==80000) { learning_rate <- -2e-8; alpha_v <- 0.4}
  if(t==100000) { learning_rate <- -1e-8; alpha_v <- 0.4}
  if(t==120000) { learning_rate <- -2e-8; alpha_v <- 0.4}
  if(t==140000) { learning_rate <- -3e-8; alpha_v <- 0.4}
  if(t==300000) { learning_rate <- -2e-8; alpha_v <- 0.4}

  ### -------- Forward pass --------
  Epsilon <- t(0.1+abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
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
  
  y_star <- W_tensor$pow(1/alpha)$mm(v_t)
  # log_v <- torch_zeros(c(k,n.t))
  # for (i in 1:n.t) {
  #   for (j in 1:k) {
  #     tmp = const1 * v_t[j,i]^{-const} * Zolo_vec * exp(-v_t[j,i]^(-const1) * Zolo_vec)
  #     log_v[j,i] = -theta_t[j,i]^alpha+log(tmp$mean()) - theta_t[j,i]*v_t[j,i] 
  #   }
  # }
  V_t <- v_t$view(c(k*n.t,1))
  Theta_t <- theta_t$view(c(k*n.t,1))
  part_log_v1  <- V_t$pow(-const)$mm(Zolo_vec) 
  part_log_v2  <- (-V_t$pow(-const1)$mm(Zolo_vec))$exp()
  part_log_v3 <- Theta_t$pow(alpha)-Theta_t$mul(V_t)
  part2 <- (part_log_v1$mul(part_log_v2)$mean(dim=2)$log()+part_log_v3$view(2500)$add(const3))$sum()
  if(as_array(part2) == -Inf) {
    part2_tmp <- part_log_v1$mul(part_log_v2)$mean(dim=2)$log()
    tmp_ind <- which.min(as_array(part2_tmp))
    part2_tmp[tmp_ind] <- part_log_v1$log()[tmp_ind,1] + (-V_t$pow(-const1)$mm(Zolo_vec))[tmp_ind,1]
    part2 <- (part2_tmp+part_log_v3$view(2500)$add(const3))$sum()
  }
  part1 = (X_tensor)$log()$sum()*const4 + y_star$log()$sum() - X_tensor$pow(const5)$mul(y_star)$sum()
  # part2 = log_v$sum()                                                                        # p(v_t|theta_t), p(theta_t)
  part4 = Epsilon$pow(2)$sum()/2 + Epsilon_prime$pow(2)$sum()/2 + sigma_sq_vec$log()$sum() + sigma_sq_vec_prime$log()$sum()
  res <- part1 + part2 + part4
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



plot(theta_sim[,1], as_array(theta_t[,1]), xlab=expression(paste('true ',theta)), ylab=expression(paste('CVAE ',theta)))
abline(a = 0, b = 1, col='orange', lty=2)



###### ---------------------------------------------------------------------- ######
###### ---------------------------   Emulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######

###### ---------  Run decoder to get the simulated weather processes -------- ######
n.sim<-500
station1_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station50_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station55_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station70_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station100_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)

### -------- Encoder for v_t --------
h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()
### -------- Encoder for v_t_prime --------
h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()

h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))

sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)

for(iter in 1:n.sim){
  if(iter %% 100==0) cat('iter=', iter, '\n')
  Epsilon <- t(0.1+abs(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k)))))
  Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
  
  Epsilon <- torch_tensor(Epsilon, dtype=torch_float())
  Epsilon_prime <- torch_tensor(Epsilon_prime, dtype=torch_float())
  
  ## re-parameterization trick
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  # Decoder
  l <- w_5$mm(v_t_prime)$add(b_5)$relu()
  l_1 <- w_6$mm(l)$add(b_6)$relu()
  theta_t <- w_7$mm(l_1)$add(b_7)$relu()
  
  y_star <- (W_tensor^(1/alpha))$mm(v_t)
  
  ##Decoder
  station1_Simulations[, iter] <- rfrechet(n.s,shape=(1)) * as_array((y_star[,1]))
  station50_Simulations[, iter] <- rfrechet(n.s,shape=(1)) * as_array((y_star[,50]))
  station55_Simulations[, iter] <- rfrechet(n.s,shape=(1)) * as_array((y_star[,55]))
  station70_Simulations[, iter] <- rfrechet(n.s,shape=(1)) * as_array((y_star[,70]))
  station100_Simulations[, iter] <- rfrechet(n.s,shape=(1)) * as_array((y_star[,100]))
}

## -------- time 1 --------
ind <- 1
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Simulated replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt3
ggsave("/Users/LikunZhang/Desktop/img1.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station1_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt31
ggsave("/Users/LikunZhang/Desktop/img2.png",width = 5.5, height = 5)

ind <- 1
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])))
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])), 
                 xlim=c(0,500), ylim=c(0,500))
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])), 
                 xlim=c(0,7), ylim=c(0,7))


## -------- time 55 --------
ind <- 55
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Simulated replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt3
ggsave("/Users/LikunZhang/Desktop/img3.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station55_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt31
ggsave("/Users/LikunZhang/Desktop/img4.png",width = 5.5, height = 5)

ind=55
extRemes::qqplot(X[,ind], station55_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[55])), 
                 ylab=expression(paste('Emulated ', X[55])))
extRemes::qqplot(X[,ind], station55_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[55])), 
                 ylab=expression(paste('Emulated ', X[55])), 
                 xlim=c(0,500), ylim=c(0,500))
extRemes::qqplot(X[,ind], station55_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[55])), 
                 ylab=expression(paste('Emulated ', X[55])), 
                 xlim=c(0,7), ylim=c(0,7))


## -------- time 70 --------
ind <- 70
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Simulated replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt3
ggsave("/Users/LikunZhang/Desktop/img3.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station70_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt31
ggsave("/Users/LikunZhang/Desktop/img4.png",width = 5.5, height = 5)

ind=70
extRemes::qqplot(X[,ind], station70_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[70])), 
                 ylab=expression(paste('Emulated ', X[70])))
extRemes::qqplot(X[,ind], station70_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[70])), 
                 ylab=expression(paste('Emulated ', X[70])), 
                 xlim=c(0,500), ylim=c(0,500))
extRemes::qqplot(X[,ind], station70_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[70])), 
                 ylab=expression(paste('Emulated ', X[70])), 
                 xlim=c(0,7), ylim=c(0,7))


## -------- time 100 --------
ind <- 100
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Simulated replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt3
ggsave("/Users/LikunZhang/Desktop/img5.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station100_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75)
plt31
ggsave("/Users/LikunZhang/Desktop/img6.png",width = 5.5, height = 5)

quantile(X[,ind],prob=0.95)
ind=100
extRemes::qqplot(X[,ind], station100_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[100])), 
                 ylab=expression(paste('Emulated ', X[100])))
extRemes::qqplot(X[,ind], station100_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[100])), 
                 ylab=expression(paste('Emulated ', X[100])), 
                 xlim=c(0,500), ylim=c(0,500))
extRemes::qqplot(X[,ind], station100_Simulations[,floor(n.sim/2)], 
                 xlab=expression(paste('Simulated ', X[100])), 
                 ylab=expression(paste('Emulated ', X[100])), 
                 xlim=c(0,7), ylim=c(0,7))
