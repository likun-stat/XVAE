# setwd("C:/Users/liaoy/OneDrive - University of Missouri/VAE Project")
setwd('~/Desktop/GEV-GP_VAE/extCVAE/')
source("utils.R")
library(autodiffr)

#### Simulation ####
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
V <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)

for (iter in 1:n.t) {
  for (i in 1:k) {
    V[i,iter] <- double_rejection_sampler(theta = theta_sim[i,iter],alpha = alpha)
  }
  # X[,iter] <- rfrechet(n.s,shape=(1/alpha)) * (rowSums(V[,iter]*(W^(1/alpha))))^alpha
  X[,iter] <- rfrechet(n.s,shape=(1/alpha)) * (rowSums((W^(1/alpha))%*%diag(V[,iter])))^alpha
}


ind=53
spatial_map(stations, var=X[,ind], tight.brks = TRUE, title=paste0('Time replicate #', ind))

y_true <-  rowSums((W^(1/alpha))%*%diag(V))
log_v <- 0
for (i in 1:n.t) {
  for (j in 1:k) {
    log_v = log_v+log(f_H(V[j,i],alpha = alpha,theta = theta_sim[j,i]))
  }
}

part1 = (-n.s * log(alpha) * n.t + sum((-1/alpha-1)*log(X)+log(y_true)) + (-sum(X^(-1/alpha-1)*y_true))) # p(X_t=x_t|v_t)
part2 = log_v  

(part1+part2)/(n.s*n.t) # 23.86811


#### MLP Decoder ####
ELBO_flexible_extreme <- function(phi, X, Epsilon, Epsilon_prime, alpha, W){
  n.s <- nrow(X); n.t <- ncol(X); k <- nrow(Epsilon)
  
  w_1 <- array(phi[1:(k*n.s)],dim=c(k,n.s))
  w_2 <- array(phi[(1+k*n.s):(k*k+k*n.s)],dim = c(k,k))
  w_3 <- array(phi[(1+k*k+k*n.s):(2*k*k+k*n.s)],dim = c(k,k))
  w_4 <- array(phi[(1+2*k*k+k*n.s):(3*k*k+k*n.s)],dim = c(k,k))
  b_1 <- phi[(1+3*k*k+k*n.s):(k+3*k*k+k*n.s)]
  b_2 <- phi[(1+k+3*k*k+k*n.s):(2*k+3*k*k+k*n.s)]
  b_3 <- phi[(1+2*k+3*k*k+k*n.s):(3*k+3*k*k+k*n.s)]
  b_4 <- phi[(1+3*k+3*k*k+k*n.s):(4*k+3*k*k+k*n.s)]
  
  w_1_prime <- array(phi[(1+4*k+3*k*k+k*n.s):(4*k+3*k*k+2*k*n.s)],dim=c(k,n.s))
  w_2_prime <- array(phi[(1+4*k+3*k*k+2*k*n.s):(4*k+4*k*k+2*k*n.s)],dim = c(k,k))
  w_3_prime <- array(phi[(1+4*k+4*k*k+2*k*n.s):(4*k+5*k*k+2*k*n.s)],dim = c(k,k))
  w_4_prime <- array(phi[(1+4*k+5*k*k+2*k*n.s):(4*k+6*k*k+2*k*n.s)],dim = c(k,k))
  b_1_prime <- phi[(1+4*k+6*k*k+2*k*n.s):(5*k+6*k*k+2*k*n.s)]
  b_2_prime <- phi[(1+5*k+6*k*k+2*k*n.s):(6*k+6*k*k+2*k*n.s)]
  b_3_prime <- phi[(1+6*k+6*k*k+2*k*n.s):(7*k+6*k*k+2*k*n.s)]
  b_4_prime <- phi[(1+7*k+6*k*k+2*k*n.s):(8*k+6*k*k+2*k*n.s)]
  
  w_5 <- array(phi[(1+8*k+6*k*k+2*k*n.s):(8*k+7*k*k+2*k*n.s)],dim = c(k,k))
  w_6 <- array(phi[(1+8*k+7*k*k+2*k*n.s):(8*k+8*k*k+2*k*n.s)],dim = c(k,k))
  w_7 <- array(phi[(1+8*k+8*k*k+2*k*n.s):(8*k+9*k*k+2*k*n.s)],dim = c(k,k))
  b_5 <- phi[(1+8*k+9*k*k+2*k*n.s):(9*k+9*k*k+2*k*n.s)]
  b_6 <- phi[(1+9*k+9*k*k+2*k*n.s):(10*k+9*k*k+2*k*n.s)]
  b_7 <- phi[(1+10*k+9*k*k+2*k*n.s):(11*k+9*k*k+2*k*n.s)]
  
  # Encoder for v_t
  h <- relu(w_1 %m% X + b_1) 
  h_1 <- relu(w_2 %m% h + b_2)
  sigma_sq_vec <- exp(w_3 %m% h_1 + b_3)
  mu <- w_4 %m% h_1 + b_4
  
  # Encoder for v_t_prime
  h_prime <- relu(w_1_prime %m% X + b_1_prime) 
  h_1_prime <- relu(w_2_prime %m% h_prime + b_2_prime)
  sigma_sq_vec_prime <- exp(w_3_prime %m% h_1_prime + b_3_prime)
  mu_prime <- w_4_prime %m% h_1_prime + b_4_prime
  
  # ## re-parameterization trick
  v_t <- mu + sqrt(sigma_sq_vec)*Epsilon
  v_t_prime <- mu_prime + sqrt(sigma_sq_vec_prime)*Epsilon_prime
  
  # Decoder
  l <- relu(w_5 %m% v_t_prime + b_5)
  l_1 <- relu(w_6 %m% l + b_6)
  theta_t <- array(relu(w_7 %m% l_1 + b_7),dim=c(k,n.t))
  
  learned_Z_t <- exp(v_t)
  y <- W^(1/alpha) %m% learned_Z_t
  
  # theta_prior <- ifelse(theta_t<=2 & theta_t>=0, 1, 0) # unif(0,2)
  log_v <- 0
  for (i in 1:n.t) {
    for (j in 1:k) {
      log_v = log_v+log(f_H(learned_Z_t[j,i],alpha = alpha,theta = theta_t[j,i]))+v_t[j,i]
    }
  }
  
  part1 = -(-n.s * log(alpha) * n.t + sum((-1/alpha-1)*log(X)+log(y)) + (-sum(X^(-1/alpha-1)*y))) # p(X_t=x_t|v_t)
  part2 = - log_v                                                                        # p(v_t|theta_t), p(theta_t)
  part3 = -sum(Epsilon^2)/2- sum(log(sigma_sq_vec)) 
  part4 = - sum(Epsilon_prime^2)/2 -sum(log(sigma_sq_vec_prime))
  res <- part1 + part2 + part3 + part4
  return(-res/(n.s*n.t))
}

## -------------- 5. Initializae & Start Algorithm 1 -------------- 

### (1) Initialize
# --- w_1, w_1_prime: k x n.s
# --- w_2, w_3, w_4, w_2_prime, w_3_prime, w_4_prime,: k x k
# --- b_1, b_2, b_3, b_4, b_1_prime, b_2_prime, b_3_prime, b_4_prime: k x 1
# --- w_5, w_6, w_7: k x k
# --- b_5, b_6, b_7: k x 1


# phi start values
w_1 <- rnorm(k*n.s,0,0.001)
w_2 <- rnorm(k*k,0,0.001)
w_3 <- rnorm(k*k,0,0.001)
w_4 <- rnorm(k*k,0,0.001)
b_1 <- rnorm(k)
b_2 <- rnorm(k)
b_3 <- rnorm(k,0,0.001)
b_4 <- runif(k,-0.001,0.001)

w_1_prime <- rnorm(k*n.s,0,0.001)
w_2_prime <- rnorm(k*k,0,0.001)
w_3_prime <- rnorm(k*k,0,0.001)
w_4_prime <- rnorm(k*k,0,0.001)
b_1_prime <- rnorm(k)
b_2_prime <- rnorm(k)
b_3_prime <- rnorm(k,0,0.001)
b_4_prime <- runif(k,10,20)

w_5 <- rnorm(k*k,0,0.001)
w_6 <- rnorm(k*k,0,0.001)
w_7 <- rnorm(k*k,0,0.001)
b_5 <- rnorm(k)
b_6 <- rnorm(k)
b_7 <- rnorm(k)

phi_star <- c(w_1, w_2, w_3, w_4, 
              b_1, b_2, b_3, b_4,
              w_1_prime, w_2_prime, w_3_prime, w_4_prime,
              b_1_prime, b_2_prime, b_3_prime, b_4_prime,
              w_5, w_6, w_7, b_5, b_6, b_7)

phi <- phi_star

Epsilon <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))

ELBO_flexible_extreme(phi=phi_star, X=X, Epsilon=Epsilon, Epsilon_prime=Epsilon_prime, 
                      W=W, alpha = alpha)
# 

## -------------- 6. Test whether autodiffr works-------------- 

phi_grad_atudiffr <- makeGradFunc(ELBO_flexible_extreme, X=X, Epsilon=Epsilon, 
                                  Epsilon_prime=Epsilon_prime, 
                                  W=W, alpha = alpha)
grad <- phi_grad_atudiffr(phi_star)

grad[14729]
#
sum(is.na(grad))

phi_star_new <- phi_star;
phi_star_new[14729] <- phi_star_new[14729] + 1e-8
(ELBO_flexible_extreme(phi_star_new, X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W, 
                       alpha = alpha)-ELBO_flexible_extreme(phi_star, X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W, alpha = alpha))/1e-8
#

#### check gradient accuracy
# tmp_fun <- function(xtmp){
#   phi_star_new <- phi_star; 
#   phi_star_new[104] <- phi_star_new[104] + xtmp
#   return(ELBO_flexible_extreme(phi_star_new, X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W,alpha = alpha))
# }
# Xs <- seq(-0.2,0.1, length.out=100)
# fn <- rep(NA, 100)
# for(i in 1:100){
#   fn[i] = tmp_fun(Xs[i])
# }
# plot(Xs, fn, type='l')
# 
# abline(a=tmp_fun(0) - grad[104]*phi_star[104], b= grad[104], col='red')
# abline(v=phi_star[104])
# 
# phi_star_new <- phi_star; 
# phi_star_new[104] <- phi_star_new[104] + 1e-12
# grad_approx = (ELBO_flexible_extreme(phi_star_new, X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W,alpha = alpha)-
#                  ELBO_flexible_extreme(phi_star, X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W,alpha = alpha))/1e-12
# # -0.2022013
# 
# abline(a=tmp_fun(0) - grad_approx*phi_star[104], b= grad_approx, col='blue')
# 

#### SGD ####


GradientDescent_allParams_ELBO_MLP_Flexible_Extreme <- function(X, Epsilon, Epsilon_prime,W,alpha,
                                                  phi_star,
                                                  gamma = 0.1, ## Starting learning_rate
                                                  max.it=600, rel.tol = 1e-10, trace=FALSE){
  n.s <- nrow(X); n.t <- ncol(X); k <- ncol(Epsilon)
  ## Make gradient functions
  phi_grad_atudiffr <- makeGradFunc(ELBO_flexible_extreme, X=X, Epsilon=Epsilon, 
                                    Epsilon_prime=Epsilon_prime, 
                                    W=W, alpha = alpha)
  grad <- phi_grad_atudiffr(phi_star)
  
  phi_current <- phi_star  # Initial values
  grads <- -phi_grad_atudiffr(phi_current) ## MINIMIZATION needs negative grad

  for (iter in 1:max.it){
    grads_old <- grads
    phi_old <- phi_current

    ## (1) Update phi
    phi_current <- phi_old - gamma * grads_old
    func_val <- ELBO_flexible_extreme(phi_current,X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W,alpha = alpha)
    while(!is.finite(func_val)){
      gamma <- gamma/5
      phi_current <- phi_old - gamma * grads_old
      func_val <- ELBO_flexible_extreme(phi_current,X=X, Epsilon=Epsilon,Epsilon_prime=Epsilon_prime,W=W,alpha = alpha)
    }
    ## (2) Update gradient
    grads <- -phi_grad_atudiffr(phi_current)

    ## (3) Determine the gamma value
    num = sum((grads-grads_old)*(phi_current-phi_old))
    denom = sum((grads-grads_old)^2)
    if(denom<1e-20) break
    gamma = abs(num/denom)

    if(trace){
      cat('iter=', iter, ', func=', func_val, ', gamma=', gamma, ', rel.err=', denom, '\n')
    }
  }
  return(phi_current)
}


## -------------- 7. Stochastic gradient descent--------------
### Update theta_phi until convergence
abs.err <- 40
max.it <- 500
minibatch_size <- 100
## plot parameters
if(minibatch_size==10) pt.col <- 'blue'; line.col<- '#66718a'
if(minibatch_size==50) pt.col <- 'green'; line.col<- '#74967a'
if(minibatch_size==100) pt.col <- 'red'; line.col<- '#947473'
if(minibatch_size==10) add=FALSE else add=TRUE


count <- 1
while (count < max.it){
  ## (a) Random minibatch of data
  minibatch_index <- sample(1:n.t, minibatch_size)

  ## (b) Random noise for every datapoint in the minibatch
  Epsilon <- t(mvtnorm::rmvnorm(minibatch_size, mean=rep(0, k), sigma = diag(rep(1, k))))
  Epsilon_prime <- t(mvtnorm::rmvnorm(n.t, mean=rep(0, k), sigma = diag(rep(1, k))))
  
  ## (c) SGD optimizer
  phi_new <- GradientDescent_allParams_ELBO_MLP_Flexible_Extreme(X[, minibatch_index], Epsilon, Epsilon_prime,W,alpha,
                                                         phi_star, max.it=2, rel.tol = 1e-10, trace=TRUE)
  new.abs.err <- sqrt(sum((phi_new-phi_star)^2))
  phi_star <- phi_new
  Elbo_val <- ELBO_flexible_extreme(phi_star, X=X[, minibatch_index], Epsilon=Epsilon, Epsilon_prime=Epsilon_prime, 
                                                                  W=W, alpha = alpha)
  cat('iter=', count, ' squared.error=', abs.err, ' ELBO=', Elbo_val,  '\n')
  if(count == 1 & !add) {
    plot(1:max.it, (1:max.it)/3, type='n', xlab='iter', ylab='squared.err', ylim=c(0, new.abs.err+15))
    if(count>1) lines(c(count-1, count), c(abs.err, new.abs.err), col=line.col)
    points(count, new.abs.err, pch=20, col=pt.col)}
  count = count + 1
  abs.err <- new.abs.err
}

legend('topright', pch=20, col=c('blue', 'green', 'red'), legend = c("10", "50", "100 (full data)"), title="Minibatch size")
# paste(round(theta_phi_star,3), collapse = ' ')

