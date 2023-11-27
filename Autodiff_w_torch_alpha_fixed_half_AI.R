library(torch)

setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Simulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))
knot <- expand.grid(x=c(1,3,5,7,9),y=c(1,3,5,7,9))
# plot(stations)
# points(knot, pch="+", col='red', cex=2)

k = nrow(knot)
n.s <- nrow(stations)
n.t <- 100 # n.t <- 500

eucD <- rdist(stations,as.matrix(knot))

r=3
W <- wendland(eucD, r=r)
W <- sweep(W, 1, rowSums(W), FUN="/")
# points(stations[W[,1]>0,], pch=20, col='blue')
# points(stations[W[,25]>0,], pch=20, col='green')
# points(stations[W[,17]>0,], pch=20, col='orange')

dat <- cbind(circleFun(unlist(knot[1,]), diameter = r*2, npoints = 100), group=1)
for(iter in 2:nrow(knot)){
  dat <- rbind(dat, cbind(circleFun(unlist(knot[iter,]), diameter = r*2, npoints = 100), group=iter))
}


library(ggh4x)
ggplot(knot) + geom_point(aes(x=x, y=y), shape='+', size=6, color='red') +
  geom_path(data=dat[dat$group==13,], aes(x=x, y=y, group=factor(group))) + 
  scale_x_continuous(expand=c(0,0), limits = c(0,10)) +
  scale_y_continuous(expand=c(0,0), limits = c(0,10)) + 
  ggtitle("Knot locations") +
  theme(plot.title = element_text(hjust = 0.5, face="bold"))+
  force_panelsizes(rows = unit(3.5, "in"),
                   cols = unit(3.5, "in"))
# ggsave("~/Desktop/GEV-GP_VAE/Figures/knots.pdf", height=4.2, width=4.3)

theta_sim_flex <- (sin(knot$x/2)*cos(knot$y/2)+1)/50
theta_sim_flex[theta_sim_flex < 0.005] <- 0
pal <- RColorBrewer::brewer.pal(9, "YlGnBu")
ggplot(knot) + geom_point(aes(x=x, y=y, colour = theta_sim_flex), size=6) +
  scale_x_continuous(expand=c(0,0), limits = c(0,10)) + scale_color_gradientn(colors = pal, name=expression(theta[k]), limits =c(0,0.04)) +
  scale_y_continuous(expand=c(0,0), limits = c(0,10)) + geom_point(dat=knot[theta_sim_flex==0,], aes(x=x,y=y), shape=1, size=6)+
  ggtitle("Model III") +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust = 0.5, face="bold"))+
  force_panelsizes(rows = unit(3.5, "in"),
                   cols = unit(3.5, "in"))
# ggsave("~/Desktop/GEV-GP_VAE/Figures/theta_k_model_III.pdf", height=4.2, width=4.5)

theta_sim <- (sin(knot$x^2/2)*cos(knot$y/4)+1)/40
theta_sim <- matrix(rep(theta_sim, n.t), ncol=n.t)

pal <- RColorBrewer::brewer.pal(9, "YlGnBu")
ggplot(knot) + geom_point(aes(x=x, y=y, colour = theta_sim[,1]), size=6) +
  scale_x_continuous(expand=c(0,0), limits = c(0,10)) + scale_color_gradientn(colors = pal, name=expression(theta[k])) +
  scale_y_continuous(expand=c(0,0), limits = c(0,10)) + geom_point(dat=knot[theta_sim[,1]==0,], aes(x=x,y=y), shape=1, size=6)+
  guides(color = 'none')+ggtitle("Model II") +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust = 0.5, face="bold"))+
  force_panelsizes(rows = unit(3.5, "in"),
                   cols = unit(3.5, "in"))
# ggsave("~/Desktop/GEV-GP_VAE/Figures/theta_k_model_II.pdf", height=4.2, width=4.3)



alpha = 0.5
tau <- 0.1
m <- 0.85

Z <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)
Epsilon_frechet <- matrix(NA, nrow=n.s, ncol=n.t)
# set.seed(12)
for (iter in 1:n.t) {
  for (i in 1:k) {
    Z[i,iter] <- single_rejection_sampler(theta = theta_sim[i,iter])
  }
  Epsilon_frechet[,iter] <-  rfrechet(n.s, shape=1, scale = tau, location = m)
  X[,iter] <-  Epsilon_frechet[,iter]* ((W^(1/alpha))%*%Z[,iter])
}


ind=1
spatial_map(stations, var=X[,ind], tight.brks = TRUE, title=paste0('Time replicate #', ind), raster = FALSE)

y_true_star <-  (W^(1/alpha))%*%Z
log_v <- 0
for (i in 1:n.t) {
  for (j in 1:k) {
    log_v = log_v+f_H(Z[j,i],alpha = alpha,theta = theta_sim[j,i], log=TRUE)
  }
}

part1 = sum((-2)*log(X)+log(y_true_star)) + (-tau*sum(X^(-1)*y_true_star)) # p(X_t=x_t|v_t)
part2 = log_v  

log(tau) + (part1 + part2)/(n.s*n.t) # -1.952435





###### ---------------------------------------------------------------------- ######
###### ---------------------------- Setting up ------------------------------ ######
###### ---------------------------------------------------------------------- ######
holdout <- c(1569, 557, 1685, 287, 614, 1169, 329, 487, 855, 851, 1654, 1522, 858, 1840, 619, 576, 990, 1514, 1760, 1023, 1127, 316, 1075, 733, 1314, 
             650, 129, 811, 955, 282, 1167, 1466, 285, 1944, 1706, 1072, 501, 1740, 511, 1319, 536, 693, 988, 1238, 1832, 737, 1363, 1370, 1699, 1067, 
             1025, 29, 1614, 1942, 838, 1820, 1652, 233, 1317, 573, 369, 451, 86, 1507, 327, 1646, 355, 1843, 812, 1073, 361, 1340, 1266, 1464, 758, 1841, 
             818, 247, 751, 219, 135, 111, 532, 377, 408, 1978, 1589, 912, 467, 1380, 130, 1987, 1089, 1836, 359, 105, 1148, 1101, 1242, 1634)
plot(stations)
points(stations[holdout,], pch=16, col="blue")


X_holdout <- X[holdout,]
stations_holdout <- as.matrix(stations[holdout,])

X <- X[-holdout,]
stations <- as.matrix(stations[-holdout,])


k = nrow(knot)
n.s <- nrow(stations)
n.t <- 100 # n.t <- 500
stations <- data.frame(stations)

W_holdout <- W[holdout, ]
W <- W[-holdout, ]
# Update because we held out 100 locations

ggplot(knot) + geom_point(aes(x=x, y=y), shape='+', size=6, color='red') +
  geom_point(data=stations[which(W[,1]>0.001),], aes(x = x, y = y), colour=scales::alpha("blue", 0.1))+
  geom_point(data=stations[which(W[,25]>0.1),], aes(x = x, y = y), colour=scales::alpha("green", 0.1))+
  geom_point(data=stations[which(W[,12]>0.001),], aes(x = x, y = y), colour=scales::alpha("yellow", 0.1))+
  geom_point(data=stations[which(apply(W,1,function(x) any(is.na(x)))),], aes(x = x, y = y), colour="black")





# alpha = 0.5; tau <- 0.1; m <- 0.85 # defined already
###### ---------------------------------------------------------------------- ######
###### ----------------------- First initial guess -------------------------- ######
###### ---------------------------------------------------------------------- ######

## -------------------- Initial guess for the latent Z variables --------------------
W_alpha <- W^(1/alpha)
Z_approx <- array(NA, dim=c(k, n.t))
for (iter in 1:n.t){
  cat('Finding good initial Z_t for time', iter, '\n')
  Z_approx[,iter] <- relu(qr.solve(a=W_alpha, b=X[,iter]))
}
plot(Z_approx[,ind],Z[,ind])
abline(a = 0, b = 1, col='orange', lty=2)



better_Z_approx <- matrix(data=NA, nrow=k, ncol=n.t)
for(iter in 1:n.t){
  cat('Improving Z_t for time', iter, '\n')
  initial_Z_t = Z_approx[,iter]/11 # Divide by 11 so 1/(X_t/Y_t-m)>0.
  initial_Z_t[initial_Z_t==0] <- 0.01
  Y_t_tmp <- (W_alpha)%*%initial_Z_t
  tmp <- 1/(X[,iter]/Y_t_tmp-m)
  index <- 1
  while(any(tmp < 0)){
    # if(index < 2) cat('-- Bad initial Z_t from matrix projection', iter, '\n')
    initial_Z_t = Z_approx[,iter]/(11+index)
    Y_t_tmp <- (W_alpha)%*%initial_Z_t
    tmp <- 1/(X[,iter]/Y_t_tmp-m)
    index <- index + 1
  }
  if(any(tmp == 0)) initial_Z_t <- rep(0.1,k)
  if(any(initial_Z_t == 0)){
    res <- optim(par=initial_Z_t, full_cond, W_alpha=W_alpha, X_t=X[,iter], tau=tau, m=m, gr=gradient_full_cond, method='CG',
                 control = list(fnscale=-1, trace=0, maxit=5000))
    initial_Z_t <- res$par
  }
  out <- tryCatch({ res <- optim(par=log(initial_Z_t), full_cond_logscale, W_alpha=W_alpha, X_t=X[,iter], tau=tau, m=m, gr=gradient_full_cond_logscale, method='CG',
                          control = list(fnscale=-1, trace=0, maxit=10000))
                        out <- exp(res$par)}, 
           error = function(e) {cat('-- Switching to orginal scale for time', iter, '\n')
                    res <- optim(par=initial_Z_t, full_cond, W_alpha=W_alpha, X_t=X[,iter], tau=tau, m=m, gr=gradient_full_cond, method='CG',
                                 control = list(fnscale=-1, trace=0, maxit=5000))
                    initial_Z_t <- res$par
                                res <- optim(par=log(initial_Z_t), full_cond_logscale, W_alpha=W_alpha, X_t=X[,iter], tau=tau, m=m, gr=gradient_full_cond_logscale, method='CG',
                                             control = list(fnscale=-1, trace=0, maxit=10000))
                                return(exp(res$par))})
  better_Z_approx[, iter] <- out
}

Z_approx <- better_Z_approx

Y_star <- (W_alpha)%*%(Z_approx)
Y_approx <- Y_star - relu(Y_star-X)
ind <- 3
tmp_range <- range(c(log(Y_approx[,ind])), log(X[,ind]))
ggplot(stations) + geom_point(aes(x=x, y=y, color=log(X[,ind]))) +
  scale_color_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA, limits=tmp_range) 

ggplot(stations) + geom_point(aes(x=x, y=y, color=log(Y_approx[,ind]))) +
  scale_color_gradientn(colours = topo.colors(100), name = paste0('Smooth replicate #', ind), na.value = NA, limits=tmp_range) 






## -------------------- Initial values for the weights --------------------
# Z = w_1 %*% X or Z^T = X^T %*% w_1^T
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
learning_rate <- -1e-15
alpha_v <- 0.9
lrelu <- nn_leaky_relu(-0.01)

niter = 80000
n <- 1e3
# only depends on alpha; we are not updating alpha for now.
vec <- Zolo_A(pi*seq(1/2,n-1/2,1)/n, alpha)
Zolo_vec <- torch_tensor(matrix(vec, nrow=1,ncol=n), dtype=torch_float(), requires_grad = FALSE) 
Zolo_vec_double <- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1; const3 <- log(const1)
W_alpha_tensor <- W_tensor$pow(1/alpha)
ones <- matrix(rep(1, n.t), ncol=1)
ones <- torch_tensor(ones,dtype=torch_float(),requires_grad = FALSE)
old_loss <- -Inf

# set.seed(86)
# start_time <- Sys.time()
for (t in 1:niter) {
  if(t==5000) { learning_rate <- -1e-15; alpha_v <- 0.9}
  if(t==20000) { learning_rate <- -5e-15; alpha_v <- 0.8}
  if(t==30000) { learning_rate <- -1e-14; alpha_v <- 0.7}
  if(t==40000) { learning_rate <- -1e-14; alpha_v <- 0.6}
  if(t==120000) { learning_rate <- -1e-14; alpha_v <- 0.4}
  if(t==140000) { learning_rate <- -2e-14; alpha_v <- 0.4}
  if(t==180000) { learning_rate <- -1e-13; alpha_v <- 0.4}
  if(t==200000) { learning_rate <- -1e-12; alpha_v <- 0.4}
  if(t==300000) { learning_rate <- -1e-11; alpha_v <- 0.4}
  
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
  # h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
  # h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
  # h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
  # theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
  h_1_prime_row_means <- 2*h_1_prime$mm(ones)/n.t
  h_1_prime_to_theta <- h_1_prime_row_means$pow(-2)
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
  if(leak>0 & leak<=175) standardized$abs_()
  leak2 <- as_array(sum(standardized==0))
  if(leak2>0 & leak2<=120) standardized$add_(1e-07)
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
# plot(theta_sim[,3], as_array(theta_t[,16]), xlab=expression(paste('true ',theta)), ylab=expression(paste('CVAE ',theta)))
# abline(a = 0, b = 1, col='orange', lty=2)
# 
# conf_theta <- t(apply(as_array(theta_t),1, function(x) quantile(x, p=c(0.025,0.975))))
# 
# 
# plot_data <- data.frame(true=theta_sim[,1], lower = conf_theta[,1], upper = conf_theta[,2])
# ggplot(plot_data) + 
#   geom_errorbar(aes(x=true, ymin=lower, ymax=upper)) +
#   geom_abline(intercept = 0)
  
###### ---------------------------------------------------------------------- ######
###### ---------------------------   Emulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######

###### ---------  Run decoder to get the simulated weather processes -------- ######
n.sim <- 1000
station1_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station50_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station55_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station70_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station100_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
Theta_t_estimations <- array(NA, dim = c(nrow(knot), n.t, n.sim))

### -------- Encoder for v_t --------
h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()

### -------- Encoder for v_t_prime --------
h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()

### -------- Activation via Laplace transformation --------
# h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
# h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
# h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
# theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
h_1_prime_row_means <- 2*h_1_prime$mm(ones)/n.t
h_1_prime_to_theta <- h_1_prime_row_means$pow(-2)
theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))

sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)

for(iter in 1:n.sim){
  if(iter %% 100==0) cat('iter=', iter, '\n')
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
  Theta_t_estimations[,,iter] <- as_array(theta_t)
  station1_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,1]))
  station50_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,50]))
  station55_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,55]))
  station70_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,70]))
  station100_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,100]))
}

save(Theta_t_estimations, file=paste0("../theta_AI_", 1, ".RData"))

theta_t_array <- as_array(theta_t)
save(theta_t_array, Theta_t_estimations, station1_Simulations, station50_Simulations, station55_Simulations, station70_Simulations, station100_Simulations, 
     file="../simulation_VAE_results_AI.RData" )


## -------- time 1 --------
ind <- 1
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Simulated replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, raster=FALSE) +
  
plt3
# ggsave("/Users/LikunZhang/Desktop/img1.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station1_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, raster=FALSE)
plt31
# ggsave("/Users/LikunZhang/Desktop/img2.png",width = 5.5, height = 5)

ind <- 1
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], regress = FALSE,  
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])))
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], regress = FALSE,  
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])), 
                 xlim=c(0,50), ylim=c(0,50))
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], regress = FALSE,  
                 xlab=expression(paste('Simulated ', X[1])), 
                 ylab=expression(paste('Emulated ', X[1])), 
                 xlim=c(0,7), ylim=c(0,7))


## -------- time 55 --------
ind <- 55
range_t <- range(X[,ind])
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

library(ggtext)
ylab = paste0("<span style='font-size: 16pt'>**Model  II**</span><br><span style='font-size: 13pt'>y</span>")
pal <- RColorBrewer::brewer.pal(9,"OrRd")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal, ylab=ylab,
                    title = paste0('Simulated replicate #', 1), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, raster=FALSE, show.legend = FALSE) +
        theme(axis.title.y = ggtext::element_markdown())
plt3
ggsave("~/Desktop/GEV-GP_VAE/Figures/sim_AI.png",width = 4.85, height = 4.9)

plt31 <- spatial_map(stations, var=station55_Simulations[,floor(n.sim/2)], pal = pal,
                     title = "extVAE", legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, raster=FALSE, show.legend = FALSE, show.axis.y = FALSE)
plt31
ggsave("~/Desktop/GEV-GP_VAE/Figures/emu_AI.png",width = 4.3, height = 4.9)


load("~/Desktop/GEV-GP_VAE/AI_hetgp_emulation.RData")
tmp_emu_gp <- p_org$mean
plt32 <- spatial_map(stations, var=tmp_emu_gp+rnorm(n=n.s, sd=sqrt(p_org$nugs)), pal = pal,
                     title = "hetGP", legend.name = "Emulated\n values", 
                     brks.round = 2, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, raster=FALSE, show.legend = TRUE, show.axis.y = FALSE)
plt32
ggsave("~/Desktop/GEV-GP_VAE/Figures/het_AI.png",width = 5.4, height = 4.9)


ind=1
png(filename="~/Desktop/GEV-GP_VAE/Figures/qq_emu_ai.png",width = 400, height = 380)
par(mar = c(5.1, 6.1, 4.1, 2.1))
extRemes::qqplot(X[,ind], station1_Simulations[,floor(n.sim/2)], regress = FALSE,  
                 xlab=expression(paste('Simulated ', X[1])), 
                 xlim=c(0,50), ylim=c(0,50), 
                 ylab=bquote(atop(bold("Model II"), paste('Emulated ', X[1]))), cex.lab = 1.2,
                 main='extVAE')
dev.off()


png(filename="~/Desktop/GEV-GP_VAE/Figures/qq_het_AI.png",width = 400, height = 380)
par(mar = c(5.1, 6.1, 4.1, 2.1))
extRemes::qqplot(X[,ind], tmp_emu_gp, regress = FALSE,
                 xlab=expression(paste('Simulated ', X[1])),
                 ylab=bquote(atop(bold(" "), paste('Emulated ', X[1]))), cex.lab = 1.2,
                 main='hetGP', xlim=c(0,50), ylim=c(0,50))
dev.off()









###### ---------------------------------------------------------------------- ######
###### -------------------------- Spatial prediction ------------------------ ######
###### ---------------------------------------------------------------------- ######

###### ---------------  Interpolate b_8 onto holdout locations -------------- ######
library(akima)
b_8_holdout <- matrix(NA, nrow=100, ncol=n.t)
for(t in 1:n.t){
  b_8_holdout[,t] <- interpp(x=stations[,1], y=stations[,2], z=as_array(b_8[,t]), xo=stations_holdout[,1], yo=stations_holdout[,2], linear=FALSE, extrap=TRUE)$z
}

b_8_holdout_tensor <- torch_tensor(b_8_holdout,dtype=torch_float())


###### ---------  Run decoder to get the simulated weather processes -------- ######
n.sim <- 1
station_Simulations <- matrix(NA, nrow=100, ncol=n.t)
W_holdout_tensor <- torch_tensor(W_holdout,dtype=torch_float())
W_holdout_alpha_tensor <- W_holdout_tensor$pow(1/alpha)


### -------- Encoder for v_t --------
h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()

### -------- Encoder for v_t_prime --------
h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()

### -------- Activation via Laplace transformation --------
# h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
# h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
# h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
# theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
h_1_prime_row_means <- 2*h_1_prime$mm(ones)/n.t
h_1_prime_to_theta <- h_1_prime_row_means$pow(-2)
theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))

sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)

for(iter in 1:n.sim){
  if(iter %% 100==0) cat('iter=', iter, '\n')
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
  
  y_star <- lrelu(W_holdout_alpha_tensor$mm(v_t)$add(b_8_holdout))
  
  ##Decoder
  station_Simulations <- matrix(rfrechet(100*n.t,shape=1, location = m, scale = tau), nrow=100) * as_array(y_star)
}


### -------- MSPE ---------
tmp <- (as_array(y_star) - X_holdout)^2
MSPE <- apply(tmp, 2, function(x) mean(x[x<1e5]))
boxplot(log(MSPE))


### -------- CRPS ---------
CRPS <- matrix(NA, nrow=100, ncol=n.t)
upper_limit <- max(max(station_Simulations), max(X_holdout))+1
for(loc in 1:100){
  ecdf_tmp <- ecdf(X_holdout[loc, ])
  cat("loc=", loc, "\n")
  for(t in 1:n.t){
    tmp_integrand <- function(z) (ecdf_tmp(z) - I(station_Simulations[loc,t]<=z))^2 
    tmp_integrand <- Vectorize(tmp_integrand)
    tmp_res <- integrate(tmp_integrand, lower=0, upper=upper_limit+1, subdivisions =2000, rel.tol = 0.00001)
    CRPS[loc,t]<- tmp_res$value
  }
}

CRPS_by_loc <- rowMeans(CRPS)
boxplot(CRPS_by_loc,ylim=c(-10,10))

save(CRPS_by_loc, MSPE, file="~/Desktop/GEV-GP_VAE/Figures/CRPS_MSPE_AI.RData")

###### ---------------------------------------------------------------------- ######
###### -------------------------- Empirical chi plots ----------------------- ######
###### ---------------------------------------------------------------------- ######

### -------- Encoder for v_t --------
h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()

### -------- Encoder for v_t_prime --------
h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()

### -------- Activation via Laplace transformation --------
# h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
# h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
# h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
# theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
h_1_prime_row_means <- 2*h_1_prime$mm(ones)/n.t
h_1_prime_to_theta <- h_1_prime_row_means$pow(-2)
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
station_Simulations_All <- matrix(rfrechet(n.s*n.t,shape=1, location = m, scale = tau), nrow=n.s) * as_array((y_star))
ylab = expression(atop(bold("Model II"), chi[u]))

chi_plot(X,stations, station_Simulations_All, distance=0.5, legend = FALSE, ylab = ylab)
ggsave("~/Desktop/GEV-GP_VAE/Figures/AI_qqplot_0_5.png",width = 3.95, height = 3.95)

chi_plot(X,stations, station_Simulations_All, distance=2, legend = FALSE, show.axis.y = FALSE)
ggsave("~/Desktop/GEV-GP_VAE/Figures/AI_qqplot_2.png",width = 3.5, height = 3.95)

chi_plot(X,stations, station_Simulations_All, distance=5, show.axis.y = FALSE)
ggsave("~/Desktop/GEV-GP_VAE/Figures/AI_qqplot_5.png",width = 4.4, height = 3.95)








###### ---------------------------------------------------------------------- ######
###### --------------------------------- ARE -------------------------------- ######
###### ---------------------------------------------------------------------- ######
### -------- Regular grid --------
pred_grid <- expand.grid(x=seq(0.05,9.95, by=0.05), y=seq(0.05,9.95, by=0.05))
eucD_grid <- rdist(pred_grid,as.matrix(knot))

W_grid <- wendland(eucD_grid,r=3)
W_grid <- sweep(W_grid, 1, rowSums(W_grid), FUN="/")
W_grid_tensor <- torch_tensor(W_grid,dtype=torch_float())
W_grid_alpha_tensor <- W_grid_tensor$pow(1/alpha)

n.sim=50
station_Simulations_grid <- array(NA, c(nrow(pred_grid), n.sim*n.t))


###### ---------------  Interpolate b_8 onto holdout locations -------------- ######
library(akima)
b_8_grid <- matrix(NA, nrow=nrow(pred_grid), ncol=n.t)
for(t in 1:n.t){
  tmp <- interp(x=stations[,1], y=stations[,2], z=as_array(b_8[,t]), xo=seq(0.05,9.95, by=0.05), yo=seq(0.05,9.95, by=0.05), linear=FALSE, extrap=TRUE)$z
  b_8_grid[,t] <- as.vector(tmp)
}

b_8_grid_tensor <- torch_tensor(b_8_grid,dtype=torch_float())


### -------- Encoder for v_t --------
h <- w_1$mm(X_tensor)$add(b_1)$relu()
h_1 <- w_2$mm(h)$add(b_2)$relu()
sigma_sq_vec <- w_3$mm(h_1)$add(b_3)$exp()
mu <- w_4$mm(h_1)$add(b_4)$relu()

### -------- Encoder for v_t_prime --------
h_prime <- w_1_prime$mm(X_tensor)$add(b_1_prime)$relu()
h_1_prime <- w_2_prime$mm(h_prime)$add(b_2_prime)$relu()

### -------- Activation via Laplace transformation --------
# h_1_prime_laplace <- h_1_prime$multiply(-0.2)$exp()$mean(dim=2)
# h_1_prime_t <- h_1_prime_laplace$log()$multiply(-1)
# h_1_prime_to_theta <- (0.2-h_1_prime_t$pow(2))$pow(2)$divide(4*h_1_prime_t$pow(2))$view(c(k,1))
# theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))
h_1_prime_row_means <- 2*h_1_prime$mm(ones)/n.t
h_1_prime_to_theta <- h_1_prime_row_means$pow(-2)
theta_propagate <- h_1_prime_to_theta$expand(c(k,n.t))

sigma_sq_vec_prime <- w_3_prime$mm(theta_propagate)$add(b_3_prime)$exp() #w_3_prime$mm(h_1_prime)$add(b_3_prime)$exp()
mu_prime <- w_4_prime$mm(theta_propagate)$add(b_4_prime) #w_4_prime$mm(h_1_prime)$add(b_4_prime)

for(iter in 1:n.sim){
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
  
  y_star <- lrelu(W_grid_alpha_tensor$mm(v_t)$add(b_8_grid_tensor))
  
  ##Decoder
  station_Simulations_grid[, ((iter-1)*n.t+1):(iter*n.t)] <- matrix(rfrechet(nrow(pred_grid)*n.t,shape=1, location = m, scale = tau), nrow=nrow(pred_grid)) * as_array((y_star))
}


U_sim_grid <- array(NA, dim=dim(station_Simulations_grid))
for(loc in 1:nrow(station_Simulations_grid)){
  if(loc %%1000==0) cat("loc=", loc,'\n')
  tmp_copula <- marginal_thetavec(x=station_Simulations_grid[loc,],L=25,theta = theta_sim[,1],
                                  alpha = alpha,k_l=W_grid[loc,])
  U_sim_grid[loc, ] <- tmp_copula
}


u_vec <- c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001))
# Threshs <- quantile(station_Simulations_grid, p = u_vec)
Emp_ARE <- array(NA, c(length(u_vec), 3))
for(iter in 1:length(u_vec)){
  if(iter%%10==0) cat('iter=', iter, '\n')
  u_tmp <- u_vec[iter]
  # thresh <- Threshs[iter]
  where.exceed <- which(U_sim_grid[19801, ]>u_tmp) ## center of the spatial domain
  if(length(where.exceed)>0){
    tmp_AE <- rep(NA, length(where.exceed))
    for(col in 1:length(where.exceed)){
      tmp_AE[col] <- sum(U_sim_grid[,where.exceed[col]]>u_tmp)*0.0025
    }
    
    Emp_ARE[iter,] <- c(mean(tmp_AE), quantile(tmp_AE, p=c(0.025, 0.975)))
  }else{
    Emp_ARE[iter,] <- c(0,0,5.641896)
  }
}


plot(u_vec, sqrt(Emp_ARE[,3]/pi), type='l')
lines(u_vec, sqrt(Emp_ARE[,2]/pi))
lines(u_vec, sqrt(Emp_ARE[,1]/pi))


###-------- Simulate real data --------
col_sim <- ncol(station_Simulations_grid)
Z_grid <- matrix(NA, nrow=k, ncol=col_sim)
X_grid <- matrix(NA, nrow=nrow(pred_grid), ncol=col_sim)
Epsilon_frechet_grid <- matrix(NA, nrow=nrow(pred_grid), ncol=col_sim)
set.seed(12)
for (iter in 1:col_sim) {
  for (i in 1:k) {
    Z_grid[i,iter] <- single_rejection_sampler(theta = theta_sim[i,1])
  }
  Epsilon_frechet_grid[,iter] <-  rfrechet(nrow(pred_grid), shape=1, scale = tau, location = m)
  X_grid[,iter] <-  Epsilon_frechet_grid[,iter]* ((W_grid^(1/alpha))%*%Z_grid[,iter])
}


### -------------- Transform to uniform scales -------------------
U_grid <- array(NA, dim=dim(X_grid))
for(loc in 1:nrow(X_grid)){
  if(loc %%1000==0) cat("loc=", loc,'\n')
  tmp_copula <- marginal_thetavec(x=X_grid[loc,],L=25,theta = theta_sim[,1],
                                  alpha = alpha,k_l=W_grid[loc,])
  U_grid[loc, ] <- tmp_copula
}


u_vec <- c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001))
Emp_ARE_ori <- array(NA, c(length(u_vec), 3))
for(iter in 1:length(u_vec)){
  if(iter%%10==0) cat('iter=', iter, '\n')
  u_tmp <- u_vec[iter]
  where.exceed <- which(U_grid[19801, ]>u_tmp) ## center of the spatial domain
  if(length(where.exceed)>0){
    tmp_AE <- rep(NA, length(where.exceed))
    for(col in 1:length(where.exceed)){
      tmp_AE[col] <- sum(U_grid[,where.exceed[col]]>u_tmp)*0.0025
    }
    
    Emp_ARE_ori[iter,] <- c(mean(tmp_AE), quantile(tmp_AE, p=c(0.025, 0.975)))
  }else{
    Emp_ARE_ori[iter,] <- c(0,0,5.641896)
  }
}

lines(u_vec, sqrt(Emp_ARE_ori[,3]/pi), col='red')
lines(u_vec, sqrt(Emp_ARE_ori[,2]/pi), col='red')
lines(u_vec, sqrt(Emp_ARE_ori[,1]/pi), col='red')


save(u_vec, Emp_ARE_ori, Emp_ARE, file="~/Desktop/GEV-GP_VAE/ARE_AI.RData")

load("~/Desktop/GEV-GP_VAE/ARE_AI.RData")

dat <- data.frame(x = u_vec, truth = sqrt(Emp_ARE_ori[,1]/pi), truth_lower = sqrt(Emp_ARE_ori[,1]/pi)+(sqrt(Emp_ARE_ori[,2]/pi)-sqrt(Emp_ARE_ori[,1]/pi))/2, 
                  truth_upper=sqrt(Emp_ARE_ori[,1]/pi)+(sqrt(Emp_ARE_ori[,3]/pi)-sqrt(Emp_ARE_ori[,1]/pi))/2, emu = sqrt(Emp_ARE[,1]/pi), 
                  emu_lower = sqrt(Emp_ARE[,1]/pi)+(sqrt(Emp_ARE[,2]/pi)-sqrt(Emp_ARE[,1]/pi))/2, 
                  emu_upper=sqrt(Emp_ARE[,1]/pi)+(sqrt(Emp_ARE[,3]/pi)-sqrt(Emp_ARE[,1]/pi))/2)
dat <- rbind(dat, c(1,0.12,0.12,0.12,0.12,0.12,0.12))


plt <- ggplot(dat,aes(x=x,y=truth)) +
  geom_line(aes(color="Truth"),linewidth=1) +
  geom_line(aes(y=emu,color="Emulation"),linewidth=1) +
  scale_color_manual(values=c('red', 'black')) + 
  geom_ribbon(data=dat,aes(ymin=emu_lower,ymax=emu_upper),alpha=0.2,fill="red") +
  geom_ribbon(data=dat,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.2,fill="black") +
  labs(colour="Type") +
  ylab("Average radius of exceedance (ARE)") + xlab("Quantile") + 
  ggtitle("Model II")+
  theme(plot.title = element_text(hjust = 0.5)) + 
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) + 
  force_panelsizes(rows = unit(3.05, "in"),
                   cols = unit(3.05, "in")) + theme(plot.title = element_text(face = 'bold'))

legend <- FALSE; show.axis.y <- FALSE
if(!legend) plt <- plt + guides(color="none")
if(!show.axis.y) plt<- plt + theme( axis.text.y=element_blank(), 
                                    axis.ticks.y=element_blank(),
                                    axis.title.y = element_blank())
plt
ggsave("~/Desktop/GEV-GP_VAE/Figures/ARE_AI.png",width = 3.5, height = 3.95)



