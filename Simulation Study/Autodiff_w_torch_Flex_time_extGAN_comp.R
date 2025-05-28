library(torch)

setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Simulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######
set.seed(24)
stations <- expand.grid(x=1:24, y=1:20)
knot <- expand.grid(x=seq(2,23, length.out=6), y=seq(2,19, length.out=4))
# plot(stations)
# points(knot, pch="+", col='red', cex=2)

k = nrow(knot)
n.s <- nrow(stations)
n.t <- 500

eucD <- rdist(stations,as.matrix(knot))

r=6
W <- wendland(eucD,r=6)
W <- sweep(W, 1, rowSums(W), FUN="/")
# points(stations[W[,1]>0,], pch=20, col='blue')
# points(stations[W[,25]>0,], pch=20, col='green')
# points(stations[W[,17]>0,], pch=20, col='orange')


theta_sim <- (sin(knot$x/2)*cos(knot$y/2)+1)/50
theta_sim[theta_sim < 0.005] <- 0
theta_sim <- matrix(rep(theta_sim, n.t), ncol=n.t)
# fields::image.plot(c(1,3,5,7,9), c(1,3,5,7,9), matrix(theta_sim[,1],5,5), col=terrain.colors(25))


alpha = 0.5; tau <- 0.1; m <- 0.85

Z <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)
Epsilon_frechet <- matrix(NA, nrow=n.s, ncol=n.t)
library(VGAM)
for (iter in 1:n.t) {
  for (i in 1:k) {
    Z[i,iter] <- single_rejection_sampler(theta = theta_sim[i,iter])
  }
  Epsilon_frechet[,iter] <-  VGAM::rfrechet(n.s, shape=1, scale = tau, location = m)
  X[,iter] <-  Epsilon_frechet[,iter]* ((W^(1/alpha))%*%Z[,iter])
}

ind <- 1
ggplot(stations) + geom_raster(aes(x=x, y=y, fill=log(X[,ind]))) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA) 

X_sub <- X[!(stations$x==1 | stations$x==24 | stations$y==1 | stations$y==20), ]
write.csv(x = X_sub, file = "../extGAN_comp/flex_data.csv")
write.csv(x = stations, file = "../extGAN_comp/coords_padding.csv")
write.csv(x = stations[!(stations$x==1 | stations$x==24 | stations$y==1 | stations$y==20),], file = "../extGAN_comp/coords.csv")




dat <- cbind(circleFun(unlist(knot[1,]), diameter = r*2, npoints = 100), group=1)
for(iter in 2:nrow(knot)){
  dat <- rbind(dat, cbind(circleFun(unlist(knot[iter,]), diameter = r*2, npoints = 100), group=iter))
}

library(ggh4x)
ggplot(knot) + geom_point(aes(x=x, y=y), shape='+', size=6, color='red') +
  geom_path(data=dat[dat$group==11,], aes(x=x, y=y, group=factor(group))) + 
  scale_x_continuous(expand=c(0,0), limits = c(0,25)) +
  scale_y_continuous(expand=c(0,0), limits = c(0,21)) + 
  ggtitle("Knot locations") +
  theme(plot.title = element_text(hjust = 0.5, face="bold"))+
  force_panelsizes(rows = unit(3.5, "in"),
                   cols = unit(3.8, "in"))
ggsave("../extGAN_comp/knots_extGAN.pdf", height=4.2, width=4.6)



pal <- RColorBrewer::brewer.pal(9, "YlGnBu")
ggplot(knot) + geom_point(aes(x=x, y=y, colour = theta_sim[,1]), size=6) +
  scale_x_continuous(expand=c(0,0), limits = c(0,25)) + scale_color_gradientn(colors = pal, name=expression(gamma[k])) +
  scale_y_continuous(expand=c(0,0), limits = c(0,21)) + geom_point(dat=knot[theta_sim[,1]==0,], aes(x=x,y=y), shape=1, size=6)+
  ggtitle("Model III") +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust = 0.5, face="bold"))+
  force_panelsizes(rows = unit(3.5, "in"),
                   cols = unit(3.8, "in"))
ggsave("../extGAN_comp/theta_sim_extGAN.pdf", height=4.2, width=4.6)

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
###### --------------------------- Data storm centers ----------------------- ######
###### ---------------------------------------------------------------------- ######
wss <- function(k, df) {
  kmeans(df, k, nstart = 10)$tot.withinss
}

threshold <- quantile(X, p=0.96)
data.knots <- data.frame(x=NA, y=NA)
for(iter.t in 1:ncol(X)){
  cat('Clustering for', iter.t, '\n')
  where.exceed <- which(X[,iter.t]>threshold)
  
  if(length(where.exceed)>10){
    # Compute and plot wss for k = 1 to k = 15
    tmp.min<- min(15, length(where.exceed)-1)
    k.values <- 1:tmp.min
    tmp_df <- stations[where.exceed,]
    tmp.obs <- X[where.exceed,iter.t]
    
    # extract wss for 2-15 clusters
    wss_values <-unlist(lapply(k.values, wss, df=tmp_df))
    # plot(k.values, wss_values, type='b')
    n.clusters <- which(wss_values/wss_values[1] < 0.15)[1]
    res <- kmeans(tmp_df, n.clusters, nstart = 10)
    # plot(tmp_df, pch=20, cex=0.5, xlim=c(29, 48), ylim=c(9, 34), main=paste("Time = ", iter.t))
    
    for(tmp.iter in 1:n.clusters){
      where.max <- which.max(tmp.obs[res$cluster==tmp.iter])
      data.knots <- rbind(data.knots, tmp_df[where.max,])
    }
  }
}
data.knots <- data.knots[-1,]
res <- kmeans(data.knots, 30, nstart = 10)
plot(data.knots, pch=20, col=res$cluster)
points(res$centers, pch='+', col='red')
# plot(res$centers, pch=20)


###### ---------------------------------------------------------------------- ######
###### ------------------------------- Knots -------------------------------- ######
###### ---------------------------------------------------------------------- ######

knot_candidates <- as.matrix(expand.grid(x=seq(1,26, length.out=5), y= seq(1,20, length.out=4)))




### ----- Combine data cluster centers and our knots
knots <- rbind(knot_candidates, res$centers)
distances <- fields::rdist(knots)
where.close <- which(distances<0.5 & distances>0, arr.ind = TRUE)
eliminate <- c()
if(nrow(where.close)>0){
  for(tmp.iter in 1:nrow(where.close)){
    tmp_row <- where.close[tmp.iter,]
    if(tmp_row[1]>tmp_row[2]) eliminate <- c(eliminate, tmp_row[2])}
}

if(length(eliminate)>0) knots <- knots[-eliminate, ]


###### ---------------------------------------------------------------------- ######
###### ------------------------------ Radius -------------------------------- ######
###### ---------------------------------------------------------------------- ######

r <- 5
dat <- cbind(circleFun(unlist(knots[1,]), diameter = r*2, npoints = 100), group=1)
for(iter in 2:nrow(knots)){
  dat <- rbind(dat, cbind(circleFun(unlist(knots[iter,]), diameter = r*2, npoints = 100), group=iter))
}





###### ---------------------------------------------------------------------- ######
###### ---------------------------- Setting up ------------------------------ ######
###### ---------------------------------------------------------------------- ######
eucD <- rdist(stations,as.matrix(knots))

W <- wendland(eucD,r=r)
dim(W)
W <- sweep(W, 1, rowSums(W), FUN="/")



# holdout <- c(1569, 557, 1685, 287, 614, 1169, 329, 487, 855, 851, 1654, 1522, 858, 1840, 619, 576, 990, 1514, 1760, 1023, 1127, 316, 1075, 733, 1314, 
#              650, 129, 811, 955, 282, 1167, 1466, 285, 1944, 1706, 1072, 501, 1740, 511, 1319, 536, 693, 988, 1238, 1832, 737, 1363, 1370, 1699, 1067, 
#              1025, 29, 1614, 1942, 838, 1820, 1652, 233, 1317, 573, 369, 451, 86, 1507, 327, 1646, 355, 1843, 812, 1073, 361, 1340, 1266, 1464, 758, 1841, 
#              818, 247, 751, 219, 135, 111, 532, 377, 408, 1978, 1589, 912, 467, 1380, 130, 1987, 1089, 1836, 359, 105, 1148, 1101, 1242, 1634)
# plot(stations)
# points(stations[holdout,], pch=16, col="blue")


# X_holdout <- X[holdout,]
# stations_holdout <- as.matrix(stations[holdout,])
# 
# X <- X[-holdout,]
# stations <- as.matrix(stations[-holdout,])


k = nrow(knots)
n.s <- nrow(stations)
n.t <- ncol(X)
stations <- data.frame(stations)



# W_holdout <- W[holdout, ]
# W <- W[-holdout, ]
# Update because we held out 100 locations
knots <- data.frame(knots)
library(ggplot2)
ggplot(knots) + geom_point(aes(x=x, y=y), shape='+', size=6, color='red') +
  geom_path(data=dat, aes(x=x, y=y, group=group)) +
  geom_point(data=stations[which(W[,1]>0.001),], aes(x = x, y = y), colour=scales::alpha("blue", 0.1))+
  geom_point(data=stations[which(W[,10]>0.1),], aes(x = x, y = y), colour=scales::alpha("green", 0.1))+
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


Y_star <- (W_alpha)%*%(Z_approx)
Y_approx <- Y_star - relu(Y_star-X)
ind <- 3
tmp_range <- range(c(log(Y_approx[,ind])), log(X[,ind]))
ggplot(stations) + geom_raster(aes(x=x, y=y, fill=log(X[,ind]))) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA, limits=tmp_range) 

ggplot(stations) + geom_raster(aes(x=x, y=y, fill=log(Y_approx[,ind]))) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Smooth replicate #', ind), na.value = NA, limits=tmp_range) 






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
Epsilon_frechet_indep <- matrix(VGAM::rfrechet(n.s*n.t, location = m, shape=1, scale = tau), nrow=n.s)
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
learning_rate <- -1e-20
alpha_v <- 0.9
lrelu <- nn_leaky_relu(-0.01)

niter = 10000
n <- 1e3
# only depends on alpha; we are not updating alpha for now.
vec <- Zolo_A(pi*seq(1/2,n-1/2,1)/n, alpha)
Zolo_vec <- torch_tensor(matrix(vec, nrow=1,ncol=n), dtype=torch_float(), requires_grad = FALSE) 
Zolo_vec_double <- torch_tensor(Zolo_vec, dtype = torch_float64(), requires_grad = FALSE)
const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1; const3 <- log(const1)
W_alpha_tensor <- W_tensor$pow(1/alpha)
old_loss <- -Inf

set.seed(86)
# start_time <- Sys.time()
for (t in 1:niter) {
  if(t==1000) { learning_rate <- -5e-19; alpha_v <- 0.9}
  if(t==5000) { learning_rate <- -1e-17; alpha_v <- 0.8}
  if(t==80000) { learning_rate <- -2e-14; alpha_v <- 0.7}
  if(t==90000) { learning_rate <- -1e-13; alpha_v <- 0.7}
  if(t==100000) { learning_rate <- -1e-12; alpha_v <- 0.5}
  if(t==110000) { learning_rate <- -5e-11; alpha_v <- 0.4}
  if(t==120000) { learning_rate <- -1e-9; alpha_v <- 0.4}
  if(t==140000) { learning_rate <- -1e-8; alpha_v <- 0.4}
  if(t==300000) { learning_rate <- -1e-7; alpha_v <- 0.4}
  
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

loss - part3/(n.s*n.t)
plot(theta_sim[,3], as_array(theta_t[,16]), xlab=expression(paste('true ',theta)), ylab=expression(paste('CVAE ',theta)))
abline(a = 0, b = 1, col='orange', lty=2)







###### ---------------------------------------------------------------------- ######
###### -------------------------- Spatial prediction ------------------------ ######
###### ---------------------------------------------------------------------- ######

###### ---------  Run decoder to get the simulated weather processes -------- ######
station_Simulations <- matrix(NA, nrow=n.s, ncol=n.t)


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
station_Simulations <- matrix(VGAM::rfrechet(n.s*n.t,shape=1, location = m, scale = tau), nrow=n.s) * as_array(y_star)

ind <- 3
tmp_range <- range(c(log(station_Simulations[,ind])), log(X[,ind]))
ggplot(stations) + geom_raster(aes(x=x, y=y, fill=log(X[,ind]))) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA, limits=tmp_range) 

ggplot(stations) + geom_raster(aes(x=x, y=y, fill=log(station_Simulations[,ind]))) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Emulated replicate #', ind), na.value = NA, limits=tmp_range) 




### -------- MSPE ---------
# MSPE <- colMeans((station_Simulations - X_holdout)^2)
tmp <- (station_Simulations - X)^2
MSPE <- apply(tmp, 2, function(x) mean(x[x<1e5]))
boxplot(MSPE,ylim=c(0,500))



save(MSPE, station_Simulations,  file="~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/MSPE_flex_custom_knots_XVAE.RData")




###### ---------------------------------------------------------------------- ######
###### ----------------------------   QQ plot ------------------------------ ######
###### ---------------------------------------------------------------------- ######
quantile <- quantile(X, p=0.99)

png(filename="../extGAN_comp/qq_het_flex_custom_knots_XVAE.png",width = 400, height = 380) # width = 500, height = 480
par(mar = c(5.1, 6.1, 4.1, 2.1))
extRemes::qqplot(X, station_Simulations, regress = FALSE,
                 xlab=expression(paste('Simulated ', X[1:n[t]])),
                 ylab=expression(paste('Emulated ', X[1:n[t]])), cex.lab = 1.2,
                 xlim=c(0,80), ylim=c(0,80), main="XVAE")
dev.off()

