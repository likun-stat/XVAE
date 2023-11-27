library(torch)

# setwd('~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extCVAE/')
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")

load("./mon_max_trans.RData")
load("./new_loc.RData")
load("./fitted_gev_par.RData")

range(new_loc[,2])
plot_data <- data.frame(lon = new_loc[,1], lat = new_loc[,2], unif = mon_max_trans[,1], 
                        loc = fitted_gev_par$location, scale = fitted_gev_par$scale, shape=fitted_gev_par$shape)
# ggplot(plot_data) + geom_raster(aes(x=lon, y=lat, fill=unif)) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#                                                    color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = "Time 1", na.value = NA, limits=c(0,1)) +
#   xlim(29, 48) + ylim(9, 34)
# 
# ggplot(plot_data) + geom_raster(aes(x=lon, y=lat, fill=loc)) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = expression(mu), na.value = NA) +
#   xlim(29, 48) + ylim(9, 34)
# 
# ggplot(plot_data) + geom_raster(aes(x=lon, y=lat, fill=scale)) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = expression(tau), na.value = NA) +
#   xlim(29, 48) + ylim(9, 34)
# 
# ggplot(plot_data) + geom_raster(aes(x=lon, y=lat, fill=shape)) +
#   geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
#             color = "gray",  size = 0.5) +
#   scale_fill_gradientn(colours = topo.colors(100), name = expression(xi), na.value = NA) +
#   xlim(29, 48) + ylim(9, 34)


###### ---------------------------------------------------------------------- ######
###### ------------------------ Marginal transformation --------------------- ######
###### ---------------------------------------------------------------------- ######
load("./mon_max_allsites.RData")
fitted_gev_par$beta <- fitted_gev_par$location - fitted_gev_par$scale/fitted_gev_par$shape
X <- array(NA, dim(mon_max_allsites))
for(iter in 1:nrow(X)){
  beta_tmp <- fitted_gev_par$beta[iter]
  tau_tmp <- fitted_gev_par$scale[iter]
  xi_tmp <-  fitted_gev_par$shape[iter]
  X[iter, ] <- (tau_tmp/(abs(xi_tmp)*(beta_tmp - mon_max_allsites[iter,])))^{1/abs(xi_tmp)}
}


###### ---------------------------------------------------------------------- ######
###### --------------------------- Data storm centers ----------------------- ######
###### ---------------------------------------------------------------------- ######
if(!exists('stations')) {
  stations <- data.frame(new_loc)
  rm(new_loc)}

wss <- function(k, df) {
  kmeans(df, k, nstart = 10)$tot.withinss
}

threshold <- quantile(X, p=0.99)
data.knots <- data.frame(lon=NA, lat=NA)
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
res <- kmeans(data.knots, 100, nstart = 10)
plot(data.knots, pch=20, col=res$cluster)
points(res$centers, pch='+', col='red')
# plot(res$centers, pch=20)


###### ---------------------------------------------------------------------- ######
###### ------------------------------- Knots -------------------------------- ######
###### ---------------------------------------------------------------------- ######
x1 <- 33.39495; x2<- 43.05982
y1 <- 29.81209; y2<- 13.21023
len <- 33; num.line=12
yseq <- seq(12,30, length.out=len)
sep_by <- seq(-3, 4, length.out=num.line)
knot_candidates <- matrix(NA, ncol=2, nrow=length(yseq)*num.line)

counter <- 1
for(iter_line in 1:num.line){
  for(iter in 1:length(yseq)){
    ytmp <- yseq[iter]
    knot_candidates[counter, ] <- c((ytmp-(y1*x2-y2*x1)/(x2-x1)+sep_by[iter_line])/((y2-y1)/(x2-x1)), ytmp)
    counter <- counter+1
  }
}



library(ggplot2)
plot(map_data("world2")[,1:2], xlim=c(29, 48), ylim=c(9, 34))
# # abline(a=(y1*x2-y2*x1)/(x2-x1), b=(y2-y1)/(x2-x1))
# abline(a=(y1*x2-y2*x1)/(x2-x1)-2.9, b=(y2-y1)/(x2-x1), col='red')
# # abline(a=(y1*x2-y2*x1)/(x2-x1)-2.1, b=(y2-y1)/(x2-x1))
# abline(a=(y1*x2-y2*x1)/(x2-x1)-1.3, b=(y2-y1)/(x2-x1), col='red')
# # abline(a=(y1*x2-y2*x1)/(x2-x1)-0.5, b=(y2-y1)/(x2-x1))
# abline(a=(y1*x2-y2*x1)/(x2-x1)+0.3, b=(y2-y1)/(x2-x1), col='red')
# # abline(a=(y1*x2-y2*x1)/(x2-x1)+1.1, b=(y2-y1)/(x2-x1))
# abline(a=(y1*x2-y2*x1)/(x2-x1)+1.9, b=(y2-y1)/(x2-x1), col='red')
points(knot_candidates, pch=20, col='blue')

# locator()
x <- "32.45594 32.19852 32.27207 32.51111 33.30178 34.31311 35.23250 35.45315 36.62997 37.01611 38.59746 39.18586 41.22690 42.64276 43.47021 43.92990 43.35988 43.01052 43.01052 41.46595 40.71205 39.77427 39.35135 39.44329 38.46874 37.54935 37.40225 36.64835 36.18866 35.58186 35.10378 35.08539 35.34282 34.86474 34.57054 34.38666 34.07407 33.55921 32.91564 32.80531"
y <- "30.19777 29.63211 29.25500 28.79708 27.66575 25.48390 23.94852 22.73638 21.79361 19.12690 17.37603 15.30193 13.98204 12.33892 11.53083 12.36586 14.43996 15.65210 16.94505 19.45014 20.20436 20.93164 21.90135 23.11349 24.21789 24.83742 25.59164 26.69604 27.55800 28.15060 28.52771 28.95870 29.92841 29.90147 29.38968 28.66240 28.23141 28.87789 29.84760 30.22471"
polygon_data <- data.frame(x = as.numeric(strsplit(x, split = " ")[[1]]),
                               y = as.numeric(strsplit(y, split = " ")[[1]]))
polygon_data <- rbind(polygon_data, polygon_data[1,])
ggplot(polygon_data) + geom_path(aes(x=x,y=y)) +
  geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
                                        color = "gray",  size = 0.5)+
  xlim(29, 48) + ylim(9, 34)

library(sp)
which.in.polygon <- which(point.in.polygon(knot_candidates[,1], knot_candidates[,2], polygon_data$x, polygon_data$y)==1)
knots <- knot_candidates[which.in.polygon, ]

### ----- Combine data cluster centers and our knots
knots <- rbind(knots, res$centers)
distances <- fields::rdist(knots)
where.close <- which(distances<0.3 & distances>0, arr.ind = TRUE)
eliminate <- c()
for(tmp.iter in 1:nrow(where.close)){
  tmp_row <- where.close[tmp.iter,]
  if(tmp_row[1]>tmp_row[2]) eliminate <- c(eliminate, tmp_row[2])
}

knots <- knots[-eliminate, ]



###### ---------------------------------------------------------------------- ######
###### ------------------------------ Radius -------------------------------- ######
###### ---------------------------------------------------------------------- ######

r <- 1.2
dat <- cbind(circleFun(unlist(knots[1,]), diameter = r*2, npoints = 100), group=1)
for(iter in 2:nrow(knots)){
  dat <- rbind(dat, cbind(circleFun(unlist(knots[iter,]), diameter = r*2, npoints = 100), group=iter))
}

knots <- data.frame(knots)
names(knots) <- c("lon", "lat")
ggplot(knots) + geom_point(aes(x=lon, y=lat), shape='+', size=6, color='red') +
  geom_path(data=dat, aes(x=x, y=y, group=factor(group))) +
  geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
            color = "gray",  size = 0.5) +
  xlim(29, 48) + ylim(9, 34)

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Setting up ------------------------------ ######
###### ---------------------------------------------------------------------- ######
k = nrow(knots)
n.s <- nrow(stations)
n.t <- ncol(X)

eucD <- rdist(stations,as.matrix(knots))

W <- wendland(eucD,r=r)
dim(W)
W <- sweep(W, 1, rowSums(W), FUN="/")


ggplot(knots) + geom_point(aes(x=lon, y=lat), shape='+', size=6, color='red') +
  geom_path(data=dat, aes(x=x, y=y, group=group)) +
  geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
            color = "gray",  size = 0.5) +
  geom_point(data=stations[which(W[,1]>0.001),], aes(x = lon, y = lat), colour=scales::alpha("blue", 0.1))+
  geom_point(data=stations[which(W[,50]>0.1),], aes(x = lon, y = lat), colour=scales::alpha("green", 0.1))+
  geom_point(data=stations[which(W[,12]>0.001),], aes(x = lon, y = lat), colour=scales::alpha("yellow", 0.1))+
  geom_point(data=stations[which(apply(W,1,function(x) any(is.na(x)))),], aes(x = lon, y = lat), colour="black")+
  xlim(29, 48) + ylim(9, 34)

# save(W, file="~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/weights_redSST.RData")
   


alpha = 0.5; tau <- 0.1; m <- 0.85
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
ggplot(stations) + geom_raster(aes(x=lon, y=lat, fill=log(X[,ind]))) +
  geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
            color = "gray",  size = 0.5) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Original replicate #', ind), na.value = NA, limits=tmp_range) +
  xlim(29, 48) + ylim(9, 34)

ggplot(stations) + geom_raster(aes(x=lon, y=lat, fill=log(Y_approx[,ind]))) +
  geom_path(data = map_data("world2"), aes(x = long, y = lat, group = group),
            color = "gray",  size = 0.5) +
  scale_fill_gradientn(colours = topo.colors(100), name = paste0('Smooth replicate #', ind), na.value = NA, limits=tmp_range) +
  xlim(29, 48) + ylim(9, 34)


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
station_Simulations_All <- matrix(rfrechet(n.s*n.t,shape=1, location = m, scale = tau), nrow=n.s) * as_array((y_star))

theta_sim <- as_array(theta_t)

chi_plot(X,stations, station_Simulations_All, distance=0.5, legend = FALSE, ylab = expression(atop(bold("Red Sea SST monthly maxima"), chi[u])))


theta_t_array <- as_array(theta_t)
save(theta_t_array,station_Simulations_All, file="~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/redSST_emulation.RData" )





###### ---------------------------------------------------------------------- ######
###### ---------------------------   Emulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######

###### ---------  Run decoder to get the simulated weather processes -------- ######
n.sim<-500
station1_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station50_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station55_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station70_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
station372_Simulations <- matrix(NA, nrow=n.s, ncol=n.sim)
Theta_t_estimations <- array(NA, dim = c(nrow(knots), n.t, n.sim))

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
  station372_Simulations[, iter] <- rfrechet(n.s,shape=1, location = m, scale = tau) * as_array((y_star[,372]))
}

theta_t_array <- as_array(theta_t)
save(theta_t_array, Theta_t_estimations, station1_Simulations, station50_Simulations, station55_Simulations, station70_Simulations, station372_Simulations, 
     file="~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/redSST_replicates.RData")



## -------- time 1 --------
ind <- 1
range_t <- c(0, max(X[,ind]))
q25 <- quantile(X[,ind], 0.25)
q75 <- quantile(X[,ind], 0.75)

pal <- RColorBrewer::brewer.pal(9,"PuBu")
plt3 <- spatial_map(stations, var=X[,ind], pal = pal,
                    title = paste0('Observed copula replicate #', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt3
# ggsave("/Users/LikunZhang/Desktop/img1.png",width = 5.5, height = 5)

plt31 <- spatial_map(stations, var=station1_Simulations[,floor(n.sim/2)], pal = pal,
                     title = paste0('Emulated copula replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt31
# ggsave("/Users/LikunZhang/Desktop/img2.png",width = 5.5, height = 5)

plt32 <- spatial_map(stations, var=as_array(y_star[,ind]), pal = pal,
                     title = paste0('Emulated copula smooth replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_t, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt32

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
                 xlim=c(0,7), ylim=c(0,7), regress = FALSE)


### ------------------------------------------------------------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------- time 1 ------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------------------
ind=1
mon_max_tmp <- fitted_gev_par$beta - fitted_gev_par$scale/(abs(fitted_gev_par$shape)*(station1_Simulations[,floor(n.sim/2)])^{abs(fitted_gev_par$shape)})
mon_max_tmp_smooth <- fitted_gev_par$beta - fitted_gev_par$scale/(abs(fitted_gev_par$shape)*(as_array(y_star[,ind]))^{abs(fitted_gev_par$shape)})

range_obs <- c(-3, 7.3)
q25 <- quantile(mon_max_allsites[,ind], 0.25)
q75 <- quantile(mon_max_allsites[,ind], 0.75)
pal <- RColorBrewer::brewer.pal(9,"YlGnBu")
plt3 <- spatial_map(stations, var=mon_max_allsites[,ind], pal = pal,
                    title = paste0('Mon_max observed at time ', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt3
# ggsave("/Users/LikunZhang/Desktop/img1.png",width = 5.5, height = 5)


plt31 <- spatial_map(stations, var=mon_max_tmp, pal = pal,
                     title = paste0('Mon_max emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt31


plt32 <- spatial_map(stations, var=mon_max_tmp_smooth, pal = pal,
                     title = paste0('Mon_max emulated smooth replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt32



## Add in seasonality
load("../original.RData")

range_obs <- range(original_data[,ind])
q25 <- quantile(original_data[,ind], 0.25)
q75 <- quantile(original_data[,ind], 0.75)
pal <- rev(RColorBrewer::brewer.pal(9,"RdYlBu"))

library(ggtext)
ylab <- paste0("<span style='font-size: 14pt'>**1985-01 monthly maxima**</span><br><span style='font-size: 14pt'>Latitude</span>")
plt3 <- spatial_map(stations, var=original_data[,ind], pal = pal,
                    title = paste0('Observations'), legend.name = "°C", show.legend = FALSE,
                    brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE,
                    xlab="Longitude", ylab=ylab, aspect_ratio = 1.17)+
  scale_x_continuous(expand=c(0,0), limits=c(29, 47))+
  scale_y_continuous(expand=c(0,0), limits=c(10, 31))+
        theme(axis.title.y = ggtext::element_markdown(),
              legend.position = "none")
plt3
ggsave("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/dat_mon_max_1.png",width = 4.8, height = 4.8)

# mon_max_emulation <- mon_max_tmp*original_sd[,ind] + original_mean[,ind]
# plt31 <- spatial_map(stations, var=mon_max_emulation, pal = pal,
#                     title = paste0('Mon_max temperature emulated at time ', ind), legend.name = "°C", 
#                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
# plt31

mon_max_emulation_smooth <- mon_max_tmp_smooth*original_sd[,ind] + original_mean[,ind]
plt32 <- spatial_map(stations, var=mon_max_emulation_smooth, pal = pal,
                     title = paste0('Emulation'), legend.name = "°C", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE,
                     xlab="Longitude", ylab="Latitude", aspect_ratio = 1.17)+
  scale_x_continuous(expand=c(0,0), limits=c(29, 47))+
  scale_y_continuous(expand=c(0,0), limits=c(10, 31))+
          theme(axis.title.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())
plt32
ggsave("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/emu_mon_max_1.png",width = 4.8, height = 4.8)

png("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/qq_mon_max_1.png",width = 360, height = 380)
par(mar = c(3.5,4,4,2) + 0.1) ## default is c(5,4,4,2) + 0.1
extRemes::qqplot(original_data[,ind], mon_max_emulation_smooth, 
                 ylab='Emulation', main='', regress = FALSE, xlab="")
title(xlab="Observations", line=1.9, cex.lab=1.2)
dev.off()

### ------------------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------------------





### ------------------------------------------------------------------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------- time 372 -----------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------------------
ind <- 366
mon_max_tmp <- fitted_gev_par$beta - fitted_gev_par$scale/(abs(fitted_gev_par$shape)*(station372_Simulations[,floor(n.sim/2)])^{abs(fitted_gev_par$shape)})
mon_max_tmp_smooth <- fitted_gev_par$beta - fitted_gev_par$scale/(abs(fitted_gev_par$shape)*(as_array(y_star[,ind]))^{abs(fitted_gev_par$shape)})

range_obs <- c(-3, 7.3)
q25 <- quantile(mon_max_allsites[,ind], 0.25)
q75 <- quantile(mon_max_allsites[,ind], 0.75)
pal <- RColorBrewer::brewer.pal(9,"YlGnBu")
plt3 <- spatial_map(stations, var=mon_max_allsites[,ind], pal = pal,
                    title = paste0('Mon_max observed at time ', ind), legend.name = "Observed\n values", 
                    brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt3
# ggsave("/Users/LikunZhang/Desktop/img1.png",width = 5.5, height = 5)


plt31 <- spatial_map(stations, var=mon_max_tmp, pal = pal,
                     title = paste0('Mon_max emulated replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt31


plt32 <- spatial_map(stations, var=mon_max_tmp_smooth, pal = pal,
                     title = paste0('Mon_max emulated smooth replicate #', ind), legend.name = "Emulated\n values", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
plt32



## Add in seasonality
load("../original.RData")

range_obs <- range(original_data[,ind])
q25 <- quantile(original_data[,ind], 0.25)
q75 <- quantile(original_data[,ind], 0.75)
pal <- rev(RColorBrewer::brewer.pal(9,"RdYlBu"))

library(ggtext)
ylab <- paste0("<span style='font-size: 14pt'>**2015-12 monthly maxima**</span><br><span style='font-size: 14pt'>Latitude</span>")
plt3 <- spatial_map(stations, var=original_data[,ind], pal = pal,
                    title = paste0('Observations'), legend.name = "°C", show.legend = FALSE,
                    brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE,
                    xlab="Longitude", ylab=ylab, aspect_ratio = 1.17)+
  scale_x_continuous(expand=c(0,0), limits=c(29, 47))+
  scale_y_continuous(expand=c(0,0), limits=c(10, 31))+
  theme(axis.title.y = ggtext::element_markdown(),
        legend.position = "none")
plt3
ggsave("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/dat_mon_max_2.png",width = 4.8, height = 4.8)

# mon_max_emulation <- mon_max_tmp*original_sd[,ind] + original_mean[,ind]
# plt31 <- spatial_map(stations, var=mon_max_emulation, pal = pal,
#                     title = paste0('Mon_max temperature emulated at time ', ind), legend.name = "°C", 
#                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE)
# plt31

mon_max_emulation_smooth <- mon_max_tmp_smooth*original_sd[,ind] + original_mean[,ind]
plt32 <- spatial_map(stations, var=mon_max_emulation_smooth, pal = pal,
                     title = paste0('Emulation'), legend.name = "°C", 
                     brks.round = 1, tight.brks = TRUE, range=range_obs, q25=q25, q75=q75, pt.size=0.4, raster=TRUE,
                     xlab="Longitude", ylab="Latitude", aspect_ratio = 1.17)+
  scale_x_continuous(expand=c(0,0), limits=c(29, 47))+
  scale_y_continuous(expand=c(0,0), limits=c(10, 31))+
  theme(axis.title.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())
plt32
ggsave("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/emu_mon_max_2.png",width = 4.8, height = 4.8)

png("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/Figures/qq_mon_max_2.png",width = 360, height = 380)
par(mar = c(3.5,4,4,2) + 0.1) ## default is c(5,4,4,2) + 0.1
extRemes::qqplot(original_data[,ind], mon_max_emulation_smooth, 
                 ylab='Emulation', main='', regress = FALSE, xlab="")
title(xlab="Observations", line=1.9, cex.lab=1.2)
dev.off()

### ------------------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------------------

