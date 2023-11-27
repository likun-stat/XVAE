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

W <- wendland(eucD,r=3)
W <- sweep(W, 1, rowSums(W), FUN="/")
# points(stations[W[,1]>0,], pch=20, col='blue')
# points(stations[W[,25]>0,], pch=20, col='green')
# points(stations[W[,17]>0,], pch=20, col='orange')


theta_sim <- (sin(knot$x/2)*cos(knot$y/2)+1)/50
theta_sim[theta_sim < 0.005] <- 0
theta_sim <- matrix(rep(theta_sim, n.t), ncol=n.t)
# fields::image.plot(c(1,3,5,7,9), c(1,3,5,7,9), matrix(theta_sim[,1],5,5), col=terrain.colors(25))


alpha = 0.5
tau <- 0.1
m <- 0.85

Z <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)
Epsilon_frechet <- matrix(NA, nrow=n.s, ncol=n.t)
set.seed(12)
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



## Time 1
library(hetGP)
start_time <- Sys.time()
fit <- mleHetGP(stations, X[,1], covtype = "Matern5_2", lower = rep(0.05, 2),
                 upper = rep(10, 2), settings = list(linkThetas = "none"), maxit = 1e4)
end_time <- Sys.time()

end_time - start_time ## 59.98871 mins

p_org <- predict(x = stations, object = fit)
p <- predict(x = stations_holdout, object = fit)
p$mean
save(X, stations, p_org, p, file = "~/Desktop/GEV-GP_VAE/Flex_hetgp_emulation.RData")



## -------- All time replicates ---------
emulation_flex <- matrix(NA, nrow=1900, ncol=100)
pred_flex <- matrix(NA, nrow=100, ncol=100)
for(iter in 1:100){
  cat('iter=',iter,'\n')
  subset <- sample(1900,1000)
  fit <- mleHomGP(X=as.matrix(stations[subset, ]), Z=X[subset,iter], covtype = "Matern5_2", lower = rep(0.05, 2),
                  upper = rep(10, 2), settings = list(linkThetas = "none"), maxit = 100)
  # end_time <- Sys.time()
  
  # end_time - start_time ## 5.843 mins
  
  p_org <- predict(x = as.matrix(stations), object = fit)
  emulation_flex[,iter] <- p_org$mean
  
  p_pred <- predict(x = as.matrix(stations_holdout), object = fit)
  pred_flex[,iter] <- p_pred$mean
  
}

save(fit, X, X_holdout, stations, emulation_flex, pred_flex, file="~/Desktop/GEV-GP_VAE/HomFlex.RData")



###### ---------------------------------------------------------------------- ######
###### ------------------------------ MSPE & CRPS --------------------------- ######
###### ---------------------------------------------------------------------- ######
load("~/Desktop/GEV-GP_VAE/HomFlex.RData")

### -------- MSPE ---------
# MSPE <- colMeans((station_Simulations - X_holdout)^2)
tmp <- (pred_flex - X_holdout)^2
MSPE <- apply(tmp, 2, function(x) mean(x))
boxplot(MSPE,ylim=c(0,500))


### -------- CRPS ---------
n.t <- 100
CRPS <- matrix(NA, nrow=100, ncol=n.t)
for(loc in 1:100){
  ecdf_tmp <- ecdf(X_holdout[loc, ])
  cat("loc=", loc, "\n")
  for(t in 1:n.t){
    tmp_integrand <- function(z) (ecdf_tmp(z) - I(pred_flex[loc,t]<=z))^2
    tmp_integrand <- Vectorize(tmp_integrand)
    tmp_res <- integrate(tmp_integrand, lower=0, upper=max(X_holdout[loc, ]), subdivisions =2000, rel.tol = 0.00001)
    CRPS[loc,t]<- tmp_res$value
  }
}

# CRPS_by_loc <- rowMeans(CRPS)
CRPS_by_loc <- apply(CRPS,1, function(x) mean(x))
boxplot(CRPS_by_loc,ylim=c(0,10))

save(CRPS_by_loc, CRPS_by_loc_old, MSPE, file="~/Desktop/GEV-GP_VAE/Figures/CRPS_MSPE_flex_hetGP.RData")



CRPS_by_loc_hetGP <- CRPS_by_loc; MSPE_hetGP <- MSPE
load("~/Desktop/GEV-GP_VAE/Figures/CRPS_MSPE_flex_true_knots.RData")





