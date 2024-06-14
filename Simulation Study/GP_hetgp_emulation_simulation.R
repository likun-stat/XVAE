setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Simulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))

n.s <- nrow(stations)
n.t <- 100 # n.t <- 500

eucD <- rdist(stations)
cov <- fields::Matern(eucD, range=0.5, nu=5/2)

holdout <- c(1569, 557, 1685, 287, 614, 1169, 329, 487, 855, 851, 1654, 1522, 858, 1840, 619, 576, 990, 1514, 1760, 1023, 1127, 316, 1075, 733, 1314, 
             650, 129, 811, 955, 282, 1167, 1466, 285, 1944, 1706, 1072, 501, 1740, 511, 1319, 536, 693, 988, 1238, 1832, 737, 1363, 1370, 1699, 1067, 
             1025, 29, 1614, 1942, 838, 1820, 1652, 233, 1317, 573, 369, 451, 86, 1507, 327, 1646, 355, 1843, 812, 1073, 361, 1340, 1266, 1464, 758, 1841, 
             818, 247, 751, 219, 135, 111, 532, 377, 408, 1978, 1589, 912, 467, 1380, 130, 1987, 1089, 1836, 359, 105, 1148, 1101, 1242, 1634)
plot(stations)
points(stations[holdout,], pch=16, col="blue")


X <- rmvnorm(n=n.t, sigma=cov)
X <- t(X)

library(mvtnorm)
X_holdout <- X[holdout,]
X <- X[-holdout,]



library(hetGP)
stations_fit <- stations[-holdout,]
stations_holdout <- stations[holdout,]
# start_time <- Sys.time()
emulation_GP <- matrix(NA, nrow=1900, ncol=100)
pred_GP <- matrix(NA, nrow=100, ncol=100)
for(iter in 1:100){
  cat('iter=',iter,'\n')
  subset <- sample(1900,500)
  fit <- mleHomGP(X=as.matrix(stations_fit[subset, ]), Z=X[subset,iter], covtype = "Matern5_2", lower = rep(0.05, 2),
                  upper = rep(10, 2), settings = list(linkThetas = "none"), maxit = 100)
  # end_time <- Sys.time()
  
  # end_time - start_time ## 5.843 mins
  
  p_org <- predict(x = as.matrix(stations_fit), object = fit)
  emulation_GP[,iter] <- p_org$mean
  
  p_pred <- predict(x = as.matrix(stations_holdout), object = fit)
  pred_GP[,iter] <- p_pred$mean
  
}


save(fit, X, X_holdout, stations, emulation_GP, pred_GP, file="~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/HomGP.RData")




###### ---------------------------------------------------------------------- ######
###### ------------------------------ MSPE & CRPS --------------------------- ######
###### ---------------------------------------------------------------------- ######
load("~/Desktop/GEV-GP_VAE/HomGP.RData")

### -------- MSPE ---------
# MSPE <- colMeans((station_Simulations - X_holdout)^2)
tmp <- (pred_GP - X_holdout)^2
MSPE <- apply(tmp, 2, function(x) mean(x[x<1e5]))
boxplot(MSPE,ylim=c(0,500))


### -------- CRPS ---------
n.t <- 100
CRPS <- matrix(NA, nrow=100, ncol=n.t)
for(loc in 1:100){
  ecdf_tmp <- ecdf(X_holdout[loc, ])
  cat("loc=", loc, "\n")
  for(t in 1:n.t){
    tmp_integrand <- function(z) ecdf_tmp(z) - I(pred_GP[loc,t]<=z) 
    tmp_integrand <- Vectorize(tmp_integrand)
    tmp_res <- integrate(tmp_integrand, lower=0, upper=max(X_holdout[loc, ]), subdivisions =2000, rel.tol = 0.00001)
    CRPS[loc,t]<- tmp_res$value
  }
}

# CRPS_by_loc <- rowMeans(CRPS)
CRPS_by_loc <- apply(CRPS,1, function(x) mean(x))
boxplot(CRPS_by_loc,ylim=c(-15,3))

save(CRPS_by_loc, MSPE, file="~/Desktop/GEV-GP_VAE/Figures/CRPS_MSPE_GP_hetGP.RData")



CRPS_by_loc_hetGP <- CRPS_by_loc; MSPE_hetGP <- MSPE
load("~/Desktop/GEV-GP_VAE/Figures/CRPS_MSPE_GP_true_knots.RData")





