setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")

###### ---------------------------------------------------------------------- ######
###### ---------------------------- Simulation ------------------------------ ######
###### ---------------------------------------------------------------------- ######
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))

n.s <- nrow(stations)
n.t <- 100 # n.t <- 500

eucD <- rdist(stations)
cov <- matern(eucD, phi=3, kappa=5/2)

holdout <- c(1569, 557, 1685, 287, 614, 1169, 329, 487, 855, 851, 1654, 1522, 858, 1840, 619, 576, 990, 1514, 1760, 1023, 1127, 316, 1075, 733, 1314, 
             650, 129, 811, 955, 282, 1167, 1466, 285, 1944, 1706, 1072, 501, 1740, 511, 1319, 536, 693, 988, 1238, 1832, 737, 1363, 1370, 1699, 1067, 
             1025, 29, 1614, 1942, 838, 1820, 1652, 233, 1317, 573, 369, 451, 86, 1507, 327, 1646, 355, 1843, 812, 1073, 361, 1340, 1266, 1464, 758, 1841, 
             818, 247, 751, 219, 135, 111, 532, 377, 408, 1978, 1589, 912, 467, 1380, 130, 1987, 1089, 1836, 359, 105, 1148, 1101, 1242, 1634)
plot(stations)
points(stations[holdout,], pch=16, col="blue")

Cov <- cov[-holdout, -holdout]
library(mvtnorm)

X <- rmvnorm(n=n.t, sigma=Cov)

library(HetGP)
start_time <- Sys.time()
fit <- mleHomGP(stations, X[,1], covtype = "Matern5_2", lower = rep(0.05, 2),
                upper = rep(10, 2), settings = list(linkThetas = "none"), maxit = 1e4)
end_time <- Sys.time()

end_time - start_time ## 59.98871 mins
