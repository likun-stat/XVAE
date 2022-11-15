setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")
double_rejection_sampler(theta=0,alpha=1/2)

alpha=1/2
U = runif(10000,0,pi)
E_prime = rexp(10000)
X = E_prime/Zolo_A(U)
res = (1/X)^((1-alpha)/alpha)
plot(density(log(res)),ylim=c(0,0.5))


res1 <- rmutil::rlevy(10000, m=0, s=1/2)
lines(density(log(res1)), col='red')

res2 <- invgamma::rinvgamma(10000, shape=1/2, scale=4)
lines(density(log(res2)), col='blue')

x <- seq(-5,25, length.out=2000)
density <- rep(NA, 2000)
for(i in 1:length(x)){
  density[i]<- f_H(exp(x[i]), alpha, theta=0)*exp(x[i])
}
lines(x, density, col='orange', lwd=2)

