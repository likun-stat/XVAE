setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")
res0 <- rep(NA, 10000)
for(iter in 1:10000){
  res0[iter] <- double_rejection_sampler(theta=0,alpha=1/2)
}
plot(density(log(res0)),ylim=c(0,0.5))

alpha=1/2
U = runif(10000,0,pi)
E_prime = rexp(10000)
X = E_prime/Zolo_A(U)
res = (1/X)^((1-alpha)/alpha)
lines(density(log(res)), lty=2, lwd=2)

## rlevy, rinvgamma and 
res1 <- rmutil::rlevy(10000, m=0, s=1/2)
lines(density(log(res1)), col='red')

res2 <- invgamma::rinvgamma(10000, shape=1/2, scale=4)
lines(density(log(res2)), col='blue')

x <- seq(-5,25, length.out=2000)
density <- rep(NA, 2000)
for(i in 1:length(x)){
  density[i]<- f_H(exp(x[i]), alpha, theta=0,log=FALSE)*exp(x[i])
}
lines(x, density, col='orange', lwd=2)




## ------- theta =/= 0 -----------
res0 <- rep(NA, 100000)
for(iter in 1:100000){
  res0[iter] <- double_rejection_sampler(theta=0.2,alpha=1/2)
}
plot(density(log(res0)),ylim=c(0,0.5))

res0_1 <- rep(NA, 100000)
for(iter in 1:100000){
  res0_1[iter] <- single_rejection_sampler(theta=0.2)
}
lines(density(log(res0_1)),col='red', lwd=2)

x <- seq(-5,25, length.out=2000)
density <- rep(NA, 2000)
for(i in 1:length(x)){
  density[i]<- f_H(exp(x[i]), alpha, theta=0.2,log=FALSE)*exp(x[i])
}
lines(x, density, col='orange', lwd=2)


## ------- theta =/= 0, alpha=/= 1/2 -----------
res0 <- rep(NA, 100000)
for(iter in 1:100000){
  res0[iter] <- double_rejection_sampler(theta=0.2,alpha=0.4)
}
plot(density(log(res0)),ylim=c(0,0.5))

res0_1 <- rep(NA, 100000)
for(iter in 1:100000){
  res0_1[iter] <- single_rejection_sampler_alpha_not_half(theta=0.2, alpha = 0.4)
}
plot(density(log(res0_1)),col='red', lwd=2)

x <- seq(-5,25, length.out=2000)
density <- rep(NA, 2000)
for(i in 1:length(x)){
  density[i]<- f_H(exp(x[i]), alpha=0.4, theta=0.2,log=FALSE)*exp(x[i])
}
lines(x, density, col='orange', lwd=2)

