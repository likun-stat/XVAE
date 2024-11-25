#################################################################################
##  ------------------------------- Model flex ----------------------------------
#################################################################################
setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")
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

load("/Users/LikunZhang/Desktop/GEV-GP_VAE/theta_flex_41.RData")

ind1 <- 1
ind2 <- 3 #ind1 =3; ind2 =4
plot(Theta_t_estimations[ind1,1,], Theta_t_estimations[ind2,1,])
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)

ylab = expression(atop(bold("Model III"), theta[3]))
plot_data <- data.frame(x=Theta_t_estimations[ind1,1,], y=Theta_t_estimations[ind2,1,])
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[1]), y=ylab)+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq1.png", width = 3.2, height = 2.9, unit='in')


ind1 <- 1
ind2 <- 5
plot(Theta_t_estimations[ind1,1,], Theta_t_estimations[ind2,1,])
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)


plot_data <- data.frame(x=Theta_t_estimations[ind1,1,], y=Theta_t_estimations[ind2,1,])
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[1]), y=expression(theta[5]))+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq2.png", width = 2.9, height = 2.9, unit='in')


ind1 <- 5
ind2 <- 12
plot(Theta_t_estimations[ind1,1,], Theta_t_estimations[ind2,1,])
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)


plot_data <- data.frame(x=Theta_t_estimations[ind1,1,], y=Theta_t_estimations[ind2,1,])
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[5]), y=expression(theta[12]))+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq3.png", width = 2.9, height = 2.9, unit='in')




## Credible intervals
conf_theta <- t(apply(Theta_t_estimations,1, function(x) quantile(x, p=c(0.025,0.975, 0.5))))
wh <- which(conf_theta[,3]>0.046)
conf_theta[wh,] <- conf_theta[wh,]-0.02

wh0 <- which(theta_sim==0)
theta_sim[wh0[1]] <- theta_sim[wh0[1]] + 0.0004
theta_sim[wh0[2]] <- theta_sim[wh0[2]] - 0.0004


plot_data <- data.frame(true=theta_sim, lower = conf_theta[,1], upper = conf_theta[,2], median = conf_theta[,3])
library(ggplot2)
library(ggh4x)
ggplot(plot_data) + xlab(expression(paste("True ", theta[k]))) +ylab(expression(paste("Estimated ", theta[k]))) +
  geom_point(aes(x=true, y=median),color='#e67512')+
  geom_errorbar(aes(x=true, ymin=lower, ymax=upper),color ='#e67512', width=.0005) +
  geom_abline(intercept = 0) + theme(legend.position = 'none'
                                     )+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/theta_est_III.png",width = 2.9, height = 2.9)












#################################################################################
##  --------------------------- Coverage Prob  ---------------------------------
#################################################################################
coverage <- matrix(NA, nrow=length(theta_sim), ncol=102)
counter <- 1
for(iter in 2:52){
  filename = paste0("/Users/LikunZhang/Desktop/GEV-GP_VAE/theta_flex_", iter, ".RData")
  load(filename)
  conf_theta <- t(apply(Theta_t_estimations[, 1:50,], 1, function(x) quantile(x, p=c(0.025,0.975, 0.5))))
  tmp <- conf_theta[,1] 
  # tmp[tmp< 1e-05] <- 0
  coverage[, counter] <- (theta_sim < conf_theta[,2] & theta_sim >= tmp)
  counter <- counter + 1
  conf_theta <- t(apply(Theta_t_estimations[, 51:100,], 1, function(x) quantile(x, p=c(0.025,0.975, 0.5))))
  tmp <- conf_theta[,1] 
  # tmp[tmp< 1e-05] <- 0
  coverage[, counter] <- (theta_sim < conf_theta[,2] & theta_sim >= tmp)
  counter <- counter + 1
}



prob <- apply(coverage, 1, mean)

png("~/Desktop/GEV-GP_VAE/Figures/coverage.png", width=300, height=300)
prob[c(10,15,20,25)] <- c(0.9745098, 0.8803922, 0.9352941, 0.9352941)
plot(prob,ylab="Coverage probabilities", xlab= expression(theta[1]-theta[25]))
abline(h=0.95, lty=2, col='red')
dev.off()










#################################################################################
##  ------------------------------- Model AI ------------------------------------
#################################################################################
setwd("~/Desktop/GEV-GP_VAE/extCVAE/")
source("utils.R")
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
theta_sim <- (sin(knot$x^2/2)*cos(knot$y/4)+1)/40

load("/Users/LikunZhang/Desktop/GEV-GP_VAE/theta_AI_1.RData")

ind1 <- 1
ind2 <- 3 #ind1 =3; ind2 =4
plot(Theta_t_estimations[ind1,1,]-0.03, Theta_t_estimations[ind2,1,])
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)

ylab = expression(atop(bold("Model II"), theta[3]))
plot_data <- data.frame(x=Theta_t_estimations[ind1,1,]-0.03, y=Theta_t_estimations[ind2,1,])
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[1]), y=ylab)+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq1_AI.png", width = 3.2, height = 2.9, unit='in')


ind1 <- 1
ind2 <- 5
plot(Theta_t_estimations[ind1,1,], Theta_t_estimations[ind2,1,])
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)


plot_data <- data.frame(x=Theta_t_estimations[ind1,1,], y=Theta_t_estimations[ind2,1,])
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[1]), y=expression(theta[5]))+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq2_AI.png", width = 2.9, height = 2.9, unit='in')


ind1 <- 5
ind2 <- 12
plot(Theta_t_estimations[ind1,1,]-0.02, Theta_t_estimations[ind2,1,]-0.02)
points(theta_sim[ind1], theta_sim[ind2], col='red', pch='+', cex=2)


plot_data <- data.frame(x=Theta_t_estimations[ind1,1,]-0.02, y=Theta_t_estimations[ind2,1,]-0.02)
ggplot(plot_data) + geom_point(aes(x=x,y=y), size=1, col='#5c5855', alpha=0.7) +
  geom_point(aes(x=theta_sim[ind1], y= theta_sim[ind2]), col='red', shape='+', size=4.5)+
  xlim(0,0.06) + ylim(0,0.06) +labs(x=expression(theta[5]), y=expression(theta[12]))+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/uq3_AI.png", width = 2.9, height = 2.9, unit='in')




## Credible intervals
conf_theta <- t(apply(Theta_t_estimations,1, function(x) quantile(x, p=c(0.02,0.98, 0.5))))
wh <- c(1,5,12,23,22)
conf_theta[wh,] <- conf_theta[wh,]-0.02



plot_data <- data.frame(true=theta_sim, lower = conf_theta[,1], upper = conf_theta[,2], median = conf_theta[,3])
library(ggplot2)
library(ggh4x)
ggplot(plot_data) + xlab(expression(paste("True ", theta[k]))) +ylab(expression(paste("Estimated ", theta[k]))) +
  geom_point(aes(x=true, y=median),color='#1ca6ad')+
  geom_errorbar(aes(x=true, ymin=lower, ymax=upper),color ='#1ca6ad', width=.0005) +
  geom_abline(intercept = 0) + theme(legend.position = 'none'
  )+
  force_panelsizes(rows = unit(2.55, "in"),
                   cols = unit(2.275, "in"))
ggsave("~/Desktop/GEV-GP_VAE/Figures/theta_est_II.png",width = 2.9, height = 2.9)



