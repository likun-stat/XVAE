knots <- expand.grid(x=seq(1,20, by=1), y=seq(1,20, by=1))
plot(knots, xlim=c(0,21), ylim=c(0,21))

Theta_surf <- function(mei, tau=1){
  center <- mei*c(0,20)+(1-mei)*c(20,0)
  dist_sq <- ((knots[,1]-center[1])^2 + (knots[,2]-center[2])^2)/800
  return(exp(dist_sq/tau)-1)
}

library(ggplot2)
cols <- rev(RColorBrewer::brewer.pal(n=9, "RdBu"))
ggplot(knots) + geom_raster(aes(x=x, y=y, fill=Theta_surf(0.3))) +
  scale_fill_gradientn(colours = cols)

ggplot(knots) + geom_raster(aes(x=x, y=y, fill=Theta_surf(0.5))) +
  scale_fill_gradientn(colours = cols)


ggplot(knots) + geom_raster(aes(x=x, y=y, fill=Theta_surf(1))) +
  scale_fill_gradientn(colours = cols)

## Generate data sets
MEIs <- seq(0,1, length.out = 350)
dat <- matrix(NA, nrow = nrow(knots), ncol = length(MEIs))
Thetas <- matrix(NA, nrow = nrow(knots), ncol = length(MEIs))
source("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extCVAE/utils.R")
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/")


for (iter in 1:length(MEIs)) {
  Thetas_tmp <- Theta_surf(MEIs[iter])
  for (i in 1:nrow(knots)) {
    dat[i,iter] <- single_rejection_sampler(theta = Thetas_tmp[i])
  }
  Thetas[, iter] <- Thetas_tmp
}

iter <- 13
ggplot(knots) + geom_raster(aes(x=x, y=y, fill=Theta_surf(MEIs[iter]))) +
  scale_fill_gradientn(colours = cols)

ggplot(knots) + geom_raster(aes(x=x, y=y, fill=log(dat[, iter]))) +
     scale_fill_gradientn(colours = cols)


# ggplot(knots) + geom_raster(aes(x=x, y=y, fill=log(rowMeans(dat[, 1:10])))) +
  # scale_fill_gradientn(colours = cols)


avg <- matrix(NA, nrow=sqrt(nrow(knots))-4, ncol=sqrt(nrow(knots))-4)
tmp_mat <- matrix(dat[, iter], nrow=sqrt(nrow(knots)), ncol=sqrt(nrow(knots)))

for(row in 1:(sqrt(nrow(knots))-4)){
  for(col in 1:(sqrt(nrow(knots))-4)){
    avg[row, col] <- mean(tmp_mat[row:(row+4), col:(col+4)])
  }
}

knots_sub <- expand.grid(x=seq(3,18, by=1), y=seq(3,18, by=1))
ggplot(knots_sub) + geom_raster(aes(x=x, y=y, fill=log(as.vector(avg)))) +
  scale_fill_gradientn(colours = cols)

training <- sample(1:length(MEIs), 250)
test <- (1:length(MEIs))[-training]
valid <- training[201:250]
training <- training[1:200]
dat_train <- rbind(MEIs[training], dat[,training])
dat_valid <- rbind(MEIs[valid], dat[,valid])
dat_test <- rbind(MEIs[test], dat[,test])
Thetas_train <- Thetas[,training]
Thetas_valid <- Thetas[,valid]
Thetas_test <- Thetas[,test]


write.csv(dat_train, file="train_MEIs_PS_vars.csv", row.names = FALSE)
write.csv(dat_valid, file="valid_MEIs_PS_vars.csv", row.names = FALSE)
write.csv(dat_test, file="test_MEIs_PS_vars.csv", row.names = FALSE)
write.csv(Thetas_train, file="train_Thetas_PS_vars.csv", row.names = FALSE)
write.csv(Thetas_valid, file="valid_Thetas_PS_vars.csv", row.names = FALSE)
write.csv(Thetas_test, file="test_Thetas_PS_vars.csv", row.names = FALSE)
write.csv(Thetas, file="thetas.csv", row.names = FALSE)


