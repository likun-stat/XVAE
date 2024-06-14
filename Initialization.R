setwd("./XCVAE/")
source("utils.R")

##############################################################################
## ----- Use Red Sea SST data to demonstrate the initialization of XVAE ------
##############################################################################
## Load the data set
load("./Red Sea SST data/mon_max_trans.RData")
load("./Red Sea SST data/new_loc.RData")
load("./Red Sea SST data/fitted_gev_par.RData")

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
load("./mon_max_allsites.RData") # GEV margins (see manuscript for justification)
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
## Use K-means to locate knot centers
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
###### ---------------------------- Extra knots ----------------------------- ######
###### ---------------------------------------------------------------------- ######
## Add a grid of knots to ensure minimal knot density
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

save(knots, file="./knots.RData")


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

# save(W, file="~/Desktop/GEV-GP_VAE/weights_redSST.RData")



alpha = 0.5; tau <- 0.1; m <- 0.85
###### ---------------------------------------------------------------------- ######
###### ----------------------- Initial guess -------------------------- ######
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


save.image("./initial_values.RData")
