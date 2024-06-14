## ------------------ Results from XVAE analysis ------------------
## "./XVAE_Red_Sea_SST_results.RData" generated from the R file "./XVAE/Data_analysis_alpha_fixed_half.R"
load("./XVAE_Red_Sea_SST_results.RData")

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
save(theta_t_array,station_Simulations_All, file="~/Desktop/GEV-GP_VAE/redSST_emulation.RData" )





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
     file="~/Desktop/GEV-GP_VAE/redSST_replicates.RData")



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
ggsave("~/Desktop/GEV-GP_VAE/Figures/dat_mon_max_1.png",width = 4.8, height = 4.8)

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
ggsave("~/Desktop/GEV-GP_VAE/Figures/emu_mon_max_1.png",width = 4.8, height = 4.8)

png("~/Desktop/GEV-GP_VAE/Figures/qq_mon_max_1.png",width = 360, height = 380)
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
ggsave("~/Desktop/GEV-GP_VAE/Figures/dat_mon_max_2.png",width = 4.8, height = 4.8)

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
ggsave("~/Desktop/GEV-GP_VAE/Figures/emu_mon_max_2.png",width = 4.8, height = 4.8)

png("~/Desktop/GEV-GP_VAE/Figures/qq_mon_max_2.png",width = 360, height = 380)
par(mar = c(3.5,4,4,2) + 0.1) ## default is c(5,4,4,2) + 0.1
extRemes::qqplot(original_data[,ind], mon_max_emulation_smooth, 
                 ylab='Emulation', main='', regress = FALSE, xlab="")
title(xlab="Observations", line=1.9, cex.lab=1.2)
dev.off()

### ------------------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------------------

