#####

load("DATA.RData")
load("est.RData")
library(stringr)
library(gifski)
library(ggpubr)

## Remove seasonality with the est.mean and est.sd 
resi <- (data-est.mean)/est.sd
save(resi,file = "resi.RData")
# setwd("transformed_value_plots")

# get_gev_par <- function(x){
#   xts.ts <- xts(x,time)
#   monthly_max <- apply.monthly(xts.ts, max)
#   gev_est <- extRemes::fevd(monthly_max,type = "GEV",method = "MLE")
#   gev_est_par <- gev_est$results$par
#   res_tmp <- extRemes::pevd(monthly_max,loc = gev_est_par[1],scale = gev_est_par[2],shape = gev_est_par[3])
#   return(res_tmp)
# }
# mon_max_trans <- apply(resi, 2, get_gev_par)

# mon_max_trans <- matrix(NA,nrow = ncol(resi),ncol = 372)
# 
# ## this loop takes a while
# ## loop over sites
# for (i in 1:nrow(mon_max_trans)) {
#   xts.ts <- xts(resi[,i],time)
#   ## Get monthly maxima
#   monthly_max <- apply.monthly(xts.ts, max) 
#   ## Fit GEV distribution with monthly maxima and get the fitted parameters
#   gev_est <- extRemes::fevd(monthly_max,type = "GEV",method = "MLE")
#   gev_est_par <- gev_est$results$par
#   ## Calculate the cdf based the three parameters above and save them 
#   res_tmp <- extRemes::pevd(monthly_max,loc = gev_est_par[1],scale = gev_est_par[2],shape = gev_est_par[3])
#   mon_max_trans[i,] <- res_tmp
#   # if(i%%100==0) cat("row: ",i,"\n")
# }
# 
# # save(mon_max_trans,file = "mon_max_trans.RData")
# 
# ## Get the monthly time
# mon_time <- seq(as.Date("1985/1/1"), by = "month", length.out = 372)
# mon_time <- format(mon_time,format="%Y-%m")
# 
# ## Plots
# p1 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,87])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[87])
# p2 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,91])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[91])
# p3 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,94])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[94])
# ggarrange(p1, p2, p3,
#           ncol = 3, nrow = 1)
# 
# ggsave("plot1.png",width = 10.28, height = 9.33)
# 
# p1 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,39])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[39])
# p2 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,175])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[175])
# p3 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,298])) + 
#   coord_fixed(expand = FALSE) +
#   scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#   xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[298])
# ggarrange(p1, p2, p3,
#           ncol = 3, nrow = 1)
# 
# ggsave("plot2.png",width = 10.28, height = 9.33)


# ### gif
# total_t <- ncol(mon_max_trans)
# 
# files <- str_c("img", str_pad(1:total_t, 3, "left", "0"), ".png")
# 
# for(i in 1:total_t){
#   ggplot() + theme_bw() + 
#     geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = mon_max_trans[,i])) + 
#     coord_fixed(expand = TRUE) +
#     scale_fill_viridis_c(name="Value",limits = c(0, 1)) +
#     xlab("Longitude") + ylab("Latitude") + ggtitle(mon_time[i])
#   ggsave(files[i], width = 8.28, height = 7.33, type = "cairo")
# }
# gifski(files, "transformed.gif", width = 800, height = 700, loop = FALSE, delay = 0.8)


# #### Spatial map of parameters/p-values
# library(EnvStats)
# gof_gev <- gofTest(monthly_max,distribution = "gev",test = "chisq")
# gof_gev$distribution.parameters
# 
# sp_par_mat <- matrix(NA,nrow = ncol(resi),ncol = 6)
# 
# error_i <- c()
# 
# for (i in 1:nrow(sp_par_mat)) {
#   xts.ts <- xts(resi[,i],time)
#   ## Get monthly maxima
#   monthly_max <- apply.monthly(xts.ts, max) 
#   t <- try(gof_gev <- gofTest(monthly_max,distribution = "gev",test = "chisq"),silent=TRUE)
#   if("try-error" %in% class(t)) error_i <- c(error_i,i)
#   sp_par_mat[i,1:3] <- gof_gev$distribution.parameters
#   sp_par_mat[i,4] <- gof_gev$p.value
#   t <- try(gof_gev <- gofTest(monthly_max,distribution = "gev",test = "ks"),silent=TRUE)
#   sp_par_mat[i,5] <- gof_gev$p.value
#   t <- try(gof_gev <- gofTest(monthly_max,distribution = "gev",test = "ad"),silent=TRUE)
#   sp_par_mat[i,6] <- gof_gev$p.value
#   if(i%%100==0) cat("Row: ",i,"\n")
# }
# 
# for (i in error_i) {
#   xts.ts <- xts(resi[,i],time)
#   ## Get monthly maxima
#   monthly_max <- apply.monthly(xts.ts, max) 
#   test <- egevd(as.numeric(monthly_max),method = "pwme")$parameters
#   par.ls <- list(location=test[1],scale=test[2],shape=test[3])
#   
#   gof_gev <- gofTest(monthly_max,distribution = "gev",test = "chisq",
#                      param.list = par.ls, estimate.params = FALSE)
#   sp_par_mat[i,1:3] <- gof_gev$distribution.parameters
#   sp_par_mat[i,4] <- gof_gev$p.value
#   gof_gev <- gofTest(monthly_max,distribution = "gev",test = "ks",
#                      param.list = par.ls, estimate.params = FALSE)
#   sp_par_mat[i,5] <- gof_gev$p.value
#   gof_gev <- gofTest(monthly_max,distribution = "gev",test = "ad")
#   sp_par_mat[i,6] <- gof_gev$p.value
#   if(i%%100==0) cat("Row: ",i,"\n")
# }
# 
# 
# 
# 
# sum(sp_par_mat[,4] < 0.05)/16703
# sum(sp_par_mat[,5] < 0.05)/16703
# sum(sp_par_mat[,6] < 0.05)/16703
# hist(sp_par_mat[,4])
# 
# hist(sp_par_mat[,5])
# hist(sp_par_mat[,6])
# 
# which(sp_par_mat[,4] < 0.05)[1]
# sp_par_mat[13,4]
# 
# 
# p1 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = sp_par_mat[,4])) + 
#   coord_fixed(expand = TRUE) +
#   scale_fill_viridis_c(name="Value") +
#   xlab("Longitude") + ylab("Latitude") + ggtitle("P-value from Chi-squared test")
# p1
# p2 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = sp_par_mat[,5])) + 
#   coord_fixed(expand = TRUE) +
#   scale_fill_viridis_c(name="Value") +
#   xlab("Longitude") + ylab("Latitude") + ggtitle("P-value from K-S test")
# p2
# p3 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = sp_par_mat[,1])) + 
#   coord_fixed(expand = TRUE) +
#   scale_fill_viridis_c(name="Value") +
#   xlab("Longitude") + ylab("Latitude") + ggtitle("Location parameters")
# p3
# p4 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = sp_par_mat[,2])) + 
#   coord_fixed(expand = TRUE) +
#   scale_fill_viridis_c(name="Value") +
#   xlab("Longitude") + ylab("Latitude") + ggtitle("Scale parameters")
# p4
# p5 <- ggplot() + theme_bw() + 
#   geom_raster(aes(x = new_loc[ , 1], y = new_loc[ , 2], fill = sp_par_mat[,3])) + 
#   coord_fixed(expand = TRUE) +
#   scale_fill_viridis_c(name="Value") +
#   xlab("Longitude") + ylab("Latitude") + ggtitle("Shape parameters")
# p5


# ## Get monthly maxima
# monthly_max <- apply.monthly(xts.ts, max) 
# t <- try(gof_gev <- gofTest(monthly_max,distribution = "pt",test = "chisq"),silent=TRUE)
# if("try-error" %in% class(t)) error_i <- c(error_i,i)
# sp_par_mat[i,1:3] <- gof_gev$distribution.parameters
# sp_par_mat[i,4] <- gof_gev$p.value
# t <- try(gof_gev <- gofTest(monthly_max,distribution = "t",test = "ks"),silent=TRUE)
# sp_par_mat[i,5] <- gof_gev$p.value
# 
# plot(as.numeric(monthly_max))
# 
# library(MASS)
# MASS::fitdistr(monthly_max,"t")
# tmp <- rt(372,df=9.829,ncp = 0.9566)
# qqplot(tmp*0.862,as.numeric(monthly_max))
# extRemes::fevd(monthly_max,type = "GEV",method = "MLE")
# tmp <- extRemes::revd(372,loc = 0.6012,scale = 0.9353,shape = -0.1997,type = "GEV")
# qqplot(tmp,as.numeric(monthly_max))

# find a test that weight tail more

#### Get mon_max_allsites ####
library(xts)
load("resi.RData")
load("est.RData")
load("DATA.Rdata")

mon_max_allsites <- matrix(NA,ncol = 372,nrow = ncol(resi))
for (i in 1:ncol(resi)) {
  xts.ts <- xts(resi[,i],time)
  ## Get monthly maxima
  monthly_max <- apply.monthly(xts.ts, max)
  
  mon_max_allsites[i,] <-  monthly_max
  if(i%%10==0) cat("Row: ",i,"\n")
}

original_mean <- matrix(NA,ncol = 372,nrow = ncol(resi))
original_sd <- matrix(NA,ncol = 372,nrow = ncol(resi))
original_data <- matrix(NA,ncol = 372,nrow = ncol(resi))
for (i in 1:ncol(resi)) {
  xts.ts <- xts(resi[,i],time)
  ## Get monthly maxima
  monthly_max <- apply.monthly(xts.ts, max)
  ind <- which(resi[,i] %in% as.numeric(monthly_max[,1]))
  original_mean[i,] <-  est.mean[ind,i]
  original_sd[i,] <-  est.sd[ind,i]
  original_data[i,] <-  data[ind,i]
  
  if(i%%10==0) cat("Row: ",i,"\n")
}

save(original_data,original_mean,original_sd,file = "original.RData")

#### The marginal of transformed varible does not match the marginal distr in our model
#### So transform it one more time with the upper bound of GEV

rs_data_mon <- matrix(NA,nrow = nrow(mon_max_allsites),ncol = ncol(mon_max_allsites))
for (i in 1:nrow(rs_data_mon)) {
  beta <- fitted_gev_par[i,1]-fitted_gev_par[i,2]/fitted_gev_par[i,3]
  rs_data_mon[i,] <- (abs(fitted_gev_par[i,3])*(beta - mon_max_allsites[i,])/fitted_gev_par[i,2])^(1/fitted_gev_par[i,3])
  if(i%%100==0) cat("Row: ",i,"\n")
}

hist(rs_data_mon[,1])
save(rs_data_mon,file = "rs_data_mon.RData")


