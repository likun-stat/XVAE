library(dplyr)
library(fields)
library(VGAM)
library(torch)

relu <- function(x){
  return(pmax(0,x))
  # return(log(1+exp(x)))
}

leaky_relu <- function(x, slope){
  res <- apply(x, 2, function(x) return(pmax(0,x)+slope*pmin(0,x)))
  return(res)
  # return(log(1+exp(x)))
}


# Wendland radial basis function
# Inputs:
#   d: A vector or matrix of distances (must be nonnegative)
#   r: The radius of influence (nonnegative scalar)
# Outputs:
#   A vector or matrix of Wendland function values
# Details:
#   This function evaluates the Wendland radial basis function with parameters s = 2 and k = 1.
#   It is compactly supported, meaning the function is zero for distances greater than r.
wendland <- function(d, r) {
  if (any(d < 0)) 
    stop("d must be nonnegative") # Ensure distances are valid
  
  return((1 - d/r)^2 * (d < r))
}




#' Calculate Total Within-Cluster Sum of Squares (WSS) for k-Means Clustering
#'
#' This function computes the total WSS for a specified number of clusters (`k`) 
#' on a given dataset. It is commonly used to evaluate clustering performance 
#' and select an optimal number of clusters.
#'
#' @param k Integer. The number of clusters for the k-means algorithm.
#' @param df Data frame or matrix. The dataset to be clustered, where rows are observations 
#'           and columns are features.
#' @return Numeric. The total within-cluster sum of squares (WSS) for the specified `k`.
#'
#' @examples
#' wss(3, iris[, -5])

wss <- function(k, df) {
  kmeans(df, k, nstart = 10)$tot.withinss
}




#' Derive Data-Driven Knots for Spatio-Temporal Models
#'
#' This function identifies data-driven knots (spatial locations) based on high values in 
#' the input data matrix and performs clustering to determine knot locations. Knots are 
#' used as the basis for constructing spatial basis functions in subsequent modeling steps.
#'
#' @param X Matrix. Observed data, where rows correspond to spatial sites and columns to time replicates.
#' @param stations Data frame or matrix. Spatial coordinates of observation sites, where rows correspond to sites.
#' @param threshold_p Numeric. The quantile level (e.g., 0.96) used to threshold high values in `X`.
#' @param echo Logical. If `TRUE`, progress messages are printed during execution.
#' @param start.knot.num Integer. Initial number of knots. If `NULL`, it defaults to approximately 
#'                       `5 * log(nrow(stations))`.
#' @return Matrix. A refined set of knot locations, where rows are coordinates.
#'
#' @details
#' 1. High values in `X` exceeding the specified quantile are identified for each time replicate.
#' 2. A clustering algorithm (`k-means`) is applied to the spatial locations of these high values 
#'    to identify clusters, with the number of clusters determined dynamically based on the WSS.
#' 3. The center within each cluster is chosen as a knot.
#' 4. Additional coarse grid knots are generated, and all knot candidates are refined to ensure 
#'    adequate spacing by eliminating close points.
#'
#' @examples
#' # Load example data
#' load("example_X.RData")
#' knots <- data_driven_knots(X, stations, threshold_p = 0.96)
data_driven_knots <- function(X, stations, threshold_p, echo=FALSE, start.knot.num = NULL){
  # Set the default number of knots if not specified
  if (is.null(start.knot.num)) 
    start.knot.num <- round(5 * log(nrow(stations)))
  
  # Define the threshold based on the quantile of X
  threshold <- quantile(X, probs = threshold_p)
  data.knots <- data.frame(x = NA, y = NA)
  
  # Loop through each time replicate
  for (iter.t in 1:ncol(X)) {
    # Identify sites exceeding the threshold
    where.exceed <- which(X[, iter.t] > threshold)
    
    # Only proceed if there are enough exceedances
    if (length(where.exceed) > 10) {
      # Determine the range of cluster counts to test
      tmp.min <- min(15, length(where.exceed) - 1)
      k.values <- 1:tmp.min
      tmp_df <- stations[where.exceed, ]
      tmp.obs <- X[where.exceed, iter.t]
      
      # Compute WSS for potential cluster counts
      wss_values <- unlist(lapply(k.values, wss, df = tmp_df))
      
      # Identify the optimal number of clusters
      n.clusters <- which(wss_values / wss_values[1] < 0.15)[1]
      
      # Perform k-means clustering with the optimal number of clusters
      res <- kmeans(tmp_df, n.clusters, nstart = 10)
      
      # Select the point with the highest value within each cluster as a knot
      for (tmp.iter in 1:n.clusters) {
        where.max <- which.max(tmp.obs[res$cluster == tmp.iter])
        data.knots <- rbind(data.knots, tmp_df[where.max, ])
      }
    }
  }
  
  # Remove the placeholder first row
  data.knots <- data.knots[-1, ]
  
  # Refine knots by clustering them again
  res <- kmeans(data.knots, start.knot.num, nstart = 10)
  
  # Generate additional coarse grid knot candidates
  coarse.grid.length <- round(sqrt(start.knot.num))
  knot_candidates <- as.matrix(expand.grid(
    x = seq(min(stations[, 1]), max(stations[, 1]), length.out = coarse.grid.length),
    y = seq(min(stations[, 2]), max(stations[, 2]), length.out = coarse.grid.length)
  ))
  
  # Combine cluster centers and coarse grid knots
  knots <- rbind(knot_candidates, res$centers)
  rownames(knots) <- NULL
  
  # Calculate pairwise distances and eliminate close knots
  distances <- fields::rdist(knots)
  min.gap <- max(distances) / 30
  where.close <- which(distances < min.gap & distances > 0, arr.ind = TRUE)
  eliminate <- c()
  if (nrow(where.close) > 0) {
    for (tmp.iter in 1:nrow(where.close)) {
      tmp_row <- where.close[tmp.iter, ]
      if (tmp_row[1] > tmp_row[2]) eliminate <- c(eliminate, tmp_row[2])
    }
  }
  
  # Remove eliminated knots
  if (length(eliminate) > 0) 
    knots <- knots[-eliminate, ]
  
  if(echo){
    plot(data.knots, pch=20, col=res$cluster)
    points(knots, pch='+', col='red')
  }
  return(knots)
}





# Function to calculate the radius needed to ensure all locations are covered by at least one knot
# Inputs:
#   knots: A matrix or dataframe of knot coordinates
#   stations: A matrix or dataframe of station coordinates
# Output:
#   radius: The automatically-determined radius such that any station is covered by at least one basis function
calc_radius <- function(knots, stations) {
  # Compute the Euclidean distance between each station and all knots
  eucD <- rdist(stations, as.matrix(knots)) 
  
  # Find the minimum distance from each station to the nearest knot
  nearest_knot_dist <- apply(eucD, 1, min) # Find the minimum distance for each station
  
  #  Determine the radius, ensuring all stations are covered by one or more knots
  radius <- 2 * max(nearest_knot_dist)
  
  return(radius)
}


# Function to visualize knots and the coverage of Wendland basis functions
# Inputs:
#   knots: A dataframe or matrix with knot coordinates (columns 'x' and 'y')
#   stations: A dataframe or matrix with station coordinates (columns 'x' and 'y')
#   r: The radius of influence for the Wendland basis functions
#   W: A matrix of weights from the Wendland basis functions, where rows correspond to stations 
#      and columns correspond to knots
#   select: A vector of indices specifying knots whose coverage will be visualized
# Output:
#   A ggplot object visualizing:
#   - Knots as red '+' markers
#   - Coverage regions as circles
#   - Stations influenced by selected knots in different colors (blue, green, yellow)

visualize_knots <- function(knots, stations, r, W, select = c(1, 12, 10)) {
  # Step 1: Generate circular paths for each knot's coverage
  # Create the first circle based on the first knot
  dat <- cbind(circleFun(unlist(knots[1, ]), diameter = r * 2, npoints = 100), group = 1)
  colnames(knots) <- c('x', 'y')
  
  # Add circles for the remaining knots
  for (iter in 2:nrow(knots)) {
    dat <- rbind(dat, cbind(circleFun(unlist(knots[iter, ]), diameter = r * 2, npoints = 100), group = iter))
  }
  
  # Step 2: Load ggplot2 library for visualization
  library(ggplot2)
  
  # Step 3: Create the plot
  fig <- ggplot(knots) +
    # Plot knots as red '+' symbols
    geom_point(aes(x = x, y = y), shape = '+', size = 6, color = 'red') +
    # Plot circular coverage regions
    geom_path(data = dat, aes(x = x, y = y, group = group)) +
    # Plot stations influenced by the first selected knot in blue
    geom_point(
      data = stations[which(W[, select[1]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("blue", 0.3)
    ) +
    # Plot stations influenced by the second selected knot in green
    geom_point(
      data = stations[which(W[, select[2]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("green", 0.3)
    ) +
    # Plot stations influenced by the third selected knot in yellow
    geom_point(
      data = stations[which(W[, select[3]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("yellow", 0.3)
    ) +
    # Plot stations with no coverage (NA weights) in black
    geom_point(
      data = stations[which(apply(W, 1, function(x) any(is.na(x)))), ],
      aes(x = x, y = y),
      colour = "black"
    )
  
  # Step 4: Return the ggplot object
  fig
}



sinc <- function(x) {
  if (sin(x)==0 & x==0)
    stop("Num&Denom of sinc() are both 0")
  else return(sin(x)/x)
}

Zolo_A <- function(u,alpha=0.7){
  y = ((sin(alpha*u)^alpha*sin(u-u*alpha)^(1-alpha))/(sin(u)))^(1/(1-alpha))
  return(y)
}

f_H <- function(x,alpha=0.7,theta=0.02, n=1e3, log=TRUE){
  if(alpha==0.5){
    alpha=1/2; beta = 1/4 # alpha = shape; beta = rate
    f_x <- alpha*log(beta) -log(gamma(alpha)) - (alpha+1)*log(x) - beta/x
    y = theta^alpha+f_x-theta*x
  } else{
    u <- seq(1/2,n-1/2,1)/n
    tmp <- Zolo_A(pi*u, alpha)
    const <- 1/(1-alpha); const1 <- 1/(1-alpha)-1
    H <- const1*x^{-const}*tmp*exp(-x^(-const1)*tmp)
    y = theta^alpha+log(mean(H))-theta*x
  }
  if(log) return(y) else return(exp(y))
}

# f_H_integrand <- function(u, x){
#   A <- (sin(pi*u/2)/sin(pi*u))^2
#   return(x^{-2}*A*exp(-A/x))
# }
# integrate(f_H_integrand, lower=0, upper=1, x=x)

# According to Devroye (2009), will get smaller sample!
double_rejection_sampler = function(theta=theta,alpha=alpha){
  if(theta!=0){
    ## set up
    gam = theta^alpha * alpha * (1-alpha)
    xi = (2+sqrt(pi/2))*(sqrt(2*gam))/pi + 1
    psi = exp(-gam*(pi^2)/8)*(2+sqrt(pi/2))*sqrt(gam*pi)/pi
    w_1 = xi*sqrt(pi/(2*gam))
    w_2 = 2*psi*sqrt(pi)
    w_3 = xi*pi
    b = (1-alpha)/alpha
    
    # Initial state
    X = 1
    second_condition = 1
    E = 0.5
    while (!(X >= 0 & second_condition <= E)) {
      ## First generate U with density proportional to g**
      # Initial state
      U = 1; Z = 2
      # Start
      while (!(U<pi & Z<=1)) {
        V = runif(1);W_prime = runif(1)
        if(gam >= 1){
          if(V < (w_1/(w_1+w_2))){
            U <- abs(rnorm(1))/sqrt(gam)
          } else {
            U <- pi*(1-W_prime^2)
          }
        } else {
          if(V < (w_3/(w_3+w_2))){
            U = pi*W_prime
          } else {
            U = pi*(1-W_prime^2)
          }
        }
        W = runif(1)
        zeta = sinc(U)/((sinc(alpha*U)^alpha)*(sinc((1-alpha)*U))^(1-alpha))
        zeta = sqrt(zeta)
        phi = (sqrt(gam)+alpha*zeta)^(1/alpha)
        z = phi/(phi-(sqrt(gam))^(1/alpha))
        
        part = xi*exp((-gam*U^2)/2)*ifelse(U>=0 & gam>=1,1,0) + psi/(sqrt(pi-U))*ifelse(U<pi & U>0,1,0) + xi*ifelse(U>=0 & U<=pi & gam<1,1,0)
        num = pi*exp((-theta^alpha)*(1-zeta^(-2)))*part
        denom = (1+sqrt(pi/2))*sqrt(gam)/zeta + z
        pho = num/denom
        Z = W*pho
      }
      
      # Generate X with density proportional to g(x,U)
      
      # set up of constants
      a = Zolo_A(u = U,alpha = alpha)
      m = (b*theta/a)^alpha
      delta = sqrt(m*alpha/a)
      a_1 = delta*sqrt(pi/2)
      a_2 = delta
      a_3 = z/a
      s = a_1+a_2+a_3
      
      V_prime = runif(1)
      N_prime = rnorm(1)
      E_prime = rexp(1)
      if(V_prime < a_1/s){
        X = m-delta*abs(N_prime)
      } else if (V_prime < a_2/s){
        X = runif(1,m,m+delta)
      } else {
        X = m+delta+E_prime*a_3
      }
      E = -log(Z)
      second_condition = a*(X-m) + theta*(X^(-b)-m^(-b)) - (N_prime^2/2)*ifelse(X<m,1,0) - E_prime*ifelse(X>(m+delta),1,0)
    }
    return(1/X^b)
  } else {
    U = runif(1,0,pi)
    E_prime = rexp(1)
    X = E_prime/Zolo_A(U)
    res = (1/X)^((1-alpha)/alpha)
    return(res)
  }
}

# alpha is fixed at 1/2
single_rejection_sampler = function(theta=theta){
  X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
  V <- runif(1)
  while(V> exp(-theta*X)){
    X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
    V <- runif(1)
  }
  return(X)
}

# alpha is not necessarily 1/2
single_rejection_sampler_alpha_not_half = function(theta=theta, alpha){
  gamma <- cos(pi*alpha/2)^{1/alpha}
  X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
  V <- runif(1)
  while(V> exp(-theta*X)){
    X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
    V <- runif(1)
  }
  return(X)
}

# Use functions in FMStable to calculate H density, but using numerical integration implicitly.
H_density <- function(x,alpha=alpha,delta=delta,theta=theta){
  gamma <- ((delta/alpha)*cos(pi*alpha/2))^{1/alpha}
  xtmp <- x/((delta/alpha)^{1/alpha})
  res1 <- FMStable::dEstable(xtmp,FMStable::setParam(alpha=alpha, location=0, logscale=log(gamma),pm="S1"))
  res2 <- res1*exp(-theta*x)/(((delta/alpha)^{1/alpha})*exp((-delta*theta^alpha)/alpha))
  return(res2)
}

# Marginal distribution function to calculate dependence measure. (Eq.10 in Bopp(2020))
marginal_thetavec <- function(x,L=L,theta=theta,alpha=alpha,k_l=k_l){
  y <- sapply(x, function(z) exp(sum(theta^alpha - (theta+(k_l/z)^(1/alpha))^alpha)))
  return(y)
}

#################################################################################
##  --------------------------- v_t initial values   ----------------------------
#################################################################################
full_cond <- function(v_t, W_alpha, X_t, tau, m){
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  if(any(tmp<=0)) return(-1e5)
  return(-mean(log(Y_t)) + 2*mean(log(tmp))- tau*mean(tmp))
}

gradient_full_cond <-  function(v_t, W_alpha, X_t, tau, m){
  res <- rep(NA, ncol(W_alpha))
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  for (iter in 1:ncol(W_alpha)){
    tmp2 <- X_t*W_alpha[,iter]/Y_t^2
    res[iter] <- mean(2*tmp*tmp2 - W_alpha[,iter]/Y_t- tau*tmp^2*tmp2)
  }
  return(res)
}

full_cond_logscale <- function(log_v_t, W_alpha, X_t, tau, m){
  v_t <- exp(log_v_t)
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  if(any(tmp<=0)) return(-1e5)
  return(-mean(log(Y_t)) + 2*mean(log(tmp))- tau*mean(tmp))
}

gradient_full_cond_logscale <-  function(log_v_t, W_alpha, X_t, tau, m){
  v_t <- exp(log_v_t)
  res <- rep(NA, ncol(W_alpha))
  Y_t <- (W_alpha)%*%(v_t)
  tmp <- 1/(X_t/Y_t-m)
  for (iter in 1:ncol(W_alpha)){
    tmp2 <- X_t*W_alpha[,iter]/Y_t^2
    res[iter] <- mean(2*tmp*tmp2 - W_alpha[,iter]/Y_t- tau*tmp^2*tmp2)
  }
  return(res*v_t)
}

# tmp =as_array(v_t[,1])
# res <- optim(par=tmp, full_cond, W_alpha=W_alpha, X_t=X_t, gr=gradient_full_cond, method='CG',
#       control = list(fnscale=-1, trace=1, maxit=20000))


#################################################################################
##  ------------------------ Plot pretty spatial maps ---------------------------
#################################################################################
circleFun <- function(center = c(0,0),diameter = 1, npoints = 100){
  r = diameter / 2
  tt <- seq(0,2*pi,length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

## Great circle distance
circ_dist <- function(x, y){
  x <- x*pi/180
  y <- y*pi/180
  6371*acos(sin(x[2])*sin(y[2])+cos(x[2])*cos(y[2])*cos(x[1]-y[1]))
}

circ_dist_mat <- function(X, Y){
  X <- as.matrix(X); Y <- as.matrix(Y)
  Output <- matrix(NA, nrow=nrow(X), ncol=nrow(Y))
  X <- X*pi/180
  Y <- Y*pi/180
  tmp1_X <- sin(X[,2]); tmp2_X<-cos(X[,2])
  tmp1_Y <- sin(Y[,2]); tmp2_Y<-cos(Y[,2])
  for(i in 1:nrow(X)){
    for(j in 1:nrow(Y)){
      Output[i,j] <- acos(tmp1_X[i]*tmp1_Y[j]+tmp2_X[i]*tmp2_Y[j]*cos(X[i,1]-Y[j,1]))
    }
  }
  return(6371*Output)
}

spatial_map <- function(stations, var=NULL, pal=RColorBrewer::brewer.pal(9,"OrRd"), 
                        title='spatial map', legend.name='val', show.legend=TRUE, show.color=TRUE, show.axis.y = TRUE,
                        xlab='x', ylab='y',
                        brks.round = 2, tight.brks = FALSE,
                        conus_fill = "white", 
                        border.wd = 0.2, pt.size = 3, shp = 16,
                        range=NULL, q25=NULL, q75=NULL, raster=TRUE, aspect_ratio =1){
  require(ggplot2);require(ggh4x)
  if(colnames(stations)[1]!='x') colnames(stations)=c('x', 'y')
  if(is.null(var)) show.color=FALSE
  if(!show.color){
    plt0 <- ggplot(stations) +
      geom_point(size = pt.size, shape = shp, aes( x = x, y = y), na.rm = TRUE) +
      ylab('y') + xlab('x') +
      theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
            plot.title = element_text(hjust = 0.5, size=14),
            legend.text=element_text(size=12), legend.title=element_text(size=13),
            axis.text=element_text(size=13), axis.title.y=element_text(size=14), 
            axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
      ggtitle(title) +
      force_panelsizes(rows = unit(3.75, "in"),
                       cols = unit(3.75, "in"))
    return(plt0)
  }
  
  if(is.null(range)) range =  range(var, na.rm=TRUE)
  nugget = 10^{-brks.round}
  brks = round(seq(range[1]-nugget, range[2]+nugget, length.out = length(pal)+1), brks.round)
  if(tight.brks) {
    if(is.null(q25)) q25 = quantile(var, p=0.25)
    if(is.null(q75)) q75 = quantile(var, p=0.75)
    brks = round(c(range[1]-nugget, seq(q25, q75, length.out = length(pal)-1), range[2]+nugget), brks.round)
  }
  color.use <- cut(var, brks, include.lowest = TRUE)
  col.labels <- c(paste("<", brks[2]), levels(color.use)[2:(length(brks)-2)],
                  paste(">", brks[(length(brks)-1)]))
  
  sc <- scale_color_manual( values = pal, name = legend.name, drop = FALSE, na.translate=FALSE,
                            labels = col.labels ) 
  if(show.legend) gd <- guides(color = guide_legend( reverse = TRUE, override.aes = list(size=5) )) else gd <- guides(color = "none")
  
  if(!raster) {
    if(show.axis.y){
      plt1 <- ggplot(stations) +
        geom_point(size = pt.size, shape = shp, aes( x = x, y = y, color = color.use ), na.rm = TRUE) +
        ylab(ylab) + xlab(xlab)  + sc + gd +
        theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
              plot.title = element_text(hjust = 0.5, size=14),
              legend.text=element_text(size=12), legend.title=element_text(size=13),
              axis.text=element_text(size=13), axis.title.y=element_text(size=14), 
              axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
        ggtitle(title) +
        force_panelsizes(rows = unit(3.75, "in"),
                         cols = unit(3.75, "in"))}else{
                           plt1 <- ggplot(stations) +
                             geom_point(size = pt.size, shape = shp, aes( x = x, y = y, color = color.use ), na.rm = TRUE) +
                             ylab(ylab) + xlab(xlab)  + sc + gd +
                             theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
                                   plot.title = element_text(hjust = 0.5, size=14),
                                   legend.text=element_text(size=12), legend.title=element_text(size=13),
                                   axis.text=element_text(size=13),
                                   axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)),
                                   axis.text.y=element_blank(), 
                                   axis.ticks.y=element_blank(),
                                   axis.title.y = element_blank())+
                             ggtitle(title) +
                             force_panelsizes(rows = unit(3.75, "in"),
                                              cols = unit(3.75, "in"))
                         }
  }else{
    plt1 <- ggplot(stations) +
      geom_raster(aes( x = x, y = y, fill = var), na.rm = TRUE) +
      ylab(ylab) + xlab(xlab)  + scale_fill_gradientn(name=legend.name, colors = pal, limits = range) + gd +
      theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
            plot.title = element_text(hjust = 0.5, size=14),
            legend.text=element_text(size=12), legend.title=element_text(size=13),
            axis.text=element_text(size=13), 
            axis.title=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
      ggtitle(title) +
      force_panelsizes(rows = unit(3.75, "in"),
                       cols = unit(3.75/aspect_ratio, "in"))
  }
  return(plt1)
}





library(ggh4x)
chi_plot <- function(X, stations, emulation, distance, tol=0.001,
                     u_vec=c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001)),
                     L=25, ylab=expression(chi[u]), uniform=FALSE, legend = TRUE, show.axis.y = TRUE, duplicated = TRUE){
  require(fields)
  require(ggplot2)
  d <- distance
  upper <- d+tol
  lower <- d-tol
  
  spatial_loc <- matrix(NA,nrow = nrow(X),ncol = 6)
  for (i in 1:(nrow(X)-1)) {
    tmp_dist <- rdist(matrix(stations[i,],nrow=1),stations[(i+1):nrow(X),])
    cand <- which(tmp_dist < upper 
                  & tmp_dist > lower)
    if(length(cand)!=0) spatial_loc[i,1:6] <- cand[1:6]
  }
  
  pairs <- matrix(NA,ncol = 2,nrow = length(spatial_loc))
  for (i in 1:nrow(spatial_loc)) {
    for (j in 1:ncol(spatial_loc)) {
      pairs[(i-1)*ncol(spatial_loc)+j,1] <- i
      pairs[(i-1)*ncol(spatial_loc)+j,2] <- i+spatial_loc[i,j]
    }
  }
  pairs <- data.frame(na.omit(pairs))
  if(!duplicated)  {
    pairs <- pairs[which(!duplicated(pairs[,1]))+1,]
    wh.sub <- floor(seq(1, nrow(pairs), length.out=10))
    pairs <- pairs[wh.sub,] 
    pairs <- data.frame(na.omit(pairs))}
  
  k1 <- ncol(X)
  k2 <- ncol(emulation)
  
  sim_pairs <- matrix(NA,nrow = ncol(X)*nrow(pairs),ncol = 4)
  emu_pairs <- matrix(NA,nrow = ncol(emulation)*nrow(pairs),ncol = 4)
  
  for (i in 1:nrow(pairs)) {
    sim_pairs[((i-1)*k1+1):(i*k1),1] <- X[pairs[i,1],]
    sim_pairs[((i-1)*k1+1):(i*k1),2] <- marginal_thetavec(x=X[pairs[i,1],],L=L,theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,1],])
    if(uniform) {
      sim_pairs[((i-1)*k1+1):(i*k1),2] <- X[pairs[i,1],]
    }
    sim_pairs[((i-1)*k1+1):(i*k1),3] <- X[pairs[i,2],]
    sim_pairs[((i-1)*k1+1):(i*k1),4] <- marginal_thetavec(x=X[pairs[i,2],],L=L,theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,2],])
    if(uniform) {
      sim_pairs[((i-1)*k1+1):(i*k1),4] <- X[pairs[i,2],]
    }
    
    emu_pairs[((i-1)*k2+1):(i*k2),1] <- emulation[pairs[i,1],]
    emu_pairs[((i-1)*k2+1):(i*k2),2] <- marginal_thetavec(x=emulation[pairs[i,1],],L=L,
                                                          theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,1],])
    if(uniform) {
      emu_pairs[((i-1)*k1+1):(i*k1),2] <- emulation[pairs[i,1],]
    }
    
    emu_pairs[((i-1)*k2+1):(i*k2),3] <- emulation[pairs[i,2],]
    emu_pairs[((i-1)*k2+1):(i*k2),4] <- marginal_thetavec(x=emulation[pairs[i,2],],L=L,
                                                          theta = theta_sim[,1],
                                                          alpha = alpha,k_l=W[pairs[i,2],])
    if(uniform) {
      emu_pairs[((i-1)*k1+1):(i*k1),4] <- emulation[pairs[i,2],]
    }
  }
  
  sim_pairs <- sim_pairs[,c(2,4)]
  emu_pairs <- emu_pairs[,c(2,4)]
  
  Min_sim <- apply(sim_pairs, 1, min)
  all_sim <- as.vector(sim_pairs)
  Min_emu <- apply(emu_pairs, 1, min)
  all_emu <- as.vector(emu_pairs)
  
  
  EmpIntv_sim <- matrix(NA, nrow = length(u_vec), ncol=3)
  EmpIntv_emu <- matrix(NA, nrow = length(u_vec), ncol=3)
  
  for(i in 1:length(u_vec)){
    p_tmp1_sim <- mean(Min_sim>u_vec[i])
    p_tmp2_sim <- mean(all_sim>u_vec[i])
    if(p_tmp1_sim==0|p_tmp2_sim==0){
      EmpIntv_sim[i,]<-c(-2,2,0)
    } else{
      var_sim <- (1/p_tmp1_sim-1)/length(Min_sim) + (1/p_tmp2_sim-1)/length(all_sim)
      EmpIntv_sim[i,]<-c(exp(log(p_tmp1_sim/p_tmp2_sim)) - qnorm(0.975)*sqrt(var_sim),
                         exp(log(p_tmp1_sim/p_tmp2_sim)) - qnorm(0.025)*sqrt(var_sim), p_tmp1_sim/p_tmp2_sim)
    }
    
    p_tmp1_emu <- mean(Min_emu>u_vec[i])
    p_tmp2_emu <- mean(all_emu>u_vec[i])
    if(p_tmp1_emu==0|p_tmp2_emu==0){
      EmpIntv_emu[i,]<-c(-2,2,0)
    } else{
      var_emu <- (1/p_tmp1_emu-1)/length(Min_emu) + (1/p_tmp2_emu-1)/length(all_emu)
      EmpIntv_emu[i,]<-c(exp(log(p_tmp1_emu/p_tmp2_emu)) - qnorm(0.975)*sqrt(var_emu),
                         exp(log(p_tmp1_emu/p_tmp2_emu)) - qnorm(0.025)*sqrt(var_emu), p_tmp1_emu/p_tmp2_emu)
    }
  }
  dat <- data.frame(x=u_vec,truth=EmpIntv_sim[,3],truth_upper=EmpIntv_sim[,2],truth_lower=EmpIntv_sim[,1],
                    emu=EmpIntv_emu[,3],emu_upper=EmpIntv_emu[,2],emu_lower=EmpIntv_emu[,1])
  dat[dat>1] <- 1
  dat[dat<0] <- 0
  plt <- ggplot(dat,aes(x=x,y=truth)) +
    geom_line(aes(color="Truth"),linewidth=1) +
    geom_line(aes(y=emu,color="Emulation"),linewidth=1) +
    scale_color_manual(values=c('red', 'black')) + 
    geom_ribbon(data=dat,aes(ymin=emu_lower,ymax=emu_upper),alpha=0.2,fill="red") +
    geom_ribbon(data=dat,aes(ymin=truth_lower,ymax=truth_upper),alpha=0.4,fill="black") +
    labs(colour="Type") +
    ylab(ylab) + xlab("Quantile") + 
    ggtitle(paste("Distance in (", lower, ", ", upper, ")", sep=''))+
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0,1)) + 
    force_panelsizes(rows = unit(3.05, "in"),
                     cols = unit(3.05, "in"))
  
  if(!legend) plt <- plt + guides(color="none")
  if(!show.axis.y) plt<- plt + theme( axis.text.y=element_blank(), 
                                      axis.ticks.y=element_blank(),
                                      axis.title.y = element_blank())
  
  return(plt)
}

