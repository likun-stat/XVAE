library(dplyr)
library(fields)
library(VGAM)
library(torch)

relu <- function(x){
  return(pmax(0,x))
  # return(log(1+exp(x)))
}
wendland <- function (d,r) {
  if (any(d < 0)) 
    stop("d must be nonnegative")
  return(((r - d)^4 * (4 * d + r)) * (d < r)) # s = 2; k = 1
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
full_cond <- function(v_t, W_alpha, X_t, tau){
  Y_t <- (W_alpha)%*%(v_t)
  if(any(Y_t<0)) return(-1e5)
  return(mean(log(Y_t)) - tau*mean(X_t^(-1)*Y_t))
}

gradient_full_cond <-  function(v_t, W_alpha, X_t, tau){
  res <- rep(NA, ncol(W_alpha))
  Y_t <- (W_alpha)%*%(v_t)
  for (iter in 1:ncol(W_alpha)){
    res[iter] <- sum(W_alpha[,iter]/Y_t) -tau*sum(W_alpha[,iter]/X_t)
  }
  return(res)
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


spatial_map <- function(stations, var=NULL, pal=RColorBrewer::brewer.pal(9,"OrRd"), 
                        title='spatial map', legend.name='val', show.legend=TRUE,
                        xlab='x', ylab='y',
                        brks.round = 2, tight.brks = FALSE,
                        conus_fill = "white", 
                        border.wd = 0.2, pt.size = 3, shp = 16,
                        range=NULL, q25=NULL, q75=NULL){
  require(ggplot2);require(ggh4x)
  if(colnames(stations)[1]!='x') colnames(stations)=c('x', 'y')
  if(is.null(var)) show.legend=FALSE
  if(!show.legend){
    plt0 <- ggplot(stations) +
      geom_point(size = pt.size, shape = shp, aes( x = x, y = y), na.rm = TRUE) +
      ylab('y') + xlab('x') +
      theme(panel.border = element_rect(colour = "black", fill=NA, size=1),
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
  gd <- guides(color = guide_legend( reverse = TRUE, override.aes = list(size=5) ))
  
  plt1 <- ggplot(stations) +
    geom_point(size = pt.size, shape = shp, aes( x = x, y = y, color = color.use ), na.rm = TRUE) +
    ylab(ylab) + xlab(xlab)  + sc + gd +
    theme(panel.border = element_rect(colour = "black", fill=NA, size=1),
          plot.title = element_text(hjust = 0.5, size=14),
          legend.text=element_text(size=12), legend.title=element_text(size=13),
          axis.text=element_text(size=13), axis.title.y=element_text(size=14), 
          axis.title.x=element_text(size=14, margin = margin(t = -4, r = 0, b = 0, l = 0)))+
    ggtitle(title) +
    force_panelsizes(rows = unit(3.75, "in"),
                     cols = unit(3.75, "in"))
  return(plt1)
}


