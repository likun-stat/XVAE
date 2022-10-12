library(dplyr)
library(autodiffr)
library(fields)
library(VGAM)

# devtools::install_github("Non-Contradiction/autodiffr")
# devtools::install_github("Non-Contradiction/JuliaCall")
ad_setup()
# ad_setup("C:/Users/xm3cf/AppData/Local/Programs/Julia-1.7.3/bin")
# ad_setup("C:/Users/Xiaoyu/AppData/Local/Programs/Julia-1.8.2/bin")
# ad_setup("/Applications/Julia-1.7.app/Contents/Resources/julia/bin")

relu <- function(x){
  return(log(1+exp(x)))
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

f_H <- function(x,alpha=0.7,theta=0.02){
  h <- function(u)(alpha/(1-alpha))*x^{-1/(1-alpha)}*Zolo_A(pi*u)*exp(-x^(-alpha/(1-alpha))*Zolo_A(pi*u))
  n = 1e3
  t = (1/n) * (h(1)/2 + sum(h(seq(1,n-1,1)/n)))
  y = (1/(exp(theta^alpha)))*t*exp(-theta*x)
  return(y)
}

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