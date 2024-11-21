load("~/Downloads/time1.RData")

# For X
n.s <- nrow(X); n.t <- ncol(X)
set.seed(123)
stations <- data.frame(x=runif(2000, 0, 10), y=runif(2000, 0, 10))



dist <- fields::rdist(stations)
plot(stations)
low <- 3.499; high <-3.501 #low <- 0.999; high <-1.001
sum(low<dist&dist<high)*n.t/2 #number of pairs



# (1) Indices of the pairs
Pairs<-matrix(NA, nrow = sum(low<dist&dist<high)/2, ncol=3)
count<-1
for(i in 1:n.s){
  for(j in 1:i){
    if(dist[i,j]>low&dist[i,j]<high){
      Pairs[count,1:2]<-c(i,j)
      Pairs[count,3] <- TRUE
      if(count>1){
        which<-which(Pairs[(1:(count-1)),3]==1)
        if(any(i == Pairs[which,1:2])|any(j == Pairs[which,1:2])) Pairs[count,3] <- FALSE 
      }
      count <- count + 1}
  }
}
# Pairs<-Pairs[which(Pairs[,3]==1), 1:2]
Pairs<-Pairs[, 1:2]

for(i in 1:nrow(Pairs)){
  lines(c(stations[Pairs[i,1],1], stations[Pairs[i,2],1]),c(stations[Pairs[i,1],2], stations[Pairs[i,2],2]), col='red')
}

Singles <- (1:n.s)[-unique(as.vector(Pairs))]
for(i in 1:length(Singles)){
  points(stations[Singles[i],1], stations[Singles[i],2], col='blue',pch=20)
}


X_pairs <- matrix(NA, nrow = nrow(Pairs)*n.t, ncol=2)
X_single <- rep(NA, length(Singles)*n.t)
count <- 1
for(j in 1:n.t){
  for(i in 1:nrow(Pairs)){
    X_pairs[count,] <- c(X[Pairs[i,1],j],X[Pairs[i,2],j])
    count <- count +1 
  }
}

count <- 1
for(j in 1:n.t){
  for(i in 1:length(Singles)){
    X_single[count]<-X[Singles[i],j]
    count <- count +1 
  }
}


# (3) Approx the joint probability
Min <- apply(X_pairs, 1, min)
U_nonpara <- seq(0.94,0.995,length=25)
Q <- quantile(X_single,p=U_nonpara)

library(binom); library(extRemes)
EmpIntv <- matrix(NA, nrow = length(U_nonpara), ncol=3)
EmpIntv_bar <- matrix(NA, nrow = length(U_nonpara), ncol=3)
for(i in 1:length(Q)){
  p_tmp1 <- mean(Min>Q[i])
  p_tmp2 <- mean(X_single>Q[i])
  var <- (1/p_tmp1-1)/length(Min) + (1/p_tmp2-1)/length(X_single)
  EmpIntv[i,]<-c(exp(log(p_tmp1/p_tmp2) - qnorm(0.975)*sqrt(var)), 
                 exp(log(p_tmp1/p_tmp2) - qnorm(0.025)*sqrt(var)), p_tmp1/p_tmp2)
  # tmp<-taildep(X_pairs[,1], X_pairs[,2], u=U_nonpara[i], type='chibar')
  # exp(2*log(1-U_nonpara[i])/(tmp+1))*length(Min)
  binomconf <- binom.confint(x = sum(Min>Q[i]), n=length(Min), method='exact')
  EmpIntv_bar[i,]<-c(2*log(1-U_nonpara[i])/log(binomconf$lower)-1, 2*log(1-U_nonpara[i])/log(binomconf$upper)-1,
                     2*log(1-U_nonpara[i])/log(p_tmp1)-1)
}


plot(U_nonpara,EmpIntv[,3],type='l', ylim=c(0,1), col='blue', 
     lwd=2, ylab=expression(xi), xlab='u')
polygon(c(U_nonpara, U_nonpara[length(U_nonpara):1]), 
        c(EmpIntv[,1], EmpIntv[length(U_nonpara):1,2]), 
        col=scales::alpha('blue', 0.5), border=NA)

lines(U_nonpara,EmpIntv[,3],type='l', ylim=c(0,1), col='red', lwd=2)
polygon(c(U_nonpara, U_nonpara[length(U_nonpara):1]), 
        c(EmpIntv[,1], EmpIntv[length(U_nonpara):1,2]), 
        col=scales::alpha('red', 0.3), border=NA)

legend('topright', legend='h=3.5')



# For X_emulated
X_emulated <- station1_Simulations
n.s <- nrow(X_emulated); n.t <- ncol(X_emulated)

sum(low<dist&dist<high)*n.t/2 #number of pairs


X_emulated_pairs <- matrix(NA, nrow = nrow(Pairs)*n.t, ncol=2)
X_emulated_single <- rep(NA, length(Singles)*n.t)
count <- 1
for(j in 1:n.t){
  for(i in 1:nrow(Pairs)){
    X_emulated_pairs[count,] <- c(X_emulated[Pairs[i,1],j],X_emulated[Pairs[i,2],j])
    count <- count +1 
  }
}

count <- 1
for(j in 1:n.t){
  for(i in 1:length(Singles)){
    X_emulated_single[count]<-X_emulated[Singles[i],j]
    count <- count +1 
  }
}


# (3) Approx the joint probability
Min <- apply(X_emulated_pairs, 1, min)
U_nonpara <- seq(0.94,0.999,length=25)
Q <- quantile(X_emulated_single,p=U_nonpara)

library(binom); library(extRemes)
EmpIntv <- matrix(NA, nrow = length(U_nonpara), ncol=3)
EmpIntv_bar <- matrix(NA, nrow = length(U_nonpara), ncol=3)
for(i in 1:length(Q)){
  p_tmp1 <- mean(Min>Q[i])
  p_tmp2 <- mean(X_emulated_single>Q[i])
  var <- (1/p_tmp1-1)/length(Min) + (1/p_tmp2-1)/length(X_emulated_single)
  EmpIntv[i,]<-c(exp(log(p_tmp1/p_tmp2) - qnorm(0.975)*sqrt(var)), 
                 exp(log(p_tmp1/p_tmp2) - qnorm(0.025)*sqrt(var)), p_tmp1/p_tmp2)
  # tmp<-taildep(X_emulated_pairs[,1], X_emulated_pairs[,2], u=U_nonpara[i], type='chibar')
  # exp(2*log(1-U_nonpara[i])/(tmp+1))*length(Min)
  binomconf <- binom.confint(x = sum(Min>Q[i]), n=length(Min), method='exact')
  EmpIntv_bar[i,]<-c(2*log(1-U_nonpara[i])/log(binomconf$lower)-1, 2*log(1-U_nonpara[i])/log(binomconf$upper)-1,
                     2*log(1-U_nonpara[i])/log(p_tmp1)-1)
}


lines(U_nonpara,EmpIntv[,2], col='blue')
lines(U_nonpara,EmpIntv[,1], col='blue')
lines(U_nonpara,EmpIntv[,3], col='blue')

