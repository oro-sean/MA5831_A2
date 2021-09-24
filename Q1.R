## set up working environment
rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")

library(quadprog, quietly = TRUE)
library(ggplot2, quietly = TRUE)

## Set up Data Frame

x1 <- c(3,4,3.5,5,4,6,2,-1,3,3,-2,-1) # enter x1 as a vector
x2 <- c(2,0,-1,1,-3,-2,5,7,6.5,7,7,10) # enter x2 as a vector
y <- c(-1,-1,-1,-1,-1,-1,1,1,1,1,1,1) # enter y as a vector

data <- data.frame(y,x1,x2) # combine into data frame
data$y <- as.factor(data$y) # make y a factor

x <- cbind(x1, x2) # create matrix of predictors

n <- length(x1) # set n to # observations

## Create scatter plot using ggplot

scatterPlot <- ggplot(data = data, aes(x2,x1)) + #x2 on horizontal axis, x1 on vertical axis
  geom_point(aes(colour = y), size = 2) + # points colored by y
  scale_color_manual(values = c("-1" = "blue", "1" = "red"))  + # set colors for -1 and 1
  labs(title = "Categories x2 vs x1") + # plot tittle
  theme_minimal()

scatterPlot

## find optimal separating hyperplane


eps <- 1e-8
Q <- sapply(1:n, function(i) y[i]*t(x)[ ,i])
D <- t(Q) %*% Q
d <- matrix(1, nrow = n)
b0 <- rbind( matrix(0, nrow=1, ncol=1) , matrix(0, nrow=n, ncol=1) )
A <- t(rbind(matrix(y, nrow=1, ncol=n), diag(nrow=n)))

sol <- solve.QP(D +eps*diag(n), d, A, b0, meq=1, factorized=FALSE)
qpsol <- matrix(sol$solution, nrow=n)

findLine <- function(a, y, X){
  nonzero <-  abs(a) > 1e-5
  W <- rowSums(sapply(which(nonzero), function(i) a[i]*y[i]*X[i,]))
  b <- mean(sapply(which(nonzero), function(i) X[i,]%*%W- y[i]))
  slope <- -W[2]/W[1]
  intercept <- b/W[1]
  return(c(intercept,slope))
}

qpline <- findLine( qpsol, y, x)


scatterPlot_marg <- scatterPlot + geom_abline(intercept=qpline[1], slope=qpline[2], size=1)
  

scatterPlot_marg
