# load data
df <- read.table("housing.data", header = FALSE)
colnames(df) <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv")

# generate regression model
model <- lm(medv ~ ., data=df)
summary(model)
plot(model)

# functions for filtering list by values greater than paramater and sort in descending order
filt_gt <- function(X, C) 
{
  ifelse(X>C, TRUE, FALSE)
}

filt_gt_and_sort_desc <- function(X, C)
{
  HX = X[filt_gt(X, C)]
  SX = HX[order(-HX)]
}

# highest standardized residuals
rd <- abs(rstandard(model))
srd <- filt_gt_and_sort_desc(rd, 3)
srd

# highest leverage
hv <- hatvalues(model)
shv <- filt_gt_and_sort_desc(hv, .1)
shv

# highest cooks
cd <- cooks.distance(model)
scd <- filt_gt_and_sort_desc(cd, 4/length(cd))
scd

# remove obvious outliers
trimmed_oo <- df[-c(369, 372, 373, 365), ]
model_trimmed_oo <- lm(medv ~ ., data=trimmed_oo)
plot(model_trimmed_oo)

# boxcox
library(MASS)

bc <- boxcox(model_trimmed)

lambda <- bc$x[which.max(bc$y)]
lambda

# transform using optimal lambda
transformed <-transform(trimmed, medv = (medv^(lambda) - 1)/lambda)
model_transformed <- lm(medv ~ ., data=transformed)
rdt <- abs(rstandard(model_transformed))
srdt <- filt_gt_and_sort_desc(rdt, 3)
srdt

plot(model_transformed)

# fitted house price against true house price (transformed)
plot(fitted(model_transformed), transformed$medv, xlab="Fitted House Price (transformed)", ylab="Actual House Price (transformed)")
abline(a=0, b=1)