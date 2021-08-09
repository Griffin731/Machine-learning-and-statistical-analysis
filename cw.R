setwd("/Users/xuehanyin/coursework/Math501/cw")

tele <- read.table("/Users/xuehanyin/coursework/Math501/cw/churndata.txt")

head(tele)


#Machine Learning Part (a)∗∗: Present the data visually using box-and-whisker plots 
#with a distinction for churn. Comment on the data in the context of the problem.

library(ggplot2)
library(ggpubr)

upload_churn <- ggplot(tele, aes(y = upload, x = churn, col = churn))+
  geom_boxplot() +
  labs(title = "The Impact of Internet Upload Speed",
       x = "Does customer switch to a different provider?",
       y = "Average internet upload speed") +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "point",
               shape = 20,
               size = 3,
               show.legend = FALSE) +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "text",
               show.legend = FALSE,
               vjust = -0.7,
               aes(label = round(..y.., digits = 2))) +
  theme(legend.position = "none")

#????
# The customers with faster internet upload speed are tended to switch to differnt providers


webget_churn <- ggplot(tele, aes(y = webget, x = churn, col = churn))+
  geom_boxplot() +
  labs(title = "The Impact of The Webpage Loading Time",
       x = "Does customer switch to a different provider?",
       y = "Average time to load a webpage") +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "point",
               shape = 20,
               size = 3,
               show.legend = FALSE) +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "text",
               show.legend = FALSE,
               vjust = -0.7,
               aes(label = round(..y.., digits = 2)))+
  theme(legend.position = "none")

# The customers with longer webpage loading time are tended to switch to differnt providers


enqcount_churn <- ggplot(tele, aes(y = enqcount, x = churn, col = churn))+
  geom_boxplot() +
  labs(title = "The Impact of Times of Calls From Customers",
       x = "Does customer switch to a different provider?",
       y = "Average times a customer call the company") +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "point",
               shape = 20,
               size = 3,
               show.legend = FALSE) +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "text",
               show.legend = FALSE,
               vjust = -0.7,
               aes(label = round(..y.., digits = 2)))+
  theme(legend.position = "none")
# The customers who call the company more times are tended to switch provider


callwait_churn <- ggplot(tele, aes(y = callwait, x = churn, col = churn))+
  geom_boxplot() +
  labs(title = "The Impact of Waiting time for customer service",
       x = "Does customer switch to a different provider?",
       y = "Average times a customer waits") +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "point",
               shape = 20,
               size = 3,
               show.legend = FALSE) +
  stat_summary(fun = mean,
               color = "darkblue",
               geom = "text",
               show.legend = FALSE,
               vjust = -0.7,
               aes(label = round(..y.., digits = 2)))+
  theme(legend.position = "none")
# The customers who are waiting longer to talk to a customer service operator are tended to swith to a differnt provider

ggarrange(upload_churn, webget_churn,
          nrow = 2, ncol = 1)

ggarrange(enqcount_churn, callwait_churn,
          nrow = 1, ncol = 2)

#---------------------------------------------------------------------------
#Machine Learning Part (b)∗: Create a training set consisting of 350 randomly 
#chosen data points and a test set consisting of the remaining 150 data points.

str(tele)

library(dplyr)
# Make the the column churn factors
tele_1 <- tele %>% mutate(churn_1 = factor(tele$churn, 
                                           levels = c("no","yes")))
str(tele_1)

#set class
cl <- tele_1$churn_1

#predictor
train <- data.frame(cbind(upload = tele_1$upload, webget =tele_1$webget, 
               enqcount = tele_1$enqcount, callwait = tele_1$callwait))

train <- scale(train)
str(train)

set.seed(1)
train_split <- sample(500,350)

#train set predictors and class
train_pre <- train[train_split, ]
train_cl <- cl[train_split]

#test set predictors and class
test_pre <- train[-train_split,]
test_cl <- cl[-train_split]


#---------------------------------------------------------------------------
#Machine Learning Part (c)∗∗∗: Using the training data set apply the K nearest neighbours 
#method to construct a classifier to predict churn based on the four available predictors.
#Find the optimal K using leave-one-out cross-validation for the training data set.
#Calculate the test error for the classification rule obtained for the optimal K.

library(class)

# use leave one out validation to find the best k
set.seed(2)
test.error <- function(k){
  knn.k <- knn(train = train_pre, test = test_pre, cl = train_cl, k=k)
  tab <- table(knn.k, test_cl)
  error <- (tab[1,2] + tab[2,1]) / sum(tab)
  return(error)
}

set.seed(3)
errors <- rep(0,20)
for (i in 1:20) errors[i] <- test.error(k=i)

plot(errors, xlab = "k", ylab = "test errors")
# The best number for k is 11

test.error(k=1)
# test error is 0.08

test.error(k=3)
# test error is 0.1


#---------------------------------------------------------------------------
#ignore this part
# Using Leave one out cross validation to test the train set

set.seed(4)
test.error.cv <- function(k){
  
  knn.cvv <- knn.cv(train = train_pre, train_cl, k=k)
  tab <- table(knn.cvv, train_cl)
  error.cv <- (tab[1,2] + tab[2,1]) / sum(tab)
  return(error.cv)
  
}

errors.cv <- rep(0,30)
for (i in 1:30) errors.cv[i] <- test.error.cv(k=i)

plot(errors.cv, xlab = "k", ylab = "test errors")

test.error.cv(k = 11)
# The best k is 3 with test error 0.1057143

#---------------------------------------------------------------------------

#Machine Learning Part (d)∗∗: Using the training data set apply the random forest (bagging) 
#method to construct a classifier to predict churn based on the four available predictors. 
#Using the obtained random forest, comment on the importance of the four variables for predicting churn.
#Calculate the test error for the obtained random forest. Compare it to the test error found 
#for the KNN classifier and provide an appropriate comment.

library(tree)

df.tree <- tele_1[c(1,2,3,4,6)]

set.seed(3)
# train set 
train.tree <- sample(nrow(df.tree), 350) 

# test set 
test.tree <- df.tree[-train.tree,]
# class for test set
test.cl.tree <- df.tree$churn_1[-train.tree]

churn.tree = tree(churn_1 ~ ., data = df.tree, subset = train_split)
summary(churn.tree)

plot(churn.tree)
text(churn.tree, pretty = 0)

tree.pre = predict(churn.tree, test.tree, type = "class")
tab <- table(tree.pre, test.cl.tree)
tab
(tab[1,2] + tab[2,1]) / sum(tab)
# the test error is 0.03333333

set.seed(4)
# cross validation for classification tree
churn.tree.cv <- cv.tree(churn.tree, FUN = prune.misclass)
churn.tree.cv

plot(churn.tree.cv$size, churn.tree.cv$dev, xlab = "tree size", 
     ylab = "error rate",type = "b")
#find the best tree size is 5

# purne the tree with size 5
tree.prune <- prune.misclass(churn.tree, best = 5)

plot(tree.prune)
text(tree.prune, pretty = 0)
# Customers are tended to switch the provider if:
# The webpag loading time is longer than 601.55
# The customer contacts the company more than 6.5 times
# Waiting for the customer service operator for more than 13.015
# uploading speed is more than 15.3

#predict the pruned tree with the test set
tree.purne.pre <- predict(tree.prune, test.tree, type = "class")
tab <- table(tree.purne.pre, test.cl.tree)

#test error of using tree is 0.02666667
test.error.tree <- (tab[1,2] + tab[2,1]) / sum(tab)
test.error.tree
#0.02666667
#test error of using KNN is 0.1066667 which is higher than using the tree


#---------------------------------------------------------------------------
# above is classification tree 
# here is random forest haha
library(randomForest)

set.seed(5)
rf.tree <- randomForest(churn_1 ~ ., data = df.tree, subset = train.tree,
                         mtry = 2)

varImpPlot(rf.tree)

rf.pre <- predict(rf.tree, test.tree)
rf.tab <- table(rf.pre, test.cl.tree)
rf.tab

rf.error <- (rf.tab[1,2]+rf.tab[2,1]) / sum(rf.tab)
rf.error
# the error is 0.02666667 when mtry=2
#test error of using KNN is 0.1066667 which is higher than using the tree

#---------------------------------------------------------------------------
# Machine Learning Part (e)∗∗: Using the entire data set (training set and test set 
# com- bined), perform Principal Component Analysis for the four variables: 
# upload, webget, enqcount and callwait. Comment on the results.
# Using principal components, create the “best” two dimensional view of the data set. 
# In this visualisation, use colour coding to indiciate the churn. 
# How much of the variation or information in the data is preserved in this plot?
# Provide an interpretation of the first two principal components.

#library(GGally)
#ggpairs(tele)

pca <- princomp(train, cor = TRUE)
summary(pca)

#Summary
#The results tell us that the new variable Component 1 accounts for 41.5% and 
#comp 2 accounts for 31.9% for the information or variance in the data.
# Cumulative percentage is 73.4%

# new variables
new_v <- data.frame(pca$scores)
head(new_v)

#put column Churn into the comp dataframe
new_v$churn <- tele_1$churn_1

# How much of the variation or information in the data is preserved in this plot?
ggplot(new_v, aes(x = Comp.1, y = Comp.2, label = churn, col = churn))+
  geom_text(size = 2) +
  labs(x = "First Principal Component",
       y = "Second Principal Component") +
  coord_fixed(ratio = 1) 
# From the first principal component, factors make customers who switch provider 
# varies more than the customers who stay with the current provider


biplot(pca,
       xlab = "Contribution to 1st Principal Component",
       ylab = "Contribution to 2nd Principal Component")

# All the 4 variables makes a positive comtribution to the 1st principal component
# Factors of callwait and enqcount make negative contributions to the 2nd principal component
# Factors of upload and wedget contribute nearly 0 to the 2nd principal component



#---------------------------------------------------------------------------
#Machine Learning Part (f)∗∗∗: Apply the random forest (bagging) method to 
#construct a classifier to predict churn based on the two first principal 
#components as predictors. In doing so, use the split of the data into 
#a training and test set (you may use the same indices as in part (b)).

#Calculate the test error for the obtained random forest and comment on it.

#Visualise the resulting classification rule on the scatter plot of the 
#two first principal compo- nents.

# review the virables

set.seed(7)
#train set predictors and class
pca_train <- train[train_split, ]
pca_train_churn <- df.tree$churn_1[train_split]
#train_cl <- cl[train_split]


#test set predictors and class
pca_test <- train[-train_split,]
pca_test_churn <- df.tree$churn_1[-train_split]
#test_cl <- cl[-train_split]


pca.train.comp <- princomp(pca_train, cor = TRUE)

train_new <- data.frame(pca.train.comp$scores[,c(1,2)])
train_new$churn <- pca_train_churn

pca.test.comp <- princomp(pca_test, cor = TRUE)
test_new <- data.frame(pca.test.comp$scores[, c(1,2)])
test_new$churn <- pca_test_churn

rf.comp <- randomForest(churn ~., data = train_new)

rf.comp.pre <- predict(rf.comp, test_new)
rf.comp.tab <- table(rf.comp.pre, pca_test_churn)
rf.comp.tab

rf.pca.error <- (rf.comp.tab[1,2]+rf.comp.tab[2,1]) / sum(rf.comp.tab)
rf.pca.error
# The error rate is 0.1466667

biplot(pca.test.comp)


#---------------------------------------------------------------------------

pca_df <- data.frame(pca$scores[,c(1,2)], cl)

p_train <- pca_df[train_split, ]

p_test <- pca_df[-train_split, ]
p_cl <- cl[-train_split]

p_rf <- randomForest(cl ~ ., data = p_train)

p_pre <- predict(p_rf,p_test)
p_pre
p_tab <- table(p_pre, p_cl)
p_tab

p_error <- (p_tab[1,2]+p_tab[2,1]) / sum(p_tab)
p_error # 0.1933333


p_df <- data.frame(p_test, p_pre)

ggplot(p_df, aes(x = Comp.1, y = Comp.2, col = cl))+
  geom_point(size = 2) +
  labs(x = "First Principal Component",
       y = "Second Principal Component") +
  coord_fixed(ratio = 1) 


len <- 80
xp <- seq(-4, 8, length = len) 
yp <- seq(-4, 5, length = len) 
xygrid <- expand.grid(Comp.1 = xp, Comp.2 = yp)

grid.rf <- predict(p_rf, xygrid)

col3 <- rep("lightgreen", len*len) 

for (i in 1:(len*len)){
  if (grid.rf[i]== 'no') col3[i] <- "red"
  else if (grid.rf[i]== 'yes') col3[i] <- "lightblue"
}

plot(xygrid, col = col3, main = "RF")

points(p_df$Comp.1, p_df$Comp.2, col = col3, pch = 16)


s <- pca$sdev^2/sum(pca$sdev^2)
s[1]

