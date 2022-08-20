
###### breast cancer risk prediction model ######
library(data.table)
data <- read.table(file = "breast-cancer-wisconsin.data", sep = ",")
str(data)


names(data)<-c("ID","clumpThickness","sizeUniformity","shapeUniformity",
               "maginalAdhesion","singleEpithelialCellsize","bareNuclei",
               "blandChromatin","normalNucleoli","mitosis","class")
str(data)


trainset <- sample(nrow(data),0.7*nrow(data))
testset <-data[-trainset,]
trainset <-data[trainset,]

trainset$class <- ifelse(trainset$class=="2", "no", "yes")
testset$class <- ifelse(testset$class=="2", "no", "yes")
trainset$class <- as.factor(trainset$class)
testset$class <- as.factor(testset$class)

library(caret)
fitControl <- trainControl(method = "none", classProbs = TRUE)

set.seed(123456)
LR_model <- train(class~clumpThickness+sizeUniformity+shapeUniformity+
                    maginalAdhesion+singleEpithelialCellsize+bareNuclei+
                    blandChromatin+normalNucleoli+mitosis, 
                    data=trainset, 
                    method = "glm", 
                    trControl = fitControl, 
                    metric = "ROC")

set.seed(123456)
CART_model <- train(class~clumpThickness+sizeUniformity+shapeUniformity+
                      maginalAdhesion+singleEpithelialCellsize+bareNuclei+
                      blandChromatin+normalNucleoli+mitosis, 
                      data=trainset,
                      method = "rpart", 
                      trControl = fitControl, 
                      tuneGrid = data.frame(cp=0.001),
                      metric = "ROC")

set.seed(123456)
RF_model <- train(class~clumpThickness+sizeUniformity+shapeUniformity+
                    maginalAdhesion+singleEpithelialCellsize+bareNuclei+
                    blandChromatin+normalNucleoli+mitosis, 
                    data=trainset,
                    method = "parRF", 
                    trControl = fitControl, 
                    verbose = FALSE, 
                    tuneGrid = data.frame(.mtry=3),
                    metric = "ROC")

SVM_model <- train(class~clumpThickness+sizeUniformity+shapeUniformity+
                     maginalAdhesion+singleEpithelialCellsize+bareNuclei+
                     blandChromatin+normalNucleoli+mitosis, 
                     data=trainset,
                     method = "svmRadial", 
                     trControl = fitControl, 
                     verbose = FALSE, 
                     tuneGrid = data.frame(sigma = 0.05,C= 0.5),
                     metric = "ROC")

LR_pro <- predict(LR_model, newdata =testset, type = "prob")
CART_pro <- predict(CART_model, newdata =testset, type = "prob")
RF_pro <- predict(RF_model, newdata =testset, type = "prob")
SVM_pro <- predict(SVM_model, newdata =testset, type = "prob")

testset$LR <- LR_pro$yes
testset$CART <- CART_pro$yes
testset$RF <- RF_pro$yes
testset$SVM <- SVM_pro$yes


library(plotROC)
library(ggplot2)
ROC <- melt_roc(testset, "class", c("LR", "CART", "RF", "SVM"))
A <- ggplot(ROC, aes(d = D, m = M, color = name)) + 
     geom_roc(n.cuts = 0) +
     labs(title = "A.ROC Curves in the testset")+
     theme(plot.title = element_text(hjust = 0.5))+geom_abline()
A

library(pROC)
roc_LR <- roc(testset$class, testset$LR)
auc_LR <- auc(roc_LR)
auc_LR # Area under the curve: 0.9887

roc_CART <- roc(testset$class, testset$CART)
auc_CART <- auc(roc_CART)
auc_CART # Area under the curve: 0.9577

roc_RF <- roc(testset$class, testset$RF)
auc_RF <- auc(roc_RF)
auc_RF # Area under the curve: 0.9830

roc_SVM <- roc(testset$class, testset$SVM)
auc_SVM <- auc(roc_SVM)
auc_SVM # Area under the curve: 0.9828


library(rms)
library(ggplot2)
trellis.par.set(caretTheme())
cal_obj <- calibration(class ~LR+CART+RF+SVM,
                       data = testset, class = "yes", cuts = 5)
B <- ggplot(cal_obj) +geom_line()+
  labs(title = "B.Calibration Curves in the testset")+
  theme(plot.title = element_text(hjust = 0.5))+geom_abline()
B

# testset$class <- ifelse(testset$class==1, "yes", "no")
# testset$class <- as.factor(testset$class)

library(rms)
library(ggplot2)
trellis.par.set(caretTheme())
cal_obj <- calibration(class ~LR+RF,
                       data = testset, class = "yes", cuts = 5)
B <- ggplot(cal_obj) +geom_line()+
  labs(title = "B.Calibration Curves in the testset")+
  theme(plot.title = element_text(hjust = 0.5))+geom_abline()
B

library(rms)
library(ggplot2)
trellis.par.set(caretTheme())
cal_obj <- calibration(class ~LR+CART+RF+SVM,
                       data = testset, class = "yes", cuts = 5)
plot(cal_obj,type = "l", auto.key = list(columns = 4,
                                         lines = TRUE,
                                         points = F))

title(main="B.Calibration Curves in the testset")
abline(a=0, b= 1, col = "black")

ggplot(cal_obj)


library(gbm)
calibrate.plot(testset$class, testset[,14], 
               replace=T, xlim=c(0,1.0), ylim=c(0,1.0))


LR_brier <- mean((testset$LR-as.numeric(testset$class)+1)^2)
LR_brier   # 0.03683759
CART_brier <- mean((testset$CART-as.numeric(testset$class)+1)^2)
CART_brier   # 0.04374112
RF_brier <- mean((testset$RF-as.numeric(testset$class)+1)^2)
RF_brier   # 0.03734895
SVM_brier <- mean((testset$SVM-as.numeric(testset$class)+1)^2)
SVM_brier   # 0.04409903



testset$class <- ifelse(testset$class=="yes", 1, 0)

source("dca.R")
attach(testset)
DCA <- dca(data=testset, outcome="class", 
           predictors=c("LR","CART","RF","SVM"))
title(main="C.DCA curves in testset")

###### a machine learning function ######
library(lilikoi)
library(RCy3)
dt = lilikoi.Loaddata(file=system.file("extdata","plasma_breast_cancer.csv", package = "lilikoi"))
Metadata <- dt$Metadata
lilikoi.machine_learning(MLmatrix = Metadata, measurementLabels = Metadata$Label,
   significantPathways = 0,
   trainportion = 0.8, cvnum = 10, dlround=50,Rpart=TRUE,
   LDA=FALSE,SVM=FALSE,RF=FALSE,GBM=FALSE,PAM=FALSE,LOG=FALSE,DL=FALSE)

###### heart dataset ######

library(DriveML)
data(heart)
heart <- as.data.frame(data(heart))


