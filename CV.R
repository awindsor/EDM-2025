
library(data.table)
library(readr)
library(dplyr)
library(LKT)
library(boot)
library(bit64)



  
  
  all_data<-setDT(val[order(val$CF..Time.),])
  
  
  all_data[, fold := cut(.I, breaks = 11, labels = 1:11)]
  
  
  # Creating the first data frame 'res'
  res <- data.frame(RMSE = numeric(), LL = numeric(), N = numeric())
  
  for (i in 1:10) {
    #print(paste("training folds",(1:i)))
    
    
    modelob2 <-    LKT(verbose=F,data = all_data, interc=T,dualfit = FALSE,factrv = 1e11,
        components = c("Anon.Student.Id","KC..Default.","KC..Default.","KC..Default.")
        ,features = c("logitdec", "logsuc","recency","logitdecevol"),fixedpars =c(0.98, 0.24,.99))
    if(i==1){print(modelob2$coefs)}
    
    pred <- as.vector(pmin(pmax(inv.logit(
      as.matrix(modelob2$predictors %*% modelob2$coefs)[,]
    ), .00001), .99999)[modelob2$newdata$fold %in% (i+1)])
    #  print(pred)
    actual<-modelob2$newdata$CF..ansbin.[modelob2$newdata$fold %in% (i+1)]
    #print(actual)
    res<-rbind(res,c(sqrt(mean((actual-pred)^2)),
                     -mean(actual * log(pred) + (1 - actual) * log(1 - pred)),
                     length(actual)))
    
  
   
  }
  colnames(res) <- c("RMSE", "LL", "N")
  
  print(res)
