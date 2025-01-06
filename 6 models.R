library(data.table); library(readr); library(dplyr); library(LKT); library(boot); library(bit64); library(pROC)

all_data <- setDT(val[order(val$CF..Time.),])
all_data[, fold := cut(.I, breaks = 100, labels = 1:100)]
res1 <- res2 <- res3 <-res4 <-res5 <-res6 <- data.frame(RMSE = numeric(), LL = numeric(), N = numeric(), AUC = numeric())

for (i in 1:30) {
  # Model AFM (fixed effects)
  modelob1 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("intercept", "intercept", "lineafm$"))
  pred1 <- pmin(pmax(inv.logit(as.matrix(modelob1$predictors %*% modelob1$coefs)[, ]), 1e-5), 0.99999)[modelob1$newdata$fold %in% (i+1):min(100, i+70)]
  actual1 <- modelob1$newdata$CF..ansbin.[modelob1$newdata$fold %in% (i+1):min(100, i+70)]
  res1 <- rbind(res1, c(sqrt(mean((actual1 - pred1)^2)), -mean(actual1 * log(pred1) + (1 - actual1) * log(1 - pred1)), length(actual1), as.numeric(roc(actual1, pred1)$auc)))
  
  # Model 2 AFM (with logitdec instead of student int)
  modelob2 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "intercept", "lineafm$",), fixedpars = c(0.98))
  if (i == 2) vals <- modelob2$coefs
  pred2 <- pmin(pmax(inv.logit(as.matrix(modelob2$predictors %*% vals)[, ]), 1e-5), 0.99999)[modelob2$newdata$fold %in% (i+1):min(100, i+70)]
  actual2 <- modelob2$newdata$CF..ansbin.[modelob2$newdata$fold %in% (i+1):min(100, i+70)]
  res2 <- rbind(res2, c(sqrt(mean((actual2 - pred2)^2)), -mean(actual2 * log(pred2) + (1 - actual2) * log(1 - pred2)), length(actual2), as.numeric(roc(actual2, pred2)$auc)))
  
  # Model 3 AFM (with logitdec evol instead of KC int)
  modelob3 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("intercept", "logitdecevol", "lineafm$"), fixedpars = c(.98))
  pred3 <- pmin(pmax(inv.logit(as.matrix(modelob3$predictors %*% modelob3$coefs)[, ]), 1e-5), 0.99999)[modelob3$newdata$fold %in% (i+1):min(100, i+70)]
  actual3 <- modelob3$newdata$CF..ansbin.[modelob3$newdata$fold %in% (i+1):min(100, i+70)]
  res3 <- rbind(res3, c(sqrt(mean((actual3 - pred3)^2)), -mean(actual3 * log(pred3) + (1 - actual3) * log(1 - pred3)), length(actual3), as.numeric(roc(actual3, pred3)$auc)))

  # Model 4 AFM (with logitdec student and logitdec KC instead of ints)
  modelob4 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "lineafm$"), fixedpars = c(.98, .98))
  pred4 <- pmin(pmax(inv.logit(as.matrix(modelob4$predictors %*% modelob4$coefs)[, ]), 1e-5), 0.99999)[modelob4$newdata$fold %in% (i+1):min(100, i+70)]
  actual4 <- modelob4$newdata$CF..ansbin.[modelob4$newdata$fold %in% (i+1):min(100, i+70)]
  res4 <- rbind(res4, c(sqrt(mean((actual4 - pred4)^2)), -mean(actual4 * log(pred4) + (1 - actual4) * log(1 - pred4)), length(actual4), as.numeric(roc(actual4, pred4)$auc)))
  
  # Model 5 Logitdec AFM
  modelob5 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "logitdec"), fixedpars = c(.98, .98, .8))
  pred5 <- pmin(pmax(inv.logit(as.matrix(modelob5$predictors %*% modelob5$coefs)[, ]), 1e-5), 0.99999)[modelob5$newdata$fold %in% (i+1):min(100, i+70)]
  actual5 <- modelob5$newdata$CF..ansbin.[modelob5$newdata$fold %in% (i+1):min(100, i+70)]
  res5 <- rbind(res5, c(sqrt(mean((actual5 - pred5)^2)), -mean(actual5 * log(pred5) + (1 - actual5) * log(1 - pred5)), length(actual5), as.numeric(roc(actual5, pred5)$auc)))
  
  # Model 6 Logitdec AFM with different student/KC change rates
  modelob6 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "logitdec$"), fixedpars = c(.98, .98, .8))
  pred6 <- pmin(pmax(inv.logit(as.matrix(modelob6$predictors %*% modelob6$coefs)[, ]), 1e-5), 0.99999)[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  actual6 <- modelob6$newdata$CF..ansbin.[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  res6 <- rbind(res6, c(sqrt(mean((actual6 - pred6)^2)), -mean(actual6 * log(pred6) + (1 - actual6) * log(1 - pred6)), length(actual6), as.numeric(roc(actual6, pred6)$auc)))
  
  
  
  
  }

colnames(res1) <- colnames(res2) <- colnames(res3) <- colnames(res4)<- colnames(res5)<- colnames(res6)<- c("RMSE", "LL", "N", "AUC")
# Dynamically calculate y-limits for RMSE
rmse_ylim <- range(c(res1$RMSE, res2$RMSE, res3$RMSE, res4$RMSE, res5$RMSE, res6$RMSE))

# RMSE Plot
par(mar = c(5, 4, 2, 2))
plot(res1$RMSE, type = "o", pch = 16, col = "blue", ylim = rmse_ylim,
     xlab = "Percentage of Data Used in Training (%)", ylab = "RMSE")
lines(res2$RMSE, type = "o", pch = 16, col = "red")
lines(res3$RMSE, type = "o", pch = 16, col = "darkgreen")
lines(res4$RMSE, type = "o", pch = 16, col = "purple")
lines(res5$RMSE, type = "o", pch = 16, col = "orange")
lines(res6$RMSE, type = "o", pch = 16, col = "brown")
legend("topright", 
       legend = c("Model 1: AFM fixed effects",
                  "Model 2: AFM w/ logitdec (no student int)",
                  "Model 3: AFM w/ logitdecevol (no KC int)",
                  "Model 4: Logitdec Student+KC",
                  "Model 5: Logitdec AFM",
                  "Model 6: Logitdec AFM w/ diff rates"),
       col = c("blue", "red", "darkgreen", "purple", "orange", "brown"),
       pch = 16, lty = 1)

# Dynamically calculate y-limits for AUC
auc_ylim <- range(c(res1$AUC, res2$AUC, res3$AUC, res4$AUC, res5$AUC, res6$AUC))

# AUC Plot
par(mar = c(5, 4, 2, 2))
plot(res1$AUC, type = "o", pch = 16, col = "blue", ylim = auc_ylim,
     xlab = "Percentage of Data Used in Training (%)", ylab = "AUC")
lines(res2$AUC, type = "o", pch = 16, col = "red")
lines(res3$AUC, type = "o", pch = 16, col = "darkgreen")
lines(res4$AUC, type = "o", pch = 16, col = "purple")
lines(res5$AUC, type = "o", pch = 16, col = "orange")
lines(res6$AUC, type = "o", pch = 16, col = "brown")
legend("bottomright", 
       legend = c("Model 1: AFM fixed effects",
                  "Model 2: AFM w/ logitdec (no student int)",
                  "Model 3: AFM w/ logitdecevol (no KC int)",
                  "Model 4: Logitdec Student+KC",
                  "Model 5: Logitdec AFM",
                  "Model 6: Logitdec AFM w/ diff rates"),
       col = c("blue", "red", "darkgreen", "purple", "orange", "brown"),
       pch = 16, lty = 1)

