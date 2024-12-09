
library(LKT)
set.seed(41)
val<-largerawsample

#clean it up
val$KC..Default.<-val$Problem.Name
# make it a data table
# val= setDT(val)

#make unstratified folds for crossvaldiations
val$fold<-sample(1:5,length(val$Anon.Student.Id),replace=T)


# make student stratified folds (for crossvalidation for unseen sample)
unq = sample(unique(val$Anon.Student.Id))
sfold = rep(1:5,length.out=length(unq))
val$fold = rep(0,length(val[,1]))
for(i in 1:5){val$fold[which(val$Anon.Student.Id %in% unq[which(sfold==i)])]=i}

# get the times of each trial in seconds from 1970
val$CF..Time.<-as.numeric(as.POSIXct(as.character(val$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val<-val[order(val$Anon.Student.Id, val$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value
val$CF..ansbin.<-ifelse(tolower(val$Outcome)=="correct",1,ifelse(tolower(val$Outcome)=="incorrect",0,-1))
val<-val[val$CF..ansbin.==0 | val$CF..ansbin.==1,]



# create durations
val$Duration..sec.<-(val$CF..End.Latency.+val$CF..Review.Latency.+500)/1000

# this function needs times and durations but you don't need it if you don't want to model time effects
val <- computeSpacingPredictors(val, "KC..Default.") #allows recency, spacing, forgetting features to run
val <- computeSpacingPredictors(val, "KC..Cluster.") #allows recency, spacing, forgetting features to run
val <- computeSpacingPredictors(val, "Anon.Student.Id") #allows recency, spacing, forgetting features to run
val <- computeSpacingPredictors(val, "CF..Correct.Answer.") #allows recency, spacing, forgetting features to run






# Load MATHia (example how to load a remote dataset)
set.seed(41)
datafile<-"C:/Users/ppavl/OneDrive/Active projects/ds4845_tx_All_Data_6977_2021_0723_141809.txt" # CHANGE THIS VALUE TO THE DataShop export file IN YOUR R WORKING DIRECTORY
val2<-read.delim(colClasses = c("Anon.Student.Id"="character"),datafile,sep="\t", header=TRUE,quote="")
val2=as.data.table(val2)
val2$CF..Time.<-as.numeric(as.POSIXct(as.character(val2$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val2<-val2[order(val2$Anon.Student.Id, val2$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value

val2$Outcome<-ifelse(tolower(val2$Outcome)=="ok","CORRECT","INCORRECT")
val2$CF..ansbin.<-ifelse(tolower(val2$Outcome)=="correct",1,0)
val2<-val2[val2$CF..ansbin.==0 | val2$CF..ansbin.==1,]

#subtot<-  aggregate(val2$CF..ansbin.,by=list(val2$Anon.Student.Id),FUN=length)
# subtot<- subtot[subtot$x<20,]
# val2<-val2[!(val2$Anon.Student.Id %in% subtot$Group.1),]
val2<-val2[val2$Attempt.At.Step==1,]
val2<-val2[val2$KC..MATHia.!="",]
# make student stratified folds (for crossvalidation for unseen population)
unq = sample(unique(val2$Anon.Student.Id))
sfold = rep(1:5,length.out=length(unq))
val2$fold = rep(0,length(val2[,1]))
for(i in 1:5){val2$fold[which(val2$Anon.Student.Id %in% unq[which(sfold==i)])]=i}


val2 <- suppressWarnings(computeSpacingPredictors(val2, "KC..MATHia.")) #allows recency, spacing, forgetting features to run
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Problem.Name")) #allows recency, spacing, forgetting features to run
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Anon.Student.Id")) #allows recency, spacing, forgetting features to run




val<-val[order(val$CF..Time.),]
modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..Default.","KC..Default.","KC..Default.")
               ,features = c("logitdec", "logsuc","recency","logitdecevol"),fixedpars =c(0.98, 0.24,.99))
modelob$coefs

modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..Default.","KC..Default.","KC..Default.")
               ,features = c("logitdec", "logsuc","recency","intercept"),fixedpars =c(0.98, 0.24,.99))


modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..Default.","KC..Default.")
               ,features = c("logitdec", "logsuc","recency"),fixedpars =c(0.98, 0.24,.99))
modelob$coefs




val<-val2[order(val2$CF..Time.),]
modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..MATHia.","KC..MATHia.","KC..MATHia.")
               ,features = c("logitdec", "logsuc","recency","logitdecevol"),fixedpars =c(0.98, 0.24,.99))
modelob$coefs

modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..MATHia.","KC..MATHia.","KC..MATHia.")
               ,features = c("logitdec", "logsuc","recency","intercept"),fixedpars =c(0.98, 0.24,.99))


modelob <- LKT(data = setDT(val), interc=TRUE,dualfit = FALSE,factrv = 1e11,
               components = c("Anon.Student.Id","KC..MATHia.","KC..MATHia.")
               ,features = c("logitdec", "logsuc","recency"),fixedpars =c(0.98, 0.24,.99))
modelob$coefs

