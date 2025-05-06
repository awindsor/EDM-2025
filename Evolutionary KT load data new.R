
library(LKT)
set.seed(41)
val<-largerawsample

#clean it up
val$KC..Default.<-val$Problem.Name
# make it a data table
# val= setDT(val)

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

colnames(val2)[colnames(val2) == "KC..MATHia."] <- "KC..Default."
val2 <- suppressWarnings(computeSpacingPredictors(val2, "KC..Default.")) #allows recency, spacing, forgetting features to run
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Problem.Name")) #allows recency, spacing, forgetting features to run
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Anon.Student.Id")) #allows recency, spacing, forgetting features to run





# Load interleaving and blocking (Patel) (example how to load a remote dataset)
set.seed(41)
datafile<-"C:/Users/ppavl/OneDrive/Active projects/ds1706_tx_All_Data_3416_2017_0623_020504.txt" # CHANGE THIS VALUE TO THE DataShop export file IN YOUR R WORKING DIRECTORY
val3<-read.delim(colClasses = c("Anon.Student.Id"="character"),datafile,sep="\t", header=TRUE,quote="")
val3=as.data.table(val3)
val3$CF..Time.<-as.numeric(as.POSIXct(as.character(val3$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val3<-val3[order(val3$Anon.Student.Id, val3$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value

#val3$Outcome<-ifelse(tolower(val3$Outcome)=="ok","CORRECT","INCORRECT")
val3$CF..ansbin.<-ifelse(tolower(val3$Outcome)=="correct",1,0)
val3<-val3[val3$CF..ansbin.==0 | val3$CF..ansbin.==1,]

#subtot<-  aggregate(val3$CF..ansbin.,by=list(val3$Anon.Student.Id),FUN=length)
# subtot<- subtot[subtot$x<20,]
# val3<-val3[!(val3$Anon.Student.Id %in% subtot$Group.1),]
val3<-val3[val3$Attempt.At.Step==1,]
val3<-val3[val3$KC..Field.!="",]
# make student stratified folds (for crossvalidation for unseen population)

colnames(val3)[colnames(val3) == "KC..Field."] <- "KC..Default."
val3 <- suppressWarnings(computeSpacingPredictors(val3, "KC..Default.")) #allows recency, spacing, forgetting features to run
val3 <- suppressWarnings(computeSpacingPredictors(val3, "Anon.Student.Id")) #allows recency, spacing, forgetting features to run




