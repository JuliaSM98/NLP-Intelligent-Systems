################################################################################
################################# NLP project code #############################
################################################################################
library(dplyr)
library(tm)
library(tidytext)
library(ggplot2)
library(SnowballC)
library(textstem)
library(caret)
library(randomForest)
library(caTools)
library(utf8)
library(readr)


#Load dataset
df_train=read.csv("train.csv",header=TRUE,sep=",",encoding = "UTF-8")

#See if there are missing values 
anyNA(df_train) #the data is clean of missing values 

################################################################################
######### See the 10 most frequent words for each topic in the data ############ 
################################################################################

#Computer Science 
Topic <- df_train[which(df_train$Computer.Science == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                                  stripWhitespace = T,
                                                  removeNumbers = T,
                                                  tolower= T,
                                                  stopwords = T,
                                                  stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Computer Science topic")

#Physics
Topic <- df_train[which(df_train$Physics == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                               stripWhitespace = T,
                                               removeNumbers = T,
                                               tolower= T,
                                               stopwords = T,
                                               stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Physics topic")

#Mathematics
Topic <- df_train[which(df_train$Mathematics == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                               stripWhitespace = T,
                                               removeNumbers = T,
                                               tolower= T,
                                               stopwords = T,
                                               stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Mathematics topic")

#Statistics
Topic <- df_train[which(df_train$Statistics == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                               stripWhitespace = T,
                                               removeNumbers = T,
                                               tolower= T,
                                               stopwords = T,
                                               stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Statistics topic")

#Quantitative.Biology
Topic <- df_train[which(df_train$Quantitative.Biology == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                               stripWhitespace = T,
                                               removeNumbers = T,
                                               tolower= T,
                                               stopwords = T,
                                               stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Quantitative Biology topic")

#Quantitative.Finance
Topic <- df_train[which(df_train$Quantitative.Finance == 1), ]
cat("total rows: ", nrow(Topic))
Topic_Corp = VCorpus(VectorSource(Topic))
CS = TermDocumentMatrix(Topic_Corp,control=list(removePunctuation = T,
                                               stripWhitespace = T,
                                               removeNumbers = T,
                                               tolower= T,
                                               stopwords = T,
                                               stemming = T))
r=rowSums(as.matrix(CS))
plot = head(sort(r, decreasing=TRUE),n=10) 
barplot(plot, main="Most frequent words for the Quantitative Finance topic")

################################################################################
##################### Machine learning application #############################
################################################################################

# To make the algorithm easier, we will remove the rows of the articles that 
# belong to more than one topic 
df_train$new_col <- 0 
for(i in 1:dim(df_train)[1]) {
  c = sum(df_train[i,4:9])
  df_train$new_col[i] <- c  
  c = 0
}
df_train<-df_train[df_train$new_col == 1, ] 

df_train <- subset( df_train, select = -new_col )

# We will joint abstract with title in the TEXT variable
df_train$TEXT <- paste(df_train$TITLE,df_train$ABSTRACT)

# We create a target variable
df_train$TARGET <- 0
for(i in 1:dim(df_train)[1]) {
  if (df_train[i,4]==1) { df_train$TARGET[i] <- 1 }
  else if (df_train[i,5]==1) { df_train$TARGET[i] <- 2 }
  else if (df_train[i,6]==1) { df_train$TARGET[i] <- 3 }
  else if (df_train[i,7]==1) { df_train$TARGET[i] <- 4 }
  else if (df_train[i,8]==1) { df_train$TARGET[i] <- 5 }
  else if (df_train[i,9]==1) { df_train$TARGET[i] <- 6 }
} # now we have a text variable and a target variable 

########################## Pre-processing the data #############################

# Cite: https://www.pluralsight.com/guides/machine-learning-text-data-using-r 

#Check encoding
df_train$TEXT[!utf8_valid(df_train$TEXT)]
#Check character normalization
df_train$TEXT_NFC <- utf8_normalize(df_train$TEXT)
#Check if the text is in NFC
sum(df_train$TEXT_NFC != df_train$TEXT)
df_train <- subset( df_train, select = -TEXT_NFC )
#Remove /n, /t and space
df_train$TEXT <- gsub("[\n]{1,}", " ", df_train$TEXT)
df_train$TEXT <- gsub("[\t]{1,}", " ", df_train$TEXT)
df_train$TEXT <- gsub("  ", " ", df_train$TEXT)

#Convert data to Corpus
corpus = Corpus(VectorSource(df_train$TEXT))
#Convert data to lowercase
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
#Remove punctuation
corpus = tm_map(corpus, removePunctuation)
#Remove stopwords
corpus = tm_map(corpus, removeWords, c("cloth", stopwords("english")))
#Stemming
corpus = tm_map(corpus, stemDocument)


######################## Creating a document term matrix #######################

frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995)
tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(tSparse))
tSparse$target = df_train$TARGET

#See proportion of labels in target variable
prop.table(table(tSparse$target)) 

##########################Create test and training set##########################
split=sample(length(tSparse$target),0.7*length(tSparse$target))
trainSparse <- tSparse[split,]
testSparse <- tSparse[-split,]


##########################Random forest#########################################

trainSparse$target = as.factor(trainSparse$target)
testSparse$target = as.factor(testSparse$target )
RF_model = randomForest(target~., data=trainSparse, ntree=200,do.trace=TRUE)
#RandomForest takes a lot of time
predictRF = predict(RF_model, newdata=testSparse)
confusionMatrix(testSparse$target, predictRF)


