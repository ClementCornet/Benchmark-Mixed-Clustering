.libPaths("/home/loup-noe/Projets/DVRC/Benchmark-Mixed-Clustering/R_lib")
library(RMixtComp)
library(rjson)

data <- read.csv(file = "temp_data.csv")
model <- fromJSON(file = "model.json")
k <- fromJSON(file = "k.json")$n_clusters
k <- as.numeric(k)

algo <- list(nbBurnInIter = 50,
             nbIter = 50,
             nbGibbsBurnInIter = 50,
             nbGibbsIter = 50,
             nInitPerClass = 20,
             nSemTry = 20,
             confidenceLevel = 0.95)


resLearn1 <- mixtCompLearn(data, model, algo,
                           nClass = k, nRun = 2, nCore = 1)

resPredict <- mixtCompPredict(data, model, algo,
                              resLearn1, nClass = k, nCore = 1)

clusters <- resPredict$variable$data$z_class$completed

write.csv(clusters, "mixtcomp_temp.csv", row.names = FALSE)