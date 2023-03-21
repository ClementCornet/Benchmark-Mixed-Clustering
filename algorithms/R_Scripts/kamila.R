library(kamila, lib.loc = "/home/loup-noe/Projets/DVRC/Benchmark-Mixed-Clustering/R_lib")
library("rjson", lib.loc = "/home/loup-noe/Projets/DVRC/Benchmark-Mixed-Clustering/R_lib")

# Python passes k (number of clusters) through a JSON file
k <- fromJSON(file = "k.json")$n_clusters
k <- as.numeric(k)

# Load numerical (continuous) variables and scale
con_vars <- read.csv(file = "temp_continue.csv")
con_vars <- data.frame(scale(con_vars))

# Load categorical variables
cat_vars_fac <- read.csv(file = "temp_cat.csv")

# Rearranging catagorical data to fit the Kamila package's needs
cat_vars_fac[] <- lapply(cat_vars_fac, factor)
cat_vars_dum <- dummyCodeFactorDf(cat_vars_fac)
cat_vars_dum <- data.frame(cat_vars_dum)

# Process Kamila
kam_res <- kamila(con_vars,
                 cat_vars_fac,
                 numClust = k,
                 numInit = 10,
                 maxIter = 25)
clusters <- kam_res$finalMemb - 1

write.csv(clusters, "temp_clustered.csv", row.names = FALSE)

# Write Clustered data to a file
#df <- cbind(
#    con_vars,
#    cat_vars_fac,
#    cluster = clusters
#)

#write.csv(df, "temp_clustered.csv", row.names = FALSE)