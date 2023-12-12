.libPaths("/home/loup-noe/Projets/DVRC/Benchmark-Mixed-Clustering/R_lib")
library(clustMD)
library(rjson)

# Python passes k (number of clusters) through a JSON file
k <- fromJSON(file = "k.json")$n_clusters
k <- as.numeric(k)

# Load numerical (continuous) variables and scale
con_vars <- read.csv(file = "temp_continue.csv")
con_vars <- data.frame(scale(con_vars))

# Load categorical variables
cat_vars_fac <- read.csv(file = "temp_cat.csv")
cat_vars_fac <- data.frame(cat_vars_fac)

con_col <- ncol(con_vars)
cat_col <- ncol(cat_vars_fac)

df <- cbind(con_vars, cat_vars_fac)

res <- clustMD(X = df, G = k, CnsIndx = con_col, OrdIndx = con_col + cat_col,
         Nnorms = 20000, MaxIter = 500, model = "EVI", store.params = FALSE,
          scale = TRUE, startCL = "kmeans")

clusters <- res$cl - 1

#print(clusters)

write.csv(clusters, "temp_clustered.csv", row.names = FALSE)