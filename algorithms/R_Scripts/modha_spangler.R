.libPaths("/home/loup-noe/Projets/DVRC/Benchmark-Mixed-Clustering/R_lib")
library(kamila)
library(rjson)

# Load k from JSON
k <- fromJSON(file = "k.json")$n_clusters
k <- as.numeric(k)

# Load numerical (continuous variables)
con_vars <- read.csv(file = "temp_continue.csv")
con_vars <- data.frame(scale(con_vars))

# Load categorical variables
cat_vars_fac <- read.csv(file = "temp_cat.csv")

# Clean categorical variables to fit kamila package 's needs
cat_vars_fac[] <- lapply(cat_vars_fac, factor)
cat_vars_dum <- dummyCodeFactorDf(cat_vars_fac)
cat_vars_dum <- data.frame(cat_vars_dum)

# Process Modha-Spangler (NORMAL IF IT TAKES LONG)
gms_res_hw <- gmsClust(con_vars, cat_vars_dum, nclust = k)

clusters <- gms_res_hw$results$cluster - 1

write.csv(clusters, "temp_clustered.csv", row.names = FALSE)