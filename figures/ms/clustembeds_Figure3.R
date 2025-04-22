# Compares clustering after dimensionality reduction of different embedding methods using rand index
#
library(Rtsne) # for t-SNE dim reduction
library(uwot) # for UMAP dim reduction
library(ggplot2)
library(RColorBrewer)
library(ggrepel)
library(dplyr)
library(scales)
library(fossil)
library(clue)

rm(list = ls()) # remove previous variables from the environment

pathname <- "my_path_for_script"
setwd(pathname)
#
theme_set(theme_bw())
select <- dplyr::select
### define functions ------------
#
# Function to join drug comb. data with druginfo based on compoundB
process_data_CompB <- function(datacomb, druginfo) {
  drugdata <- datacomb %>%
    left_join(druginfo, by = c("COMPOUND_A" = "Compound")) %>%
    rename(Target_A = Target, Function_A = Function, Pathway_A = Pathway)
  # Then, join again with druginfo based on compound_B
  drugdata <- drugdata %>%
    left_join(druginfo, by = c("COMPOUND_B" = "Compound")) %>%
    rename(Target_B = Target, Function_B = Function, Pathway_B = Pathway) %>%
    mutate(Score_z_pred = (SYNERGY_PRED - mean(SYNERGY_PRED)) / sd(SYNERGY_PRED),
           Score_z_exp = (SYNERGY_SCORE - mean(SYNERGY_SCORE)) / sd(SYNERGY_SCORE))
  # Reorder columns to place Target_A, Function_A, Pathway_A after compound_A,
  # and Target_B, Function_B, Pathway_B after compound_B
  drugdata <- drugdata %>%
    select(COMPOUND_A, Target_A, Function_A, Pathway_A, 
           COMPOUND_B, Target_B, Function_B, Pathway_B, everything())
  return(drugdata)
}

# Function to merge cluster information with drug data combinations and synergy scores
assign_clusters <- function(drugdata, cluster_results) {
  # Filter for training dataset
  train_data <- drugdata %>% filter(DATASET == "train")
  
  # Merge cluster information based on COMPOUND_B matching 'compound' in cluster_results
  all_clusters_data <- train_data %>%
    left_join(cluster_results, by = c("COMPOUND_B" = "Compound")) %>%
    mutate(Cluster = paste0("Cluster_", cluster)) %>%
    select(SYNERGY_PRED, SYNERGY_SCORE, COMPOUND_A, COMPOUND_B, CELL_LINE, 
           Cluster, Score_z_pred, Score_z_exp)
  
  return(all_clusters_data)
}

# Function to perform t-SNE dimensionality reduction
run_tsne <- function(embedding_data, drug_name, target, Function, pathway, seed, perplexity) {
  set.seed(seed)
  tsne_result <- Rtsne(embedding_data, dims = 2, perplexity = perplexity, theta = 0)
  
  # Create a dataframe with t-SNE results and metadata
  tsne_df <- data.frame(
    X = tsne_result$Y[, 1],
    Y = tsne_result$Y[, 2],
    Compound = drug_name,
    Target = target,
    Function = Function,
    Pathway = pathway
  )
  
  return(tsne_df)
}
#
# Function to perform t-SNE dimensionality reduction
run_umap <- function(embedding_data, drug_name, target, Function, pathway, seed, nneigh, mindist) {
  set.seed(seed)
  umap_result <- umap(embedding_data, n_components = 2, n_neighbors = nneigh, min_dist = mindist) 
  
  # Create a embeds frame with t-SNE results and labels
  umap_df <- data.frame(
    X = umap_result[,1],
    Y = umap_result[,2],
    Compound = drug_name,
    Target = target,
    Function = Function,
    Pathway = pathway
  )
  
  return(umap_df)
}
#
# Function to perform K-means clustering
run_kmeans <- function(tsne_df, num_clusters) {
  embed_tsne <- tsne_df[, c("X", "Y")]
  
  set.seed(42)
  kmeans_result <- kmeans(embed_tsne, centers = num_clusters, iter.max = 1000, nstart = 200)
  
  tsne_df$cluster <- as.factor(kmeans_result$cluster)
  return(tsne_df)
}

# Function to align cluster labels by best overall match between cluster elements
align_clusters_LSAP <- function(df1, df2, num_clusters) {
  # Create the overlap matrix
  overlap_matrix <- matrix(0, nrow = num_clusters, ncol = num_clusters)
  
  for (i in 1:num_clusters) {
    for (j in 1:num_clusters) {
      overlap_matrix[i, j] <- sum(df1$Compound[df1$cluster == i] %in% df2$Compound[df2$cluster == j])
    }
  }
  print(overlap_matrix)
  # Solve the optimal assignment using the Hungarian algorithm
  assignment <- solve_LSAP(overlap_matrix, maximum = TRUE)
  cluster_map <- setNames(seq_len(num_clusters), assignment) # original labels of df2$clusters are stored as 'name' attribbute, and 'values' are the new cluster labels
  print(cluster_map)
  # Apply the mapping: Replace original df2$cluster values using cluster_map
  df2$cluster <- factor(cluster_map[as.character(df2$cluster)])
  return(df2)
}  # from chatGPT

# Function to compute rand Index
rand_index <- function(df1, df2) {
  rand_index <- rand.index(as.numeric(df1$cluster), as.numeric(df2$cluster))
  return(rand_index)
}

# Function to compute adjjsted rand index
adj_rand_index <- function(df1, df2) {
  rand_index <- adj.rand.index(as.numeric(df1$cluster), as.numeric(df2$cluster))
  return(rand_index)
}

# Function to plot both t-SNE clustering results to compare
plot_clusters <- function(result1, result2, num_clusters) {
  p1 <- ggplot(result1, aes(x = X, y = Y, color = cluster, shape = Function)) +
    geom_point(size = 2, alpha = 0.7) +
    labs(x = "t-SNE 1", y = "t-SNE 2", title = "t-SNE Clusters - Dataset 1") +
    theme_minimal() +
    theme(legend.position = "right",
          axis.title.x = element_text(color="black", size=14, face="bold"),
          axis.title.y = element_text(color="black", size=14, face="bold"),
          axis.text.x = element_text(size=12, face = 'bold'),
          axis.text.y = element_text(size=12, face = 'bold'),
          axis.line = element_line(color = 'black', size = 1)) +
    scale_color_manual(name = 'Cluster', values = scales::hue_pal()(num_clusters)) +
    guides(color = guide_legend(override.aes = list(size = 4)), 
           shape = guide_legend(override.aes = list(size = 3))) +
    geom_text_repel(aes(label = Compound), size = 2.5, max.overlaps = Inf, 
                    box.padding = 0.5, point.padding = 0.3, show.legend = FALSE)
  
  p2 <- ggplot(result2, aes(x = X, y = Y, color = cluster, shape = Function)) +
    geom_point(size = 2, alpha = 0.7) +
    labs(x = "t-SNE 1", y = "t-SNE 2", title = "t-SNE Clusters - Dataset 2 (Aligned)") +
    theme_minimal() +
    theme(legend.position = "right",
          axis.title.x = element_text(color="black", size=14, face="bold"),
          axis.title.y = element_text(color="black", size=14, face="bold"),
          axis.text.x = element_text(size=12, face = 'bold'),
          axis.text.y = element_text(size=12, face = 'bold'),
          axis.line = element_line(color = 'black', size = 1)) +
    scale_color_manual(name = 'Cluster', values = scales::hue_pal()(num_clusters)) +
    guides(color = guide_legend(override.aes = list(size = 4)), 
           shape = guide_legend(override.aes = list(size = 3))) +
    geom_text_repel(aes(label = Compound), size = 2.5, max.overlaps = Inf, 
                    box.padding = 0.5, point.padding = 0.3, show.legend = FALSE)
  
  gridExtra::grid.arrange(p1, p2, ncol = 2)
}

# Plot only one t-SNE clustering for Figure 3 
plot_1cluster <- function(result, num_clusters) {
  p <- ggplot(result, aes(x = X, y = Y, color = cluster, shape = Function)) +
    geom_point(size = 3, alpha = 0.7) +
    labs(x = "t-SNE 1", y = "t-SNE 2") +
    theme_minimal() +
    theme(# legend.position = "none",
      legend.position = "right",
      legend.title = element_text(size = 14, face = "bold"),  # Change legend title
      legend.text = element_text(size = 12, face = "bold"),
      axis.title.x = element_text(color="black", size=16, face="bold"),
      axis.title.y = element_text(color="black", size=16, face="bold"),
      axis.text.x = element_text(size=14, face = 'bold'),
      axis.text.y = element_text(size=14, face = 'bold'),
      axis.line = element_line(color = 'black', size = 1)) +
    scale_color_manual(name = 'Cluster', values = scales::hue_pal()(num_clusters)) +
    guides(color = guide_legend(override.aes = list(size = 4)), 
           shape = guide_legend(override.aes = list(size = 3))) +
    geom_text_repel(aes(label = Compound), size = 2.5, max.overlaps = Inf, 
                    box.padding = 0.5, point.padding = 0.3, show.legend = FALSE, fontface = "bold")
  p
}

## seeds for reproducible results of Figure 3 -----
#
# for t-SNE (perplexity always equal to 3) 
#
seedCE_tSNE <- 163 # seed for Category Embedding
seedAI_tSNE <- 186 # seed for AutoInt Embedding
seedTT_tSNE <- 80 # seed for TabTransformer Embedding
#
# for UMAP (n=3, d=0.05 always) 
#
seedCE_UMAP <- 17 # seed for Category Embedding
seedAI_UMAP <- 160 # seed for AutoInt Embedding
seedTT_UMAP <- 156 # seed for TabTransformer Embedding

#### reads embedding files and combines with drug information -----
## Files for different embedding methods  
embCE <- 'CE_embedCompoundB.csv'
embAI <- 'AI_embedCompoundB.csv'
embTT <- 'TT_embedCompoundB.csv'
### Files for combination data with synergy predictions
combdataCE <- 'CE_synpredS.txt'
combdataAI <- 'AI_synpredS.txt'
combdataTT <- 'TT_synpredS.txt'
#
## example: comparing AutoInt against TabTransformer embeddings
inputemb1 <- embAI
seed1 <- seedAI_tSNE
combdata1 <- combdataAI

inputemb2 <- embTT
combdata2 <- combdataTT
seed2 <- seedTT_tSNE
# 
pathindata <- paste0(pathname,'data for Figure 3/') # directory with embedding, drug and synergy data
fname1 <- paste0(pathindata,inputemb1,collapse = NULL)
fname2 <- paste0(pathindata,inputemb2,collapse = NULL)
# Read the files with embedings and drug info data
filenamedrugs <- 'Drug_info_action.txt'
fnamedrugs <- paste0(pathindata,filenamedrugs,collapse = NULL)

embeds_raw1 <- read.csv(fname1, header = TRUE)
embeds_raw2 <- read.csv(fname2, header = TRUE)
druginfo <- read.table(file = fnamedrugs, header = TRUE,sep="\t",dec=".")

# Adds drug info to embedding data and reorder columns to place Target, Function and Pathway after Compound
embedswi1 <- embeds_raw1 %>%
  left_join(druginfo, by = c("Compound.B" = "Compound")) %>% rename(Compound = Compound.B) %>%
  select(Compound, Target, Function, Pathway, everything())
embedswi2 <- embeds_raw2 %>%
  left_join(druginfo, by = c("Compound.B" = "Compound")) %>% rename(Compound = Compound.B) %>%
  select(Compound, Target, Function, Pathway, everything())


# Extracts drug info and embedding matrices separately
drug_name <- embedswi1[[1]]  # First column: compound names
target <- as.factor(embedswi1[[2]])  # Second column: categorical label_1
Function <- as.factor(embedswi1[[3]])  # Third column: categorical label_2
pathway <- as.factor(embedswi1[[4]])  # 4th column: categorical label_3
embedding_data1 <- as.matrix(embedswi1[, -c(1, 2, 3, 4)])
embedding_data2 <- as.matrix(embedswi2[, -c(1, 2, 3, 4)])

### Performs dimensionality reduction, clustering, re-labels clusters and calculates rand index --------

## Calculates 2 main t-SNE components and clusters for both embeddings 
perplexity <- 3
num_clusters <- 8
# Run t-SNE and clustering for both datasets
result1 <- run_tsne(embedding_data1, drug_name, target, Function, pathway, seed1, perplexity) 
result1 <- run_kmeans(result1, num_clusters)

result2 <- run_tsne(embedding_data2, drug_name, target, Function, pathway, seed2, perplexity)
result2 <- run_kmeans(result2, num_clusters)

# Reassign cluster labels in embedding2 to match best overlap with clusters in first embedding
aligned_result2 <- align_clusters_LSAP(result1, result2, num_clusters)

adjrand_value <- adj_rand_index(result1, aligned_result2)
rand_value <- rand_index(result1, aligned_result2)

cat("Adjusted Rand Index between clusterings:", adjrand_value, "\n")
cat("Rand Index between clusterings:", rand_value, "\n")

# Plot both t-SNE cluster results
plot_clusters(result1, aligned_result2, num_clusters)

## Plot separate clusters of embeddings for Figure 3 ----------
#clust2plot <- aligned_result2
clust2plot <- result1
plot_1cluster(clust2plot,num_clusters)

### Extracts drug combinations and synergy scores (experimental and predicted) corresponding to each cluster -----
fnamecdata1 <- paste0(pathindata,combdata1)
fnamecdata2 <- paste0(pathindata,combdata2)

datacomb1 <- read.table(file = fnamecdata1, header = TRUE,sep="\t",dec=".")
datacomb2 <- read.table(file = fnamecdata2, header = TRUE,sep="\t",dec=".")

## joins drug combinatiom data including experimental and predicted synergy scores with drug info
drugdata1 <- process_data_CompB(datacomb1, druginfo)
drugdata2 <- process_data_CompB(datacomb2, druginfo)
all_clusters_data1 <- assign_clusters(drugdata1, result1)
all_clusters_data2 <- assign_clusters(drugdata2, result2)
## Plots distributions of experimental synergy scores by clusters for a given clustering ------
#
all_clusters_data <- all_clusters_data1
# find significant statistical differences of predictions between clusters
stat.test <- all_clusters_data %>%
  wilcox_test(Score_z_exp ~ Cluster) %>%
  adjust_pvalue(method = "BH") %>%
  add_significance("p.adj")
print(as.data.frame(stat.test))
significant_comparisons <- stat.test %>%
  filter(p.adj < 1e-7)
print(as.data.frame(significant_comparisons))
sigbar_pos <- seq(2.7, 4.5, length.out = nrow(significant_comparisons))
print(sigbar_pos)

# ggplot(all_clusters_data, aes(x = Cluster, y = Score_z_exp)) +
#   geom_boxplot(notch = TRUE, aes(fill=Cluster)) +  # Add notches to boxplots. Make local aestethics into geom_boxplot
ggplot(all_clusters_data, aes(x = Cluster, y = Score_z_exp)) +
  geom_boxplot(notch = TRUE, width = 0.2, alpha=1, aes(fill=Cluster)) +  # Add notches to boxplots. Make local aestethics into geom_boxplot
  geom_violin(alpha = 0.3, color = NA, scale = 'width', aes(fill=Cluster)) +
  geom_jitter(aes(color = Cluster), width = 0.2, alpha = 0.4) +  # Add jittered points
  geom_hline(yintercept=median(all_clusters_data$Score_z_exp), linetype='dashed', color='black', size=1) +
  scale_color_manual(values = scales::hue_pal()(num_clusters)) +
  scale_x_discrete(labels = 1:num_clusters) +  # Custom x-axis labels
  scale_y_continuous(breaks = c(-2, 0, 2, 4), labels = c(-2, 0, 2, 4), limits = c(-2.5, 4.5)) + # Custom y-axis labels
  stat_pvalue_manual(significant_comparisons,
                     label = "p.adj.signif",
                     bracket.size = 0.5,
                     size = 5,
                     vjust = 0.5,
                     y.position = sigbar_pos,
                     tip.length = 0.02) +
  labs(#title = 'Compund B t-SNE. Predictions, all data',
    x = "Cluster",
    y = "Experimental z-Synergy Scores") +
  theme_classic() +
  theme(axis.title.x = element_text(color="black", size=14, face="bold"),
        axis.title.y = element_text(color="black", size=14, face="bold"),
        axis.text.x=element_text(size=12, face = 'bold'),
        axis.text.y=element_text(size=12, face = 'bold'),
        axis.line = element_line(color = 'black', size = 1),
        legend.position="none")