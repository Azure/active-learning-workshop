library(dplyr)
library(ggplot2)
library(tidyr)
library(pROC)

source("active_learning_lib_reticulate.R")

set.seed(1) # for consistent test set

FEATURIZED_DATA_FILE <- "attacks_use_encoded.Rds" #featurized_wiki_comments_attack.Rds"
if (!file.exists(FEATURIZED_DATA_FILE)){
  storage_address <- 'https://altutorialweu.blob.core.windows.net/activelearningdemo/'
  download_url <- paste0(storage_address, FEATURIZED_DATA_FILE)
  download.file(download_url, FEATURIZED_DATA_FILE)
}

TEST_SET_SIZE <- 10000

FEATURIZED_DATA <- readRDS(FEATURIZED_DATA_FILE)
FILE_TAG <- "lang_lc"

FULL_DATASET_RESULTS_FILE <- sprintf("full_fit_list_%s.Rds", FILE_TAG)
ACTIVE_LEARNING_RESULTS_FILE <- sprintf("active_learning_res_%s.Rds", FILE_TAG)
PASSIVE_LEARNING_RESULTS_FILE <- sprintf("passive_learning_res_%s.Rds", FILE_TAG)

FEATURIZED_DATA <- FEATURIZED_DATA[complete.cases(FEATURIZED_DATA),]

test_set_ids <- sample(FEATURIZED_DATA$rev_id, TEST_SET_SIZE)
TEST_SET <- FEATURIZED_DATA %>% filter(rev_id %in% test_set_ids)

unlabeled_data_df <- FEATURIZED_DATA[!(FEATURIZED_DATA$rev_id %in% TEST_SET$rev_id),]

inputs <- grep("^V", names(FEATURIZED_DATA), value=TRUE)
outcome <- "flagged"
FORM <- paste0(outcome, ' ~ ', paste(inputs, collapse="+"), " - 1")


# params <- list(seed=1, initial_examples_per_class=20, examples_to_label_per_iteration=20, num_iterations=20, presample_size=20000, monte_carlo_samples=10, mu=0.5, sigma=0.1)

NUM_ITERATIONS <- 50L
NUM_REPS <- 5
  
### repeated fittings to full dataset
if (file.exists(FULL_DATASET_RESULTS_FILE)){
  full_fit_list <- readRDS(FULL_DATASET_RESULTS_FILE)
} else {
  full_fit_list <- lapply(1:NUM_REPS, function(i) fit_and_evaluate_model(unlabeled_data_df))
  saveRDS(full_fit_list, FULL_DATASET_RESULTS_FILE)
}

if (file.exists(ACTIVE_LEARNING_RESULTS_FILE)){
  active_learning_res <- readRDS(ACTIVE_LEARNING_RESULTS_FILE)
} else {
  param_table <- expand.grid(seed=(1:NUM_REPS), 
                             initial_examples_per_class=25, 
                             examples_to_label_per_iteration=25, 
                             num_iterations=NUM_ITERATIONS, 
                             presample_size=20000, 
                             mu=0.5, sigma=0.1)
  
  active_learning_res <- 1:nrow(param_table) %>%
    lapply(function(i) run_active_learning_curve(param_table[i,], unlabeled_data_df, FEATURIZED_DATA)) %>% 
    bind_rows
  
  active_learning_res$mode <- 'active'
  
  saveRDS(active_learning_res, ACTIVE_LEARNING_RESULTS_FILE)
}

if (file.exists(PASSIVE_LEARNING_RESULTS_FILE)){
  passive_learning_res <- readRDS(PASSIVE_LEARNING_RESULTS_FILE)
} else {
  tss <- active_learning_res$tss %>% unique %>% sort
  
  params <- list(initial_examples_per_class=25, 
                 examples_to_label_per_iteration=25, 
                 num_iterations=NUM_ITERATIONS, presample_size=20000, 
                 mu=0.5, sigma=0.1)
  
  passive_learning_res <- lapply(1:NUM_REPS, run_passive_learning_curve, tss, unlabeled_data_df, FEATURIZED_DATA, params)
  passive_learning_res <- passive_learning_res %>% bind_rows
  
  saveRDS(passive_learning_res, PASSIVE_LEARNING_RESULTS_FILE)
}

common_columns <- c('tss','auc', 'accuracy','seed', 'mode')

learning_curve_data <- rbind(active_learning_res[common_columns], passive_learning_res[common_columns])

auc_vec_full <- sapply(full_fit_list, function(f) f$performance[['auc']])
g_auc <- learning_curve_data %>% ggplot(aes(x=tss, y=auc, group=interaction(factor(seed), factor(mode)), col=mode)) + 
  geom_line(size=1.2, alpha=0.5) + 
  ggtitle(sprintf("AUC, active vs random (%d iterations)", NUM_ITERATIONS)) + 
  geom_hline(yintercept=auc_vec_full, size=0.25, alpha=0.5)
ggsave(sprintf("AUC_%03d_iterations.png", NUM_ITERATIONS), g_auc, device="png")

accuracy_vec_full <- sapply(full_fit_list, function(f) f$performance[['accuracy']])
g_acc <- learning_curve_data %>% ggplot(aes(x=tss, y=accuracy, group=interaction(factor(seed), factor(mode)), col=mode)) + 
  geom_line(size=1.2, alpha=0.5) + 
  ggtitle(sprintf("Accuracy, active vs random selection (%d iterations)", NUM_ITERATIONS)) + 
  geom_hline(yintercept=accuracy_vec_full)
ggsave(sprintf("Accuracy_%03d_iterations.png", NUM_ITERATIONS), g_acc, device="png")

