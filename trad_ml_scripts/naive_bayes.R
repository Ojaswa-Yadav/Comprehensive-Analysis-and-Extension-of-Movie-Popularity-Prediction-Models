library(tidyverse)
library(dplyr)
library(caret)
library(readxl)
library(ggplot2)
library(corrplot)
library(dplyr)
library(GGally)
library(e1071)

data <- read_xlsx("DATA_PATH")

standardize_columns <- function(df) {
  numeric_columns <- sapply(df, is.numeric)
  df[, numeric_columns] <- scale(df[, numeric_columns])
  return(df)
}


preprocess_data <- function(df) {
  numeric_columns <- sapply(df, is.numeric)
  categorical_columns <- sapply(df, is.factor)
  
  # Fill NA in numeric columns with median
  df[, numeric_columns] <- lapply(df[, numeric_columns], function(x) {
    ifelse(is.na(x), median(x, na.rm = TRUE), x)
  })
  
  # Fill NA in categorical columns with mode (most common value)
  df[, categorical_columns] <- lapply(df[, categorical_columns], function(x) {
    ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x)
  })
  
  return(df)
}


set.seed(100)

conventional_features <- data[, c("Genre", "Budget", "Screens", "Sequel", "GrossLabel", "RatingLabel")]
social_media_features <- data[, c("Sentiment", "Views", "Likes", "Dislikes", "Comments", "Aggregate Followers", "GrossLabel", "RatingLabel")]

train_size_conv <- floor(0.8 * nrow(conventional_features))
train_size_social <- floor(0.8 * nrow(social_media_features))

train_index_conv <- sample(seq_len(nrow(conventional_features)), size = train_size_conv)
train_index_social <- sample(seq_len(nrow(social_media_features)), size = train_size_social)

train_conv <- conventional_features[train_index_conv, ]
test_conv <- conventional_features[-train_index_conv, ]

train_social <- social_media_features[train_index_conv, ]
test_social <- social_media_features[-train_index_conv, ]


train_conv <- preprocess_data(train_conv)
test_conv <- preprocess_data(test_conv)
train_social <- preprocess_data(train_social)
test_social <- preprocess_data(test_social)


make_numeric <- function(df) {
  numeric_df <- data.frame(lapply(df, function(x) {
    if (is.factor(x)) {
      as.numeric(as.factor(x))  # Convert factors to numeric
    } else if (is.character(x)) {
      as.numeric(factor(x))  # Convert characters to factors, then to numeric
    } else {
      x  # Keep numeric as is
    }
  }))
  return(numeric_df)
}

gross_label_mapping <- c('1' = 'Average', '2' = 'Blockbuster', '3' = 'Flop', '4' = 'Succuss')  # Example mapping
rating_label_mapping <- c('1' = 'Average', '2' = 'Excellent', '3' = 'Good', '4' = 'Poor')  # Example mapping


train_conv_features <- train_conv[, !names(train_conv) %in% c("GrossLabel", "RatingLabel")]
test_conv_features <- test_conv[, !names(test_conv) %in% c("GrossLabel", "RatingLabel")]

train_social_features <- train_social[, !names(train_social) %in% c("GrossLabel", "RatingLabel")]
test_social_features <- test_social[, !names(test_social) %in% c("GrossLabel", "RatingLabel")]

train_conv$GrossLabel <- as.factor(train_conv$GrossLabel)
train_conv$RatingLabel <- as.factor(train_conv$RatingLabel)
train_social$GrossLabel <- as.factor(train_social$GrossLabel)
train_social$RatingLabel <- as.factor(train_social$RatingLabel)

#train_conv$Sequel<- as.factor(train_conv$Sequel)
#train_conv$Genre<- as.factor(train_conv$Genre)
#train_social$Sequel<- as.factor(train_social$Sequel)
#train_social$Genre<- as.factor(train_social$Genre)


train_full <- cbind(train_conv[, !names(train_conv) %in% "RatingLabel"], 
                    train_social[, !names(train_social) %in% "GrossLabel"])

train_full$GrossLabel <- train_conv$GrossLabel
train_full$RatingLabel <- train_conv$RatingLabel

train_full$RatingLabel <- train_conv$RatingLabel# Combine the conventional and social media features for testing, without the target variable
test_full_features <- cbind(test_conv_features, test_social_features)
test_full_features$RatingLabel <- test_conv$RatingLabel
test_full_features$GrossLabel <- test_conv$GrossLabel

train_conv <- standardize_columns(train_conv)
test_conv <- standardize_columns(test_conv)
train_social <- standardize_columns(train_social)
test_social <- standardize_columns(test_social)
train_full <- standardize_columns(train_full)
test_full_features <- standardize_columns(test_full_features)
naive_bayes_classification <- function(train_data, test_data, target_variable) {
  model <- naiveBayes(as.formula(paste(target_variable, "~ .")), data = train_data)
  
  predictions <- predict(model, test_data)
  
  return(predictions)
}

predictions_gross_conv <- naive_bayes_classification(train_conv, test_conv, "GrossLabel")

predictions_gross_social <- naive_bayes_classification(train_social, test_social, "GrossLabel")

predictions_rating_conv <- naive_bayes_classification(train_conv, test_conv, "RatingLabel")

predictions_rating_social <- naive_bayes_classification(train_social, test_social, "RatingLabel")

predictions_gross_full <- naive_bayes_classification(train_full, test_full_features, "GrossLabel")

predictions_rating_full <- naive_bayes_classification(train_full, test_full_features, "RatingLabel")



plot_confusion_matrix <- function(cm, title) {
  # Convert the matrix to a data frame for plotting
  cm_df <- as.data.frame(cm)
  
  # Reshape the data frame from wide to long format
  cm_long <- reshape2::melt(cm_df, varnames = c("Predicted", "Actual"), value.name = "Frequency")
  
  # Plot the confusion matrix
  ggplot(data = cm_long, aes(x = Predicted, y = Actual, fill = Frequency)) +
    geom_tile(color = "white") +  # Use white lines to separate the tiles
    geom_text(aes(label = Frequency), vjust = 1) +  # Add text in each tile
    scale_fill_gradient(low = "white", high = "steelblue") +  # Gradient color for the tiles
    labs(title = title, x = "Predicted", y = "Actual") +  # Labels for the plot
    theme_minimal() +  # Minimal theme
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis text for better readability
}


cm_gross_conv <- table(Predicted = predictions_gross_conv, Actual = test_conv$GrossLabel)
plot_confusion_matrix(cm_gross_conv, "Confusion Matrix for GrossLabel\n(Conventional Features)")

cm_gross_social <- table(Predicted = predictions_gross_social, Actual = test_social$GrossLabel)
plot_confusion_matrix(cm_gross_social, "Confusion Matrix for GrossLabel\n(Social Media Features)")

cm_gross_full <- table(Predicted = predictions_gross_full, Actual = test_full_features$GrossLabel)
plot_confusion_matrix(cm_gross_full, "Confusion Matrix for GrossLabel\n(Full Feature Set)")

cm_rating_conv <- table(Predicted = predictions_rating_conv, Actual = test_conv$RatingLabel)
plot_confusion_matrix(cm_rating_conv, "Confusion Matrix for RatingLabel\n(Conventional Features)")

cm_rating_social <- table(Predicted = predictions_rating_social, Actual = test_social$RatingLabel)
plot_confusion_matrix(cm_rating_social, "Confusion Matrix for RatingLabel\n(Social Media Features)")

cm_rating_full <- table(Predicted = predictions_rating_full, Actual = test_full_features$RatingLabel)
plot_confusion_matrix(cm_rating_full, "Confusion Matrix for RatingLabel\n(Full Feature Set)")



common_levels <- intersect(levels(factor(predictions_gross_conv)), levels(factor(test_conv$GrossLabel)))

predictions_gross_conv_factor <- factor(predictions_gross_conv, levels = common_levels)
test_conv_gross_label_factor <- factor(test_conv$GrossLabel, levels = common_levels)

calculate_metrics <- function(predictions, actual) {
  cm <- table(Predicted = predictions, Actual = actual)

  # Calculate overall accuracy
  accuracy <- sum(diag(cm)) / sum(cm)

  # Initialize vectors to store precision and recall for each class
  precision <- rep(NA, ncol(cm))
  recall <- rep(NA, nrow(cm))
  
  # Calculate precision and recall for each class
  for (i in 1:ncol(cm)) {
    if (sum(cm[, i]) != 0) {
      precision[i] <- cm[i, i] / sum(cm[, i])
    }
    if (sum(cm[i, ]) != 0) {
      recall[i] <- cm[i, i] / sum(cm[i, ])
    }
  }

  # Calculate F1 score
  f1 <- ifelse((precision + recall) != 0, 2 * (precision * recall) / (precision + recall), NA)

  mean_precision <- mean(precision, na.rm = TRUE)
  mean_recall <- mean(recall, na.rm = TRUE)
  mean_f1 <- mean(f1, na.rm = TRUE)

  return(data.frame(Accuracy = accuracy, Precision = mean_precision, Recall = mean_recall, F1 = mean_f1))
}

metrics_gross_conv <- calculate_metrics(predictions_gross_conv_factor, test_conv_gross_label_factor)# Print the metrics

metrics_gross_social <- calculate_metrics(predictions_gross_social, test_social$GrossLabel)
metrics_gross_full <- calculate_metrics(predictions_gross_full, test_full_features$GrossLabel)

metrics_rating_conv <- calculate_metrics(predictions_rating_conv, test_conv$RatingLabel)
metrics_rating_social <- calculate_metrics(predictions_rating_social, test_social$RatingLabel)
metrics_rating_full <- calculate_metrics(predictions_rating_full, test_full_features$RatingLabel)

metrics <- rbind(
  cbind(Model = "GrossLabel - Conventional", metrics_gross_conv),
  cbind(Model = "GrossLabel - Social", metrics_gross_social),
  cbind(Model = "GrossLabel - Combined", metrics_gross_full),
  cbind(Model = "RatingLabel - Conventional", metrics_rating_conv),
  cbind(Model = "RatingLabel - Social", metrics_rating_social),
  cbind(Model = "RatingLabel - Combined", metrics_rating_full)
)

print(metrics)



metrics_df <- data.frame(
  Group = rep(c("GrossLabel", "RatingLabel"), each = 3),
  Model = c(
    "Conventional", "Social", "Combined",
    "Conventional", "Social", "Combined"
  ),
  Accuracy = c(
    metrics_gross_conv[1, "Accuracy"], 
    metrics_gross_social[1, "Accuracy"], 
    metrics_gross_full[1, "Accuracy"], 
    metrics_rating_conv[1, "Accuracy"], 
    metrics_rating_social[1, "Accuracy"], 
    metrics_rating_full[1, "Accuracy"]
  )
)

metrics_df$Model <- factor(metrics_df$Model, levels = c("Social", "Conventional", "Combined"))

dodge <- position_dodge(width = 0.9)
ggplot(metrics_df, aes(x = Group, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = dodge) +
  geom_text(aes(label = Model, y = 0.03), position = dodge, vjust = -1.1, angle=45, color = "white", size = 4) +
  scale_fill_manual(values = c("Social" = "blue3", "Conventional" = "red3", "Combined" = "green3")) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none",  
        panel.spacing = unit(1.5, "lines")) 
