library(readxl)
library(ggplot2)
library(corrplot)
library(dplyr)
library(GGally)

data = read_excel("DATA_PATH")

numeric_data <- data[, sapply(data, is.numeric)]
numeric_data_without_year_genre <- numeric_data[, !(names(numeric_data) %in% c("Year", "Genre"))]
cor_matrix <- cor(numeric_data_without_year_genre, use = "complete.obs")

# Plot the correlation heatmap
corrplot::corrplot(cor_matrix, method = "color")


data$RatingLabel <- factor(data$RatingLabel, levels = c("Poor", "Average", "Good", "Excellent"))

ggplot(data, aes(x = RatingLabel, fill = RatingLabel)) +
  geom_bar() +
  labs(title = "Count of Movies by Rating Label", x = "Rating Label", y = "Count") +
  theme_minimal()

ggplot(data, aes(x = Budget, y = Gross)) +
  geom_point(aes(color = GrossLabel)) + # Color points by GrossLabel
  geom_smooth(method = "lm", se = FALSE, color = "black") + # Add a linear regression line
  labs(title = "Scatter Plot of Budget vs Gross", x = "Budget", y = "Gross") +
  theme_minimal()


ggplot(data, aes(x = Ratings)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.1, fill = "#69b3a2", color = "black") +
  geom_density(alpha = 0.7, fill = "#ff6666") +
  labs(title = "Distribution of Movie Ratings with KDE", 
       x = "Ratings", 
       y = "Density") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.x = element_text(face = "bold", color = "grey20", size = 12),
    axis.title.y = element_text(face = "bold", color = "grey20", size = 12),
    legend.position = "none"
  )

data$GrossLabel <- factor(data$GrossLabel, levels = c("Flop", "Average", "Success", "Block-Buster"))

# Now create the bar plot
ggplot(data, aes(x = GrossLabel, fill = GrossLabel)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") + 
  labs(title = "Count of Movies by Gross Label", x = "Gross Label", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none") 


library(scales) 

data$GrossLabel <- factor(data$GrossLabel, levels = c("Flop", "Average", "Success", "Block-Buster"))

ggplot(data, aes(x = Gross)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.1, fill = "grey70", alpha = 0.3) +
  geom_density(aes(fill = GrossLabel), alpha = 0.6) +
  scale_x_log10(labels = scales::comma) +
  scale_fill_manual(values = c("Flop" = "red", "Average" = "yellow", "Success" = "blue", "Block-Buster" = "green")) +
  geom_area(stat = "density", aes(fill = GrossLabel), alpha = 0.2, position = "identity") +
  labs(title = "Continuous Distribution of Gross Earnings with KDE",
       x = "Gross Earnings ($)",
       y = "Density") +
  theme_minimal() +
  theme(legend.position = "right") +
  annotate("rect", xmin = c(0, 1e6, 1e7, 1e8), xmax = c(1e6, 1e7, 1e8, Inf), ymin = -Inf, ymax = 0, alpha = 0.3, fill = c("red", "yellow", "blue", "green"))

