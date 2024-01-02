library(readxl)
library(dplyr)
library(writexl)

data1 <- read_xlsx("DATA_PATH")
# Categorize the Ratings
data1 <- data1 %>%
  mutate(
    RatingLabel = case_when(
      Ratings <= 5.0 ~ "Poor",
      Ratings <= 6.4 ~ "Average",
      Ratings <= 7.2 ~ "Good",
      Ratings <= 10 ~ "Excellent",
      TRUE ~ NA_character_  # for any ratings outside the specified range or NA
    ),
    # Categorize the Gross
    GrossLabel = case_when(
      Gross < 900000 ~ "Flop",
      Gross >= 900000  & Gross < 20000000 ~ "Average",
      Gross >= 20000000 & Gross < 100000000 ~ "Success",
      Gross >= 100000000 ~ "Block-Buster",
      TRUE ~ NA_character_  # for any gross outside the specified range or NA
    )
  )

write_xlsx(data1, "DATA_PATH")