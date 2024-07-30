#install.packages("tidyverse")
library(tidyverse)
#install.packages("dplyr")
library(dplyr)
#install.packages("nflreadr")
library(nflreadr)
#install.packages("caret")
library(caret)
#install.packages("xgboost")
library(xgboost)
#install.packages("Matrix")
library(Matrix)
#install.packages("nflfastr")
library(nflfastR)
#install.packages("ggimage")
library(ggimage)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("scales")
library(scales)
#install.packages("extrafont")
library(extrafont)
#install.packages("gt")
library(gt)
#install.packages("gtExtras")
library(gtExtras)
#install.packages("ggthemes")
library(ggthemes)
#install.packages("ggsci")
library(ggsci)

sterb_analytics_theme <- function(..., base_size = 12) {
  
  theme(
    text = element_text(family = "Bahnschrift", size = base_size),
    axis.ticks = element_blank(),
    axis.title = element_text(color = "black",
                              face = "bold"),
    axis.text = element_text(color = "black",
                             face = "bold",
                             size = base_size),
    plot.title.position = "plot",
    plot.title = element_text(size = base_size * 1.52,
                              face = "bold",
                              color = "black",
                              vjust = .02,
                              hjust = 0.08),
    plot.subtitle = element_text(size = base_size * 1.08,
                                 color = "black",
                                 hjust = 0.08),
    plot.caption = element_text(size = base_size,
                                color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major =  element_line(color = "#cccccc"),
    panel.background = element_rect(fill = "#f8f8f8"),
    plot.background = element_rect(fill = "#ffffff"),
    panel.border = element_blank())
}

# load and join datasets
ultStats <- nflreadr::load_participation(2023, include_pbp = TRUE) %>% 
  left_join(nflreadr::load_ftn_charting(2023), by = c("nflverse_game_id" = "nflverse_game_id",
                                                      "play_id" = "nflverse_play_id",
                                                      "week" = "week", "season" = "season"))

# step A: cumulative game totals
stepA <- ultStats %>%
  filter(!is.na(down), !is.na(posteam), !is.na(defteam)) %>% 
  mutate(total_posteam_epa = ifelse(posteam_type == "home", total_home_epa, total_away_epa),
         total_defteam_epa = -total_posteam_epa) %>%
  mutate(total_posteam_rush_epa = ifelse(posteam_type == "home", total_home_rush_epa, total_away_rush_epa),
         total_posteam_pass_epa = ifelse(posteam_type == "home", total_home_pass_epa, total_away_pass_epa),
         total_posteam_comp_air_epa = ifelse(posteam_type == "home", total_home_comp_air_epa, total_away_comp_air_epa),
         total_posteam_comp_yac_epa = ifelse(posteam_type == "home", total_home_comp_yac_epa, total_away_comp_yac_epa),
         total_posteam_raw_air_epa = ifelse(posteam_type == "home", total_home_raw_air_epa, total_away_raw_air_epa),
         total_posteam_raw_yac_epa = ifelse(posteam_type == "home", total_home_raw_yac_epa, total_away_raw_yac_epa),
         posteam_wp = wp,
         defteam_wp = 1 - posteam_wp)

# step B: select relevant columns
stepB <- stepA %>%
  ungroup() %>% 
  summarize(season, posteam, defteam, defense_man_zone_type, offense_formation,
            offense_personnel, down, ydstogo, defenders_in_box, defense_personnel,
            yardline_100, quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining,
            game_half, drive, qtr, goal_to_go, time, posteam_timeouts_remaining, defteam_timeouts_remaining,
            posteam_score, defteam_score, score_differential, fixed_drive, drive_start_transition,
            drive_game_clock_start, spread_line, temp, wind, xpass, starting_hash, qb_location,
            n_offense_backfield, n_defense_box,
            total_posteam_epa, total_defteam_epa, total_posteam_rush_epa, total_posteam_pass_epa,
            total_posteam_comp_air_epa, total_posteam_comp_yac_epa, total_posteam_raw_air_epa,
            total_posteam_raw_yac_epa, posteam_wp, defteam_wp)

# step C: Clean and filter data
stepC <- stepB %>% 
  mutate(wind = ifelse(is.na(wind), 0, wind),
         temp = ifelse(is.na(temp), 70, temp)) %>% 
  filter(!is.na(defense_man_zone_type), !is.na(offense_formation), 
         !is.na(offense_personnel), !is.na(down), !is.na(ydstogo), !is.na(defenders_in_box),
         !is.na(defense_personnel), !is.na(yardline_100), !is.na(quarter_seconds_remaining),
         !is.na(half_seconds_remaining), !is.na(game_seconds_remaining), !is.na(game_half),
         !is.na(drive), !is.na(qtr), !is.na(goal_to_go), !is.na(posteam_timeouts_remaining),
         !is.na(defteam_timeouts_remaining), !is.na(posteam_score), !is.na(defteam_score),
         !is.na(score_differential), !is.na(fixed_drive), !is.na(drive_start_transition),
         !is.na(spread_line), !is.na(temp), !is.na(wind), !is.na(xpass), !is.na(starting_hash),
         !is.na(qb_location), !is.na(n_offense_backfield), !is.na(n_defense_box),
         !is.na(total_posteam_epa), !is.na(total_defteam_epa), !is.na(total_posteam_rush_epa),
         !is.na(total_posteam_pass_epa), !is.na(total_posteam_comp_air_epa),
         !is.na(total_posteam_comp_yac_epa), !is.na(total_posteam_raw_air_epa),
         !is.na(total_posteam_raw_yac_epa), !is.na(posteam_wp), !is.na(defteam_wp))

# define factor columns
factor_cols <- c("offense_formation", "offense_personnel", "down", "defenders_in_box",
                    "defense_personnel", "game_half", "drive", "qtr", "goal_to_go", 
                    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "fixed_drive", 
                    "drive_start_transition", "starting_hash", "qb_location", 
                    "n_offense_backfield", "n_defense_box")

# create dataset for factors only
factors <- stepC %>%
  select(all_of(factor_cols))

# one-hot encoding for factors
dmy <- dummyVars(" ~ .", data = factors)
factors_encoded <- data.frame(predict(dmy, newdata = factors))

# combine encoded factors with the non-factor columns
non_factors <- stepC %>% 
  select(-all_of(factor_cols))

# merge the factors back with non-factor columns
expManZone_encoded <- cbind(non_factors, factors_encoded)

# ensure 'defense_man_zone_type' is factor
expManZone_encoded$defense_man_zone_type <- as.factor(expManZone_encoded$defense_man_zone_type)

# separate target variable from predictors
target <- expManZone_encoded$defense_man_zone_type
predictors <- expManZone_encoded[, !names(expManZone_encoded) %in% c("defense_man_zone_type", "season", "posteam", "defteam")]

# convert predictors to numeric matrix
predictors_numeric <- data.matrix(predictors)

# set seed for repricability
set.seed(3)

# train/test split
trainIndex <- createDataPartition(expManZone_encoded$defense_man_zone_type, p = 0.8, list = FALSE, times = 1)
trainData <- predictors_numeric[trainIndex,]
testData <- predictors_numeric[-trainIndex,]
train_labels <- as.numeric(target[trainIndex]) - 1
test_labels <- as.numeric(target[-trainIndex]) - 1

# create DMatrix for XGBoost
train_matrix <- xgb.DMatrix(data = trainData, label = train_labels)
test_matrix <- xgb.DMatrix(data = testData, label = test_labels)

# define parameters for model
params <- list(
  objective = "multi:softprob",
  num_class = 2, # since we have MAN_COVERAGE and ZONE_COVERAGE
  eval_metric = "mlogloss",
  max_depth = 6,
  eta = 0.1,
  nthread = 2
)

# train model
xgb_model <- xgb.train(params, train_matrix, nrounds = 1000, watchlist = list(train = train_matrix, eval = test_matrix), early_stopping_rounds = 10)

# predict and evaluate
pred_probs <- predict(xgb_model, test_matrix)
pred_labels <- max.col(matrix(pred_probs, ncol = 2)) - 1

# importance matrix
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)

importance_matrix_df <- as.data.frame(importance_matrix)

top10feats <- importance_matrix_df %>% 
  arrange(-Gain) %>% 
  ungroup() %>% 
  mutate(rk_gain = rank(-Gain)) %>% 
  filter(rk_gain <= 10)

importanceMatrTable <- gt(top10feats, rowname_col = "rk_gain") %>% 
  tab_header(
    title = "Top 10 Features by Gain",
    subtitle = "Expected Man/Zone Coverage XGBoost Model"
  ) %>% 
  fmt_number(
    columns = vars(Gain, Cover, Frequency),
    decimals = 4
  ) %>% 
  cols_align(
    align = 'center',
    columns = vars(Feature, Gain, Cover, Frequency)
  ) %>% 
  data_color(
    columns = vars(Gain, Cover, Frequency),
    colors = col_numeric(
      palette = pal_material("blue")(10),
      domain = NULL
    )
  ) %>%
  gt_theme_538()

importanceMatrTable

#gtsave(importanceMatrTable, "top-10-feats-gain.png")

# convert numeric predictions back to factor levels (man/zone)
pred_factor <- factor(pred_labels, levels = c(0, 1), labels = c("MAN_COVERAGE", "ZONE_COVERAGE"))

# confusion matrix
confusionMatrix(pred_factor, factor(test_labels, levels = c(0, 1), labels = c("MAN_COVERAGE", "ZONE_COVERAGE")))

# get the probabilities for each coverage type
pred_prob_matrix <- matrix(pred_probs, ncol = 2, byrow = TRUE)
colnames(pred_prob_matrix) <- c("MAN_COVERAGE", "ZONE_COVERAGE")
pred_prob_df <- as.data.frame(pred_prob_matrix)

# view predicted probabilities
head(pred_prob_df)

# append predictions to original dataset
full_pred_probs <- predict(xgb_model, xgb.DMatrix(data = predictors_numeric))
full_pred_prob_matrix <- matrix(full_pred_probs, ncol = 2, byrow = TRUE)
colnames(full_pred_prob_matrix) <- c("exp_man", "exp_zone")
full_pred_prob_df <- as.data.frame(full_pred_prob_matrix)

# add predicted probabilities to original dataset
expManZone_with_probs <- cbind(expManZone_encoded, full_pred_prob_df)

# calculate 'man_over_expected' and 'zone_over_expected'
expManZone_with_probs <- expManZone_with_probs %>%
  mutate(
    man_over_expected = ifelse(defense_man_zone_type == "MAN_COVERAGE", 1 - exp_man, -exp_man),
    zone_over_expected = ifelse(defense_man_zone_type == "ZONE_COVERAGE", 1 - exp_zone, -exp_zone)
  )

# calculate average and mean squared error (MSE) for 'man_over_expected' and 'zone_over_expected'
avgOverExps <- expManZone_with_probs %>% 
  summarise(
    mse_man = mean(man_over_expected^2),
    mse_zone = mean(zone_over_expected^2),
    avg_man_over_expected = mean(man_over_expected),
    avg_zone_over_expected = mean(zone_over_expected)
  ) %>% 
  unique()

print(avgOverExps)

#### Insights and Visualization ####

fSeasons = c(2023) # select seasons 2019-2023

fSeason_expManZone_with_probs <- expManZone_with_probs %>% 
  filter(season %in% fSeasons)

fSeasonsText <- toString(fSeasons)

# drawing team insights
posTeamOE <- fSeason_expManZone_with_probs %>% 
  ungroup() %>% 
  group_by(posteam) %>% 
  summarise(avg_man_oe = mean(man_over_expected),
            avg_zone_oe = mean(zone_over_expected))

defTeamOE <- fSeason_expManZone_with_probs %>% 
  ungroup() %>% 
  group_by(defteam) %>% 
  summarise(avg_man_oe = mean(man_over_expected),
            avg_zone_oe = mean(zone_over_expected))

# calculate mean epa for each coverage type
posteam_coverage_epa <- ultStats %>%
  filter(season %in% fSeasons) %>% 
  filter(!is.na(epa), !is.na(defense_man_zone_type)) %>%
  group_by(posteam, defense_man_zone_type) %>%
  summarise(epa = mean(epa), .groups = 'drop')

defteam_coverage_epa <- ultStats %>%
  filter(season %in% fSeasons) %>% 
  filter(!is.na(epa), !is.na(defense_man_zone_type)) %>%
  group_by(defteam, defense_man_zone_type) %>%
  summarise(epa = mean(epa), .groups = 'drop')

# separate man and zone coverage epa
posteam_man_epa <- posteam_coverage_epa %>%
  filter(defense_man_zone_type == "MAN_COVERAGE") %>%
  select(posteam, man_epa = epa)

defteam_man_epa <- defteam_coverage_epa %>%
  filter(defense_man_zone_type == "MAN_COVERAGE") %>%
  select(defteam, man_epa = epa)

posteam_zone_epa <- posteam_coverage_epa %>%
  filter(defense_man_zone_type == "ZONE_COVERAGE") %>%
  select(posteam, zone_epa = epa)

defteam_zone_epa <- defteam_coverage_epa %>%
  filter(defense_man_zone_type == "ZONE_COVERAGE") %>%
  select(defteam, zone_epa = epa)

# join back to main dataframe
posTeamManZoneEPA <- ultStats %>%
  filter(season %in% fSeasons) %>% 
  filter(!is.na(epa), !is.na(defense_man_zone_type)) %>%
  group_by(posteam) %>%
  summarise(epa = mean(epa), .groups = 'drop') %>%
  left_join(posteam_man_epa, by = "posteam") %>%
  left_join(posteam_zone_epa, by = "posteam")

defTeamManZoneEPA <- ultStats %>%
  filter(season %in% fSeasons) %>% 
  filter(!is.na(epa), !is.na(defense_man_zone_type)) %>%
  group_by(defteam) %>%
  summarise(epa = mean(epa), .groups = 'drop') %>%
  left_join(defteam_man_epa, by = "defteam") %>%
  left_join(defteam_zone_epa, by = "defteam")

posTeamEpaVSoe <- left_join(posTeamOE, posTeamManZoneEPA, by = "posteam") %>% 
  left_join(teams_colors_logos, by = c("posteam" = "team_abbr"))

defTeamEpaVSoe <- left_join(defTeamOE, defTeamManZoneEPA, by = "defteam") %>% 
  left_join(teams_colors_logos, by = c("defteam" = "team_abbr"))

# offenses facing man coverage
ggplot(data = posTeamEpaVSoe, aes(x = avg_man_oe,
                           y = man_epa)) +
  geom_hline(yintercept = mean(posTeamEpaVSoe$man_epa),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_vline(xintercept = mean(posTeamEpaVSoe$avg_man_oe),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_image(aes(image = team_logo_wikipedia), asp = 16/9) +
  scale_x_continuous(breaks = pretty_breaks(n = 8),
                     labels = percent_format()) +
  scale_y_continuous(breaks = pretty_breaks(n = 12)) +
  sterb_analytics_theme() +
  labs(x = "Average Man Coverage Seen Over Expectation",
       y = "Average EPA Against Man Coverage",
       title = paste0(fSeasonsText, " NFL Offenses Against Man Coverage: Rate Over Expectation vs. Efficiency"),
       subtitle = "Expectation via XGBoost Model",
       caption = "@EthanSterbis on X
       data via nflverse and FTN")

#ggsave('off-mc-vs-oe.jpeg', height = 8, width = 10, dpi = 'retina')

# defenses playing man coverage
ggplot(data = defTeamEpaVSoe, aes(x = avg_man_oe,
                                  y = man_epa)) +
  geom_hline(yintercept = mean(defTeamEpaVSoe$man_epa),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_vline(xintercept = mean(defTeamEpaVSoe$avg_man_oe),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_image(aes(image = team_logo_wikipedia), asp = 16/9) +
  scale_x_continuous(breaks = pretty_breaks(n = 12),
                     labels = percent_format()) +
  scale_y_continuous(breaks = pretty_breaks(n = 12)) +
  sterb_analytics_theme() +
  labs(x = "Average Man Coverage Over Expectation",
       y = "Average EPA in Man Coverage",
       title = paste0(fSeasonsText, " NFL Defenses Playing Man Coverage: Rate Over Expectation vs. Efficiency"),
       subtitle = "Expectation via XGBoost Model",
       caption = "@EthanSterbis on X
       data via nflverse and FTN")

#ggsave('def-mc-vs-oe.jpeg', height = 8, width = 10, dpi = 'retina')

# offenses facing zone coverage
ggplot(data = posTeamEpaVSoe, aes(x = avg_zone_oe,
                           y = zone_epa)) +
  geom_hline(yintercept = mean(posTeamEpaVSoe$zone_epa),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_vline(xintercept = mean(posTeamEpaVSoe$avg_zone_oe),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_image(aes(image = team_logo_wikipedia), asp = 16/9) +
  scale_x_continuous(breaks = pretty_breaks(n = 8),
                     labels = percent_format()) +
  scale_y_continuous(breaks = pretty_breaks(n = 8)) +
  sterb_analytics_theme() +
  labs(x = "Average Zone Coverage Seen Over Expectation",
       y = "Average EPA Against Zone Coverage",
       title = paste0(fSeasonsText, " NFL Offenses Against Zone Coverage: Rate Over Expectation vs. Efficiency"),
       subtitle = "Expectation via XGBoost Model",
       caption = "@EthanSterbis on X
       data via nflverse and FTN")

#ggsave('off-zc-vs-oe.jpeg', height = 8, width = 10, dpi = 'retina')

# defenses playing zone coverage
ggplot(data = defTeamEpaVSoe, aes(x = avg_zone_oe,
                                  y = zone_epa)) +
  geom_hline(yintercept = mean(defTeamEpaVSoe$zone_epa),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_vline(xintercept = mean(defTeamEpaVSoe$avg_zone_oe),
             color = "red", lty = 'dashed', size = 0.8) +
  geom_image(aes(image = team_logo_wikipedia), asp = 16/9) +
  scale_x_continuous(breaks = pretty_breaks(n = 12),
                     labels = percent_format()) +
  scale_y_continuous(breaks = pretty_breaks(n = 8)) +
  sterb_analytics_theme() +
  labs(x = "Average Zone Coverage Over Expectation",
       y = "Average EPA in Zone Coverage",
       title = paste0(fSeasonsText, " NFL Defenses Playing Zone Coverage: Rate Over Expectation vs. Efficiency"),
       subtitle = "Expectation via XGBoost Model",
       caption = "@EthanSterbis on X
       data via nflverse and FTN")

#ggsave('def-zc-vs-oe.jpeg', height = 8, width = 10, dpi = 'retina')