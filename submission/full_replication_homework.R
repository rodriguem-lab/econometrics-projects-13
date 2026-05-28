# ============================================================
# Partial replication workflow translated from Python to R
# ============================================================
#
# Notes:
# 1. This script mirrors the Python homework workflow.
# 2. It is meant as a readable translation and may require local
#    package installation before execution.
# 3. The original paper relies on a richer hazard specification.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(zoo)
  library(AER)
})

root <- "."
data_root <- file.path(root, "Data for Household Mobility and Mortgage Rate Lock", "Replication Package")
synth_dir <- file.path(data_root, "raw", "synthetic data")
out_dir <- file.path(root, "analysis_outputs")
dir.create(out_dir, showWarnings = FALSE)

latest_fred <- "/private/tmp/MORTGAGE30US_latest.csv"

load_mortgage_panel <- function() {
  df <- read_csv(file.path(synth_dir, "mortgage_payments_data.csv"), show_col_types = FALSE)
  df %>%
    mutate(
      file_date_dt = ymd(file_date),
      quarter = as.yearqtr(file_date_dt),
      acct_open_dt_dt = ymd(acct_open_dt),
      year = year(file_date_dt),
      month = month(file_date_dt),
      open_year = year(acct_open_dt_dt),
      open_month = month(acct_open_dt_dt),
      loan_age_months = (year - open_year) * 12 + (month - open_month),
      remaining_term = terms_trans - loan_age_months,
      log_balance = log1p(acct_balance_am)
    )
}

load_moves <- function() {
  read_csv(file.path(synth_dir, "national_get_all_moves.csv"), show_col_types = FALSE) %>%
    mutate(
      quarter = as.yearqtr(ymd(paste0(file_year, "-", sprintf("%02d", file_month), "-01"))),
      move_flag = if_else(zip1 != zip2, 1, 0)
    ) %>%
    group_by(personid, quarter) %>%
    summarise(move_flag = max(move_flag), .groups = "drop")
}

load_rates <- function() {
  if (file.exists(latest_fred)) {
    rates <- read_csv(latest_fred, show_col_types = FALSE) %>%
      transmute(date = ymd(observation_date), market_rate = MORTGAGE30US)
  } else {
    rates <- read_csv(file.path(data_root, "raw", "MORTGAGE30US.csv"), show_col_types = FALSE) %>%
      transmute(date = ymd(DATE), market_rate = MORTGAGE30US)
  }

  rates %>%
    mutate(quarter = as.yearqtr(date)) %>%
    group_by(quarter) %>%
    summarise(market_rate = mean(market_rate, na.rm = TRUE), .groups = "drop")
}

panel <- load_mortgage_panel() %>%
  left_join(load_moves(), by = c("personid", "quarter")) %>%
  left_join(load_rates(), by = "quarter") %>%
  mutate(
    move_flag = if_else(is.na(move_flag), 0, move_flag),
    rate_gap = market_rate - mortgage_rate_annual,
    balance_to_orig = if_else(amount_1 == 0, NA_real_, acct_balance_am / amount_1)
  ) %>%
  arrange(personid, quarter) %>%
  group_by(personid) %>%
  mutate(
    trend = row_number(),
    quarter_num = row_number(),
    rate_gap_between = mean(rate_gap, na.rm = TRUE),
    move_flag_between = mean(move_flag, na.rm = TRUE),
    balance_to_orig_between = mean(balance_to_orig, na.rm = TRUE),
    loan_age_years = loan_age_months / 12,
    loan_age_years_between = mean(loan_age_years, na.rm = TRUE),
    log_balance_between = mean(log_balance, na.rm = TRUE),
    rate_gap_within = rate_gap - rate_gap_between,
    move_flag_within = move_flag - move_flag_between,
    balance_to_orig_within = balance_to_orig - balance_to_orig_between,
    loan_age_years_within = loan_age_years - loan_age_years_between,
    log_balance_within = log_balance - log_balance_between,
    rate_gap_fd = rate_gap - lag(rate_gap),
    move_flag_fd = move_flag - lag(move_flag),
    balance_to_orig_fd = balance_to_orig - lag(balance_to_orig),
    loan_age_years_fd = loan_age_years - lag(loan_age_years),
    log_balance_fd = log_balance - lag(log_balance)
  ) %>%
  ungroup()

write_csv(panel, file.path(out_dir, "panel_data_R_export.csv"))

# Core regressions
between_df <- panel %>%
  group_by(personid) %>%
  summarise(
    move_flag = mean(move_flag, na.rm = TRUE),
    rate_gap = mean(rate_gap, na.rm = TRUE),
    balance_to_orig = mean(balance_to_orig, na.rm = TRUE),
    loan_age_years = mean(loan_age_years, na.rm = TRUE),
    log_balance = mean(log_balance, na.rm = TRUE),
    .groups = "drop"
  )

between_mod <- lm(move_flag ~ rate_gap + balance_to_orig + loan_age_years + log_balance, data = between_df)
within_mod <- lm(move_flag_within ~ rate_gap_within + balance_to_orig_within + loan_age_years_within + log_balance_within, data = panel)
fd_mod <- lm(move_flag_fd ~ rate_gap_fd + balance_to_orig_fd + loan_age_years_fd + log_balance_fd, data = panel)

# Dynamic IV illustration
dyn_df <- panel %>%
  group_by(personid) %>%
  mutate(
    dy = move_flag_fd,
    dx = rate_gap_fd,
    dy_l1 = lag(dy),
    dx_l1 = lag(dx),
    y_l2 = lag(move_flag, 2),
    x_l2 = lag(rate_gap, 2)
  ) %>%
  ungroup() %>%
  filter(!is.na(dy), !is.na(dx), !is.na(dy_l1), !is.na(dx_l1), !is.na(y_l2), !is.na(x_l2))

iv_mod <- ivreg(
  dy ~ dx + balance_to_orig_fd + loan_age_years_fd + log_balance_fd + dy_l1 + dx_l1 |
    dx + balance_to_orig_fd + loan_age_years_fd + log_balance_fd + y_l2 + x_l2,
  data = dyn_df
)

print(summary(between_mod))
print(summary(within_mod))
print(summary(fd_mod))
print(summary(iv_mod))

cat("\nR translation completed. The exact implied mortgage rate inversion used in Python\n")
cat("would need a dedicated root-finding step for full equivalence.\n")
