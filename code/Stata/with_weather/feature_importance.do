* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*/
* ------------------------------------------------------------------------------

clear all
macro drop _all
if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
global stats "stats/with_weather"
global output "output/with_weather"

import delimited "$output/lightgbm_feature_importance.csv", clear

gen lab = ""
replace feature = subinstr(feature, ".0", "", .)
replace lab = subinstr(feature, "vill_", "Village ", .)
replace lab = "Age"                if feature=="age"
replace lab = "US yrs"             if feature=="L1_yrs_in_us_cum"
replace lab = "US work"            if feature=="L1_work_us"
replace lab = "US remittances"     if feature=="L1_rem_us"
replace lab = "HH US yrs"          if feature=="L1_hh_yrs_in_us_cum"
replace lab = "HH migrant"         if feature=="L1_hh_migrant"
replace lab = "Farm yrs"           if feature=="L1_yrs_in_ag_cum"
replace lab = "HH workforce"       if feature=="L1_hhworkforce"
replace lab = "MX yrs"             if feature=="L1_yrs_in_mx_cum"
replace lab = "Livestock inc"      if feature=="L1_liv_inc"
replace lab = "Transfer inc"       if feature=="L1_trans_inc"
replace lab = "Ag inc"             if feature=="L1_ag_inc"
replace lab = "Nonag inc"          if feature=="L1_nonag_inc"
replace lab = "HH children"        if feature=="L1_hhchildren"
replace lab = "Rent inc"           if feature=="L1_rec_inc"
replace lab = "Farm-labor inc"     if feature=="L1_farmlab_inc"
replace lab = "Asset inc"          if feature=="L1_asset_inc"
replace lab = "Male"               if feature=="male"
replace lab = "Nonag yrs"          if feature=="L1_yrs_in_nonag_cum"
replace lab = "Ag work"            if feature=="L1_ag"
replace lab = "MX work"            if feature=="L1_work_mx"
replace lab = "MX remittances"     if feature=="L1_rem_mx"
replace lab = "Non ag work"        if feature=="L1_nonag"
replace lab = "Ag plot rent"       if feature=="L1_plot_inc_renta_ag"
replace lab = "Nonag plot rent"    if feature=="L1_plot_inc_renta_nonag"

gsort -importance
keep in 1/20

graph hbar (asis) importance, name(g1) over(lab, sort(1) descending gap(10)) ///
title("LightGBM (Benchmark)") bar(1,color(edkblue)) ///
ytitle("",size(small)) ylab(,angle(horizontal))

********************************************************************************

import delimited "$output/logistic_nm1_w_feature_importance.csv", clear

gen lab = ""
replace feature = subinstr(feature, ".0", "", .)
replace lab = subinstr(feature, "vill_", "Village ", .)
replace lab = "Age"                if feature=="age"
replace lab = "US yrs"             if feature=="L1_yrs_in_us_cum"
replace lab = "US work"            if feature=="L1_work_us"
replace lab = "US remittances"     if feature=="L1_rem_us"
replace lab = "HH US yrs"          if feature=="L1_hh_yrs_in_us_cum"
replace lab = "HH migrant"         if feature=="L1_hh_migrant"
replace lab = "Farm yrs"           if feature=="L1_yrs_in_ag_cum"
replace lab = "HH workforce"       if feature=="L1_hhworkforce"
replace lab = "MX yrs"             if feature=="L1_yrs_in_mx_cum"
replace lab = "Livestock inc"      if feature=="L1_liv_inc"
replace lab = "Transfer inc"       if feature=="L1_trans_inc"
replace lab = "Ag inc"             if feature=="L1_ag_inc"
replace lab = "Nonag inc"          if feature=="L1_nonag_inc"
replace lab = "HH children"        if feature=="L1_hhchildren"
replace lab = "Rent inc"           if feature=="L1_rec_inc"
replace lab = "Farm-labor inc"     if feature=="L1_farmlab_inc"
replace lab = "Asset inc"          if feature=="L1_asset_inc"
replace lab = "Male"               if feature=="male"
replace lab = "Nonag yrs"          if feature=="L1_yrs_in_nonag_cum"
replace lab = "Ag work"            if feature=="L1_ag"
replace lab = "MX work"            if feature=="L1_work_mx"
replace lab = "MX remittances"     if feature=="L1_rem_mx"
replace lab = "Non ag work"        if feature=="L1_nonag"
replace lab = "Ag plot rent"       if feature=="L1_plot_inc_renta_ag"
replace lab = "Nonag plot rent"    if feature=="L1_plot_inc_renta_nonag"
replace lab = "Avg temp (lag 5)"       if feature=="avgtemp5"
replace lab = "Avg temp (lag 6)"       if feature=="avgtemp6"
replace lab = "Avg temp (lag 7)"       if feature=="avgtemp7"
replace lab = "Avg temp (lag 8)"       if feature=="avgtemp8"
replace lab = "Precip (lag 5)"         if feature=="precip_tot5"
replace lab = "Precip (lag 6)"         if feature=="precip_tot6"
replace lab = "Precip (lag 7)"         if feature=="precip_tot7"
replace lab = "Precip (lag 8)"         if feature=="precip_tot8"
replace lab = "GDD (lag 5)"            if feature=="GDD5"
replace lab = "GDD (lag 6)"            if feature=="GDD6"
replace lab = "GDD (lag 7)"            if feature=="GDD7"
replace lab = "GDD (lag 8)"            if feature=="GDD8"
replace lab = "HDD (lag 5)"            if feature=="HDD5"
replace lab = "HDD (lag 6)"            if feature=="HDD6"
replace lab = "HDD (lag 7)"            if feature=="HDD7"
replace lab = "HDD (lag 8)"            if feature=="HDD8"
replace lab = "Precip (ag season)"     if feature=="precip_tot_MDagseason"
replace lab = "HDD (ag season)"        if feature=="HDD_MDagseason"
replace lab = "GDD (ag season)"        if feature=="GDD_MDagseason"
replace lab = "Precip (non‐ag season)" if feature=="precip_tot_nonagseason"
replace lab = "HDD (non‐ag season)"    if feature=="HDD_nonagseason"
replace lab = "GDD (non‐ag season)"    if feature=="GDD_nonagseason"

gsort -importance
keep in 1/20

graph hbar (asis) importance, name(g2) over(lab, sort(1) descending gap(10)) ///
title("LightGBM (Restricted)") bar(1,color(maroon)) ///
ytitle("",size(small)) ylab(,angle(horizontal))

graph combine g1 g2, title("Top 20 Most Important Features") ///
note("All predictors are one-year lagged. Values show relative contribution to reduction in binary log-loss." "Benchmark trained on 1980–2006 with migration history. Restricted trained on 2003–2006 without migration history but with weather data.", size(vsmall))
graph export "$stats/importance_comparison.eps", replace preview(on)
