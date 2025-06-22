* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Nov 24, 2024
*/
* ------------------------------------------------------------------------------

cls
clear all
macro drop _all
if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
global stats "stats/with_weather"
global output "output/with_weather"

////////////////////////////////////////////////////////////////////////////////
* Prediction changes under different scenarios
////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2007_temp_shock.csv", clear

collapse (sum) ///
    pred_original                   ///
    pred_temp_plus_10pct            ///
    pred_precip_plus_10pct          ///
    pred_gdd_plus_10pct             ///
    pred_hdd_plus_10pct             ///
    pred_all_weather_plus_10pct     ///
    pred_age_plus_10pct             ///
    pred_hhworkforce_minus_10pct    ///
    pred_hhchildren_minus_10pct     ///
    pred_income_minus_10pct         ///
    pred_demo_income_minus_10pct    ///
    pred_age_plus_20pct             ///
    pred_hhworkforce_minus_20pct    ///
    pred_hhchildren_minus_20pct     ///
    pred_income_minus_20pct

label var pred_original                   "No Shock"
label var pred_temp_plus_10pct            "Avg Temp +10%"
label var pred_precip_plus_10pct          "Precipitation +10%"
label var pred_gdd_plus_10pct             "GDD +10%"
label var pred_hdd_plus_10pct             "HDD +10%"
label var pred_all_weather_plus_10pct     "All Weather Vars +10%"
label var pred_age_plus_10pct             "Age +10%"
label var pred_hhworkforce_minus_10pct    "HH Workforce −10%"
label var pred_hhchildren_minus_10pct     "HH Children −10%"
label var pred_income_minus_10pct         "Income −10%"
label var pred_demo_income_minus_10pct    "Demo + Income −10%"
label var pred_age_plus_20pct             "Age +20%"
label var pred_hhworkforce_minus_20pct    "HH Workforce −20%"
label var pred_hhchildren_minus_20pct     "HH Children −20%"
label var pred_income_minus_20pct         "All Income −20%"

graph hbar (asis) ///
    pred_original                   ///
    pred_temp_plus_10pct            ///
    pred_precip_plus_10pct          ///
    pred_all_weather_plus_10pct     ///
    pred_age_plus_10pct             ///
    pred_hhworkforce_minus_10pct    ///
    pred_hhchildren_minus_10pct     ///
	pred_hhchildren_minus_20pct    	///
	pred_income_minus_10pct,    	///
    title("2007 Predicted Migrants under Shock Scenarios", size(medium)) ///
    ytitle("No. of Predicted Migrants", size(small)) ///
	yla(,labs(small)) b1title(,size(small)) exclude0 ///
    asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
    intensity(70) bargap(20) blabel(bar, size(vsmall))
graph export "$stats/migrants_2007_nshocks.eps", replace preview(on)
