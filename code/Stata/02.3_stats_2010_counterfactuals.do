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
global stats "stats"
global output "output"

////////////////////////////////////////////////////////////////////////////////
* Prediction changes under different scenarios
////////////////////////////////////////////////////////////////////////////////

// import delimited "$output/test_predictions_2010_shocks1.csv", clear
//
// collapse (sum) predicted_y_*
//
// label var predicted_y_original "No Shock"
// label var predicted_y_eliminate_remittance "Eliminate Remittances"
// label var predicted_y_double_l1_rem_us "Double US Remittances"
// label var predicted_y_halve_l1_rem_us "Halve US Remittances"
// label var predicted_y_double_l1_trans_inc "Double Income from Transfers"
// label var predicted_y_halve_l1_trans_inc "Halve Income from Transfers"
// label var predicted_y_double_l1_hhworkforc "Double HH Workforce"
// label var predicted_y_halve_l1_hhworkforce "Halve HH Workforce"
// label var predicted_y_double_age "Double Workforce Age"
// label var predicted_y_halve_age "Halve Workforce Age"
// label var predicted_y_no_l1_ag "No Ag Jobs"
// label var predicted_y_no_l1_nonag "No Nonag Jobs"
// label var predicted_y_no_l1_work_us "No US Jobs"
// label var predicted_y_yes_l1_work_mx "Only MX Jobs"
// label var predicted_y_no_l1_hh_migrant "No Other Migrants from HH"
// label var predicted_y_double_l1_farmlab_in "Double Farm Labor Income"
// label var predicted_y_halve_l1_farmlab_inc "Halve Farm Labor Income"
// label var predicted_y_double_l1_hhchildren "Double HH Children"
// label var predicted_y_halve_l1_hhchildren "Halve HH Children"
//
// graph hbar (asis) predicted_y_original ///
//                   predicted_y_eliminate_remittance ///
// 				  predicted_y_double_age ///
// 				  predicted_y_halve_l1_hhworkforce ///
// 				  predicted_y_double_l1_hhchildren ///
// 				  predicted_y_no_l1_ag ///
// 				  predicted_y_no_l1_nonag ///
// 				  predicted_y_halve_l1_farmlab_inc, ///
//     title("Counterfactual Predictions in 2010 Following Negative Shocks in 2009", size(medium)) ///
//     ytitle("No. of Predicted Migrants", size(small)) ///
// 	yla(350(25)500,labs(small)) b1title(,size(small)) exclude0 ///
//     asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
//     intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
// 	note("NOTE: Remittances and income only vary in 2002, 2003, and 2008, and are extrapolated, likely underestimating their impact." "",size(vsmall))
// graph export "$stats/migrants_2010_nshocks.png", replace
//
// graph hbar (asis) predicted_y_original ///
//                   predicted_y_double_l1_rem_us ///
// 				  predicted_y_halve_age ///
// 				  predicted_y_double_l1_hhworkforc ///
// 				  predicted_y_halve_l1_hhchildren ///
// 				  predicted_y_no_l1_hh_migrant ///
// 				  predicted_y_double_l1_trans_inc ///
// 				  predicted_y_double_l1_farmlab_in, ///
//     title("Counterfactual Predictions in 2010 Following Positive Shocks in 2009", size(medium)) ///
//     ytitle("No. of Predicted Migrants", size(small)) ///
// 	yla(350(25)500,labs(small)) b1title(,size(small)) exclude0 ///
//     asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
//     intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
// 	caption("NOTE: Remittances and income only vary in 2002, 2003, and 2008, and are extrapolated, likely underestimating their impact.",size(vsmall))
// graph export "$stats/migrants_2010_pshocks.png", replace

////////////////////////////////////////////////////////////////////////////////
* Prediction changes under different scenarios, select IDs
////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2010_shocks1.csv", clear
merge m:1 ind using "$stats/ids.dta", keepusing(ind correct_prediction)
keep if _merge==3
keep ind correct_prediction predicted_y*
tempfile temp
save `temp'

use "$data/MexMigData.dta", clear 
merge m:1 ind using `temp'
keep if _merge==3

keep ind year work_us correct_prediction predicted_y_*
egen id = group(ind)

////////////////////////////////////////////////////////////////////////////////
preserve

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_eliminate_remittance if year == 2010
gen highlight = (year==2010)

tab id work_us if year==2010
label define id_lbl 1 "Still Migrates" 2 "No Longer Migrates" 3 "No Longer Migrates" ///
	4 "Still Migrates" 5 "Still Migrates" 6 "Still Doesn't Migrate" 7 "Still Migrates" ///
	8 "Still Migrates" 9 "Still Migrates" 10 "Still Migrates" 11 "No Longer Migrates" ///
	12 "Still Migrates" 13 "Still Migrates" 14 "Still Migrates" 15 "Still Migrates" ///
	16 "Still Migrates" 17 "Still Migrates" 18 "No Longer Migrates" ///
	19 "Still Migrates" 20 "Still Migrates"
label values id id_lbl

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories and Counterfactual Predictions of 2010 Migrants: Shock - Eliminate Remittances", size(medium)) ///
	note("These graphs show individuals with less than 3 years of US experience during the 5 years preceding 2010 (2005–2009). Counterfactual predictions are highlighted in red.",size(vsmall)) ///
	caption("NOTE: Remittances only vary in 2002, 2003, and 2008, which likely underestimates their overall impact.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/migrants_2010_elim_remittances.png", replace

////////////////////////////////////////////////////////////////////////////////
restore
preserve

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_double_age if year == 2010
gen highlight = (year==2010)

tab id work_us if year==2010
label define id_lbl 1 "Still Migrates" 2 "No Longer Migrates" 3 "No Longer Migrates" ///
	4 "Still Migrates" 5 "Still Migrates" 6 "Still Doesn't Migrate" 7 "No Longer Migrates" ///
	8 "Still Migrates" 9 "Still Migrates" 10 "Still Migrates" 11 "No Longer Migrates" ///
	12 "Still Migrates" 13 "Still Migrates" 14 "No Longer Migrates" 15 "No Longer Migrates" ///
	16 "Still Migrates" 17 "Still Migrates" 18 "No Longer Migrates" ///
	19 "Still Migrates" 20 "Still Migrates"
label values id id_lbl

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories and Counterfactual Predictions of 2010 Migrants: Shock - Double Workforce Age", size(medium)) ///
	note("These graphs show individuals with less than 3 years of US experience during the 5 years preceding 2010 (2005–2009). Counterfactual predictions are highlighted in red.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/migrants_2010_double_age.png", replace

////////////////////////////////////////////////////////////////////////////////
restore

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_double_l1_hhworkforc if year == 2010
gen highlight = (year==2010)

tab id work_us if year==2010
label define id_lbl 1 "Still Migrates" 2 "No Longer Migrates" 3 "Still Migrates" ///
	4 "Still Migrates" 5 "Still Migrates" 6 "Still Doesn't Migrate" 7 "Still Migrates" ///
	8 "Still Migrates" 9 "Still Migrates" 10 "Still Migrates" 11 "Still Migrates" ///
	12 "Still Migrates" 13 "Still Migrates" 14 "Still Migrates" 15 "Still Migrates" ///
	16 "Still Migrates" 17 "Still Migrates" 18 "Still Migrates" ///
	19 "Still Migrates" 20 "Still Migrates"
label values id id_lbl

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories and Counterfactual Predictions of 2010 Migrants: Shock - Double HH Workforce", size(medium)) ///
	note("These graphs show individuals with less than 3 years of US experience during the 5 years preceding 2010 (2005–2009). Counterfactual predictions are highlighted in red.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/migrants_2010_double_hhworkforce.png", replace
