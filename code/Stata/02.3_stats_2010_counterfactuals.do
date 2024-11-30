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
// gen correct_prediction = (actual_y==predicted_y_original)
// rename (work_us l1_work_us) (work_us_2010 work_us_2009)
//
// keep ind correct_prediction predicted_y_* actual_y
// //preserve
// collapse (sum) predicted_y_*
// gen i = 1
// reshape long predicted_y_, i(i) j(shock) string
// drop i
// export excel using "$stats/predicted_y_shocks.xlsx", replace firstrow(var)

// restore
// preserve
// tempfile nonnew_migrant
// save `nonnew_migrant'
//
// restore
// keep if new_migrant==1
// tempfile new_migrant
// save `new_migrant'

////////////////////////////////////////////////////////////////////////////////

// use "$data/MexMigData.dta", clear 
// merge m:1 ind using `new_migrant'
// keep if _merge==3
// keep ind year work_us correct_prediction predicted_y_*
// egen id = group(ind)

////////////////////////////////////////////////////////////////////////////////
// preserve
//
// * Replace actual y with the prediction from scenario:
// replace work_us = predicted_y_eliminate_remittance if year == 2010
// gen highlight = (year==2010)
//
// sepscatter work_us year, separate(highlight) sort by(id, ///
//     title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Eliminate Remittances", size(small)) ///
// 	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
// 	caption("NOTE: Remittances only vary in 2002, 2003, and 2008, which likely underestimates their overall impact.",size(vsmall)) ///
// 	legend(off)) mc(blue red) ms(O O) ///
//     yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
//     ytitle("Worked in...", size(small)) xtitle("")
// graph export "$stats/new_migrants_2010_elim_remittances.png", replace
//
// ////////////////////////////////////////////////////////////////////////////////
// restore
// preserve
//
// * Replace actual y with the prediction from scenario:
// replace work_us = predicted_y_double_age if year == 2010
// gen highlight = (year==2010)
//
// sepscatter work_us year, separate(highlight) sort by(id, ///
//     title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Double Workforce Age", size(small)) ///
// 	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
// 	legend(off)) mc(blue red) ms(O O) ///
//     yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
//     ytitle("Worked in...", size(small)) xtitle("")
// graph export "$stats/new_migrants_2010_double_age.png", replace
//
// ////////////////////////////////////////////////////////////////////////////////
// restore
//
// * Replace actual y with the prediction from scenario:
// replace work_us = predicted_y_no_l1_nonag if year == 2010
// gen highlight = (year==2010)
//
// sepscatter work_us year, separate(highlight) sort by(id, ///
//     title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Only Ag Jobs", size(small)) ///
// 	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
// 	legend(off)) mc(blue red) ms(O O) ///
//     yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
//     ytitle("Worked in...", size(small)) xtitle("")
// graph export "$stats/new_migrants_2010_no_l1_nonag.png", replace

////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2010_shocks1.csv", clear

collapse (sum) predicted_y_*

label var predicted_y_original "No Shock"
label var predicted_y_eliminate_remittance "Eliminate Remittances"
label var predicted_y_double_l1_rem_us "Double US Remittances"
label var predicted_y_halve_l1_rem_us "Halve US Remittances"
label var predicted_y_double_l1_trans_inc "Double Income from Transfers"
label var predicted_y_halve_l1_trans_inc "Halve Income from Transfers"
label var predicted_y_double_l1_hhworkforc "Double HH Workforce"
label var predicted_y_halve_l1_hhworkforce "Halve HH Workforce"
label var predicted_y_double_age "Double Workforce Age"
label var predicted_y_halve_age "Halve Workforce Age"
label var predicted_y_no_l1_ag "No Ag Jobs"
label var predicted_y_no_l1_nonag "No Nonag Jobs"
label var predicted_y_no_l1_work_us "No US Jobs"
label var predicted_y_yes_l1_work_mx "Only MX Jobs"
label var predicted_y_no_l1_hh_migrant "No Other Migrants from HH"
label var predicted_y_double_l1_farmlab_in "Double Farm Labor Income"
label var predicted_y_halve_l1_farmlab_inc "Halve Farm Labor Income"
label var predicted_y_double_l1_hhchildren "Double HH Children"
label var predicted_y_halve_l1_hhchildren "Halve HH Children"

graph hbar (asis) predicted_y_original ///
                  predicted_y_eliminate_remittance ///
				  predicted_y_double_age ///
				  predicted_y_halve_l1_hhworkforce ///
				  predicted_y_double_l1_hhchildren ///
				  predicted_y_no_l1_ag ///
				  predicted_y_no_l1_nonag ///
				  predicted_y_halve_l1_farmlab_inc, ///
    title("Counterfactual Predictions in 2010 Following Negative Shocks in 2009", size(medium)) ///
    ytitle("No. of Predicted Migrants", size(small)) ///
	yla(350(25)500,labs(small)) b1title(,size(small)) exclude0 ///
    asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
    intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
	note("NOTE: Remittances and income only vary in 2002, 2003, and 2008, and are extrapolated, likely underestimating their impact." "",size(vsmall))
graph export "$stats/migrants_2010_nshocks.png", replace

graph hbar (asis) predicted_y_original ///
                  predicted_y_double_l1_rem_us ///
				  predicted_y_halve_age ///
				  predicted_y_double_l1_hhworkforc ///
				  predicted_y_halve_l1_hhchildren ///
				  predicted_y_no_l1_hh_migrant ///
				  predicted_y_double_l1_trans_inc ///
				  predicted_y_double_l1_farmlab_in, ///
    title("Counterfactual Predictions in 2010 Following Positive Shocks in 2009", size(medium)) ///
    ytitle("No. of Predicted Migrants", size(small)) ///
	yla(350(25)500,labs(small)) b1title(,size(small)) exclude0 ///
    asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
    intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
	caption("NOTE: Remittances and income only vary in 2002, 2003, and 2008, and are extrapolated, likely underestimating their impact.",size(vsmall))
graph export "$stats/migrants_2010_pshocks.png", replace
