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

import delimited "$output/test_predictions_2010_shocks.csv", clear
gen correct_prediction = (actual_y==predicted_y_original)
rename (work_us l1_work_us) (work_us_2010 work_us_2009)

gen correct_migrant = (correct_prediction==1 & work_us_2010==1)

gen stayed_migrant = (work_us_2010==1 & work_us_2009==1)
tab stayed_migrant
* 328 were migrants in 2009 and stayed in 2010

gen new_migrant = (work_us_2010==1 & work_us_2009==0)
tab new_migrant
* 16 changed status from nonmigrant to migrant in 2010 (11 new)

gen correct_stayed_migrant = (correct_migrant==1 & work_us_2009==1)
tab correct_stayed_migrant
* 328/328 of those who stayed migrant correctly predicted, 0 incorrect

gen correct_new_migrant = (correct_migrant==1 & work_us_2009==0)
tab correct_new_migrant
* 14/16 of new migrants correctly predicted, 2 incorrect

preserve
keep ind correct_prediction predicted_y_* actual_y
tempfile nonnew_migrant
save `nonnew_migrant'

restore
keep if new_migrant==1
keep ind correct_prediction predicted_y_* actual_y
tempfile new_migrant
save `new_migrant'

////////////////////////////////////////////////////////////////////////////////

use "$data/MexMigData.dta", clear 
merge m:1 ind using `new_migrant'
keep if _merge==3
keep ind year work_us correct_prediction predicted_y_*
egen id = group(ind)

////////////////////////////////////////////////////////////////////////////////
preserve

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_eliminate_remittance if year == 2010
gen highlight = (year==2010)

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Eliminate Remittances", size(small)) ///
	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
	caption("NOTE: Remittances only vary in 2002, 2003, and 2008, which likely underestimates their overall impact.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/new_migrants_2010_elim_remittances.png", replace

////////////////////////////////////////////////////////////////////////////////
restore
preserve

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_double_age if year == 2010
gen highlight = (year==2010)

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Double Workforce Age", size(small)) ///
	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/new_migrants_2010_double_age.png", replace

////////////////////////////////////////////////////////////////////////////////
restore

* Replace actual y with the prediction from scenario:
replace work_us = predicted_y_no_nonag if year == 2010
gen highlight = (year==2010)

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories of 2010 New Migrants and Counterfactual Predictions: Shock - Only Ag Jobs", size(small)) ///
	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/new_migrants_2010_no_nonag.png", replace

////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2010_shocks.csv", clear

collapse (sum) predicted_y_*

label var predicted_y_original "Original Predictions"
label var predicted_y_eliminate_remittance "Eliminate Remittances"
label var predicted_y_double_rem_us "Double US Remittances"
label var predicted_y_halve_rem_us "Halve US Remittances"
label var predicted_y_double_trans_inc "Double Income from Transfers"
label var predicted_y_halve_trans_inc "Halve Income from Transfers"
label var predicted_y_double_hhworkforce "Double Household Workforce"
label var predicted_y_halve_hhworkforce "Halve Household Workforce"
label var predicted_y_double_age "Double Workforce Age"
label var predicted_y_halve_age "Halve Workforce Age"
label var predicted_y_no_ag "No Ag Jobs"
label var predicted_y_no_nonag "Only Ag Jobs"
label var predicted_y_no_l1_work_us "No US Jobs in Previous Period"
label var predicted_y_yes_l1_work_mx "Only MX Jobs in Previous Period"

graph hbar (asis) predicted_y_original ///
                  predicted_y_eliminate_remittance ///
				  predicted_y_halve_hhworkforce ///
				  predicted_y_double_age ///
				  predicted_y_no_ag ///
				  predicted_y_no_nonag ///
				  predicted_y_no_l1_work_us, ///
    title("Counterfactual Predictions in 2010 Under Negative Shocks", size(medium)) ///
    ytitle("No. of Migrants", size(small)) ///
	yla(,labs(small)) b1title(,size(small)) ///
    asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
    intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
	note("NOTE: Remittances only vary in 2002, 2003, and 2008, which likely underestimates their overall impact.",size(vsmall))
graph export "$stats/migrants_2010_nshocks.png", replace

graph hbar (asis) predicted_y_original ///
                  predicted_y_double_rem_us ///
				  predicted_y_double_trans_inc ///
				  predicted_y_double_hhworkforce ///
				  predicted_y_halve_age ///
				  predicted_y_yes_l1_work_mx, ///
    title("Counterfactual Predictions in 2010 Under Positive Shocks", size(medium)) ///
    ytitle("No. of Migrants", size(small)) ///
	yla(,labs(small)) b1title(,size(small)) ///
    asyvars showyvars leg(off) label yvaroptions(label(labs(small))) ///
    intensity(70) bargap(20) blabel(bar, size(vsmall)) ///
	caption("NOTE: Remittances and income from transfers only vary in 2002, 2003, and 2008, which likely underestimates their" "overall impact.",size(vsmall))
graph export "$stats/migrants_2010_pshocks.png", replace
