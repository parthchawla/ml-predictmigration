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
use "$data/MexMigData.dta", clear

////////////////////////////////////////////////////////////////////////////////
* 2010 general stats
////////////////////////////////////////////////////////////////////////////////

* Years in US 1980-2009:
bys ind: egen totalus_1980_2009 = sum(work_us*(year<2010))

gen us_experience_1980_2009 = ""
replace us_experience_1980_2009 = "No experience" if totalus_1980_2009==0
replace us_experience_1980_2009 = "1-3 years" if totalus_1980_2009>=1 & totalus_1980_2009<=3
replace us_experience_1980_2009 = "4-7 years" if totalus_1980_2009>=4 & totalus_1980_2009<=7
replace us_experience_1980_2009 = "8-15 years" if totalus_1980_2009>=8 & totalus_1980_2009<=15
replace us_experience_1980_2009 = "16+ years" if totalus_1980_2009>=16 & totalus_1980_2009!=.
tab us_experience_1980_2009

* Create lagged migration status
order ind year work_us
sort ind year
xtset ind year
forval i = 1/5 {
	local y = 2010 - `i'
	display "Year: `y'"
	gen work_us_`y' = L`i'.work_us
}

keep if year == 2010
rename work_us work_us_2010
keep ind work_us* totalus_1980_2009 us_experience_1980_2009

egen past5tot = rowtotal(work_us_2009 work_us_2008 work_us_2007 work_us_2006 work_us_2005)
gen past5once_us = (past5tot > 0)

tab past5tot work_us_2010
tab past5once_us work_us_2010
tab totalus_1980_2009 work_us_2010
tab us_experience_1980_2009 work_us_2010
preserve

////////////////////////////////////////////////////////////////////////////////
* What did it get right
////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2010_add_vars.csv", clear
gen correct_prediction = (actual_y==predicted_y)
tab correct_prediction
tempfile pred
save `pred'

restore
merge 1:1 ind using `pred', nogen keepusing(correct_prediction predicted_y)

gen correct_migrant = (correct_prediction==1 & work_us_2010==1)
tab correct_migrant
* 342/344 migrants correctly predicted, 2 incorrect (1 new)

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

tab us_experience_1980_2009 new_migrant
* 11 migrants in 2010 with no US experience

gen correct_new_migrant_nous = correct_new_migrant==1 & us_experience_1980_2009=="No experience"
tab correct_new_migrant_nous
* 10/11 of new migrants correctly predicted, 1 incorrect

tab us_experience_1980_2009 correct_new_migrant
* The other incorrect one is 4-7 years experience

////////////////////////////////////////////////////////////////////////////////

keep if new_migrant==1
keep ind correct_prediction predicted_y
tempfile new_migrant
save `new_migrant'

use "$data/MexMigData.dta", clear 
merge m:1 ind using `new_migrant'
keep if _merge==3
keep ind year work_us correct_prediction predicted_y
egen id = group(ind)

tab id correct_prediction if year==2010
label define id_lbl 1 "Correct" 2 "Correct" 3 "Correct" 4 "Incorrect" ///
    5 "Correct" 6 "Correct" 7 "Correct" 8 "Correct" 9 "Correct" ///
    10 "Incorrect" 11 "Correct" 12 "Correct" 13 "Correct" 14 "Correct" ///
    15 "Correct" 16 "Correct"
label values id id_lbl

gen highlight = (year==2010)
// Replace actual y with the prediction:
replace work_us = predicted_y if year == 2010

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories of 2010 New Migrants and Predictions", size(medium)) ///
	note("New migrants are the 16 individuals in the sample who worked in the US in 2010 but not in 2009. 328 individuals worked in the US in both 2009 and 2010.",size(vsmall)) ///
	caption("The ML model correctly predicted 14 of the 16 new migrants in 2010 and all 328 existing migrants.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("Worked in...", size(small)) xtitle("")
graph export "$stats/new_migrants_2010.png", replace
