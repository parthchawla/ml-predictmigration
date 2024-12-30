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
tab us_experience_1980_2009 work_us_2010 // 11 migrants in 2010 with no US exp
preserve

////////////////////////////////////////////////////////////////////////////////
* What did it get right
////////////////////////////////////////////////////////////////////////////////

import delimited "$output/test_predictions_2010_add_vars1.csv", clear
gen correct_prediction = (actual_y==predicted_y)
tab correct_prediction
tempfile pred
save `pred'

restore
merge 1:1 ind using `pred', nogen keepusing(correct_prediction predicted_y)

gen correct_migrant = (correct_prediction==1 & work_us_2010==1)
gen incorrect_migrant = (correct_prediction==0 & work_us_2010==1)
tab correct_migrant
* 336/344 migrants correctly predicted, 9 incorrect

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
* 8/16 of new migrants correctly predicted, 8 incorrect

tab us_experience_1980_2009 new_migrant
* 11 migrants in 2010 with no US experience

gen correct_new_migrant_nous = correct_new_migrant==1 & us_experience_1980_2009=="No experience"
tab correct_new_migrant_nous
* 5/11 of new migrants correctly predicted, 6 incorrect

tab us_experience_1980_2009 incorrect_migrant
* Out of 8 incorrect, 6 are totally new (no US exp), 2 have 4-7 years of US exp

gen migrant_nopast3 = (work_us_2010==1 & past5tot<3)
tab migrant_nopast3
* 97 migrants in 2010 with less than 3 years in the US in the past 5 years

gen correct_nopast3_migrant = (correct_migrant==1 & migrant_nopast3==1)
tab correct_nopast3_migrant
* 91/97 correct

gen stayed_migrant_nopast3 = (stayed_migrant==1 & past5tot<3)
tab stayed_migrant_nopast3
* 84 stayed migrants with less than 3 years in the US in the past 5 years

gen correct_stayed_migrant_nopast3 = (correct_migrant==1 & stayed_migrant_nopast3==1)
tab correct_stayed_migrant_nopast3
* 84/84 correct

////////////////////////////////////////////////////////////////////////////////

set seed 1234

gen keep1 = (work_us_2009==0 & correct_prediction==1 & migrant_nopast3==1)
keep if migrant_nopast3==1 | keep1==1
sample 13 if keep1!=1, count
keep ind correct_prediction predicted_y
save "$stats/ids.dta", replace

use "$data/MexMigData.dta", clear 
merge m:1 ind using "$stats/ids.dta"
keep if _merge==3

keep ind year work_us correct_prediction predicted_y
egen id = group(ind)

tab id correct_prediction if year==2010
label define id_lbl 1 "Correct" 2 "Correct" 3 "Correct" 4 "Correct" ///
    5 "Correct" 6 "Incorrect" 7 "Correct" 8 "Correct" 9 "Correct" ///
    10 "Correct" 11 "Correct" 12 "Correct" 13 "Correct" 14 "Correct" ///
    15 "Correct" 16 "Correct" 17 "Correct" 18 "Correct" 19 "Correct" 20 "Correct"
label values id id_lbl

* Replace actual y with the prediction:
replace work_us = predicted_y if year == 2010
gen highlight = (year==2010)

sepscatter work_us year, separate(highlight) sort by(id, ///
    title("Trajectories and Predictions of 2010 Migrants with Limited Recent US Experience", size(medium)) ///
	note("These graphs show individuals with less than 3 years of US experience during the 5 years preceding 2010 (2005–2009).",size(vsmall)) ///
	caption("Of the 97 migrants in 2010 with limited recent US experience, my ML model correctly predicted 91. Predictions are highlighted in red.",size(vsmall)) ///
	legend(off)) mc(blue red) ms(O O) ///
    yla(0 "Mexico" 1 "US") xla(1980(10)2010) ///
    ytitle("", size(small)) xtitle("")
graph export "$stats/migrants_2010.png", replace
