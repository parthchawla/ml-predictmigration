* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Sep 30, 2024
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

gen migrant_type = "Never Migrant" if totalus==0 // 84%
replace migrant_type = "Short-term Migrant" if totalus>=1 & totalus<=3 // 5%
replace migrant_type = "Medium-term Migrant" if totalus>=4 & totalus<=10 // 6%
replace migrant_type = "Long-term Migrant" if totalus>10 // 4%

sort ind year
order ind year migrant_type
tab year migrant_type
tab year work_us

egen times_observed = count(year), by(ind)

* Look at entry and exit
sort ind year
by ind: gen entry = _n==1
by ind: gen exit = _n==_N
tab year if entry==1
tab year if exit==1

* Generate indicator for individuals present in all years
rename balanced balanced_02_07_10
gen balanced = (times_observed == 31)
label var balanced "1 if individual in all years"
tab balanced_02_07_10
tab balanced

* Track changes in migration status
sort ind year
by ind: gen change_status = work_us != work_us[_n-1] if _n>1
tab year change_status

* Create lagged migration status (previous period)
// by ind: gen migrant_t1 = work_us[_n-1] if _n > 1   // previous period (t-1)
// by ind: gen migrant_t2 = work_us                    // current period (t)

* Different from above because it doesn't ignore missing previous values
xtset ind year
gen migrant_t1 = L1.work_us  // previous period (t-1)
gen migrant_t2 = work_us     // current period (t)

* Create detailed transition categories
gen transition = "Stayed in Mexico" if migrant_t1==0 & migrant_t2==0
replace transition = "Moved to the US" if migrant_t1==0 & migrant_t2==1
replace transition = "Went back to Mexico" if migrant_t1==1 & migrant_t2==0
replace transition = "Stayed in the US" if migrant_t1==1 & migrant_t2==1
replace transition = "First observed" if migrant_t1==.

* Tabulate to see transitions
tab transition year, row
//encode transition

* 1. Share working in the US over time
preserve
    collapse (mean) work_us, by(year)
    twoway (connected work_us year), ///
        title("Share of Population in US Over Time", size(medium)) ///
        yla(,labs(small)) xla(1980(10)2010, labs(small)) ///
        ytitle("Share in US", size(small)) xtitle("")    
    graph export "$stats/migration_trend.png", replace
restore

* 2. Stacked bar chart of transition types by year
    * Calculate shares of each transition type
preserve
    collapse (count) n = ind, by(year transition)
    bys year: egen total = sum(n)
    gen share = n/total
	
	gen order = 1 if transition=="Moved to the US"
	replace order = 2 if transition=="Stayed in the US"
	replace order = 3 if transition=="Stayed in Mexico"
	replace order = 4 if transition=="First observed"
	replace order = 5 if transition=="Went back to Mexico"

    * Create stacked bar graph
	graph hbar (asis) share, name(g1) ///
		over(transition,label(labs(vsmall)) sort(order)) over(year,label(labs(vsmall))) ///
		title("Transitions", size(medium)) intensity(70) ///
		ytitle("Share of Population", size(small)) ///
		b1title(,size(small)) yla(,labs(small)) stack asyvars ///
		legend(size(small) rows(2) pos(6)) ///
		bar(1,color(dimgray)) bar(2,color(maroon)) bar(3,color(eltblue)) ///
		bar(4,color(erose)) bar(5,color(edkblue))
restore

* 3. Stacked bar chart of migrant types by year
    collapse (count) n = ind, by(year migrant_type)
    bys year: egen total = sum(n)
    gen share = n/total

	gen order = 1 if migrant_type=="Never Migrant"
	replace order = 2 if migrant_type=="Short-term Migrant"
	replace order = 3 if migrant_type=="Medium-term Migrant"
	replace order = 4 if migrant_type=="Long-term Migrant"

    * Create stacked bar graph
	graph hbar (asis) share, name(g2) ///
		over(migrant_type,label(labs(vsmall)) sort(order)) over(year,label(labs(vsmall))) ///
		title("Types", size(medium)) intensity(70) ///
		ytitle("Share of Population", size(small)) ///
		b1title(,size(small)) yla(,labs(small)) stack asyvars ///
		legend(size(small) rows(2) pos(6)) ///
		bar(1,color(orange_red)) bar(2,color(dkorange)) ///
		bar(3,color(eltblue)) bar(4,color(orange))

graph combine g2 g1, ///
title("Share of Migrant Types and Transitions Over Time", size(medium))
graph export "$stats/migrantion_shares.png", replace
