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

* No. of individuals who ever migrated
// collapse (max) work_us, by(ind)
// collapse (sum) work_us

* No. of households who ever had any migrating household member
// collapse (max) work_us, by(numc)
// collapse (mean) work_us

* Share of individuals migrating per year
// collapse (count) total=work_us (sum) migrants=work_us (mean) share=work_us, by(year)

//preserve
* Share of households with at least one migrating member per year
	collapse (max) work_us, by(year numc)
	collapse (mean) work_us, by(year)
    twoway (connected work_us year), ///
        title("Share of Households With at Least One Member in the US Over Time", size(medium)) ///
        yla(0(0.05)0.3,labs(small)) xla(1980(10)2010, labs(small)) ///
        ytitle("Share in the US", size(small)) xtitle("")    
    graph export "$stats/hh_migration_trend.png", replace
//restore 0(0.02)0.12



