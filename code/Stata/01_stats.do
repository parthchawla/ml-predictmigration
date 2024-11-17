* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Sep 30, 2024
*/
* ------------------------------------------------------------------------------

clear all
macro drop _all

if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
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
gen in_1980 = (year == 1980)
gen in_1990 = (year == 1990)
gen in_2000 = (year == 2000)
gen in_2010 = (year == 2010)

preserve
    collapse (max) in_1980 in_1990 in_2000 in_2010, by(ind)

    * Create pattern variable
    gen pattern = ""

    * Single decade patterns
    replace pattern = "1980 only" if in_1980==1 & in_1990==0 & in_2000==0 & in_2010==0
    replace pattern = "1990 only" if in_1980==0 & in_1990==1 & in_2000==0 & in_2010==0
    replace pattern = "2000 only" if in_1980==0 & in_1990==0 & in_2000==1 & in_2010==0
    replace pattern = "2010 only" if in_1980==0 & in_1990==0 & in_2000==0 & in_2010==1

    * Two decade patterns
    replace pattern = "1980 & 1990" if in_1980==1 & in_1990==1 & in_2000==0 & in_2010==0
    replace pattern = "1980 & 2000" if in_1980==1 & in_1990==0 & in_2000==1 & in_2010==0
    replace pattern = "1980 & 2010" if in_1980==1 & in_1990==0 & in_2000==0 & in_2010==1
    replace pattern = "1990 & 2000" if in_1980==0 & in_1990==1 & in_2000==1 & in_2010==0
    replace pattern = "1990 & 2010" if in_1980==0 & in_1990==1 & in_2000==0 & in_2010==1
    replace pattern = "2000 & 2010" if in_1980==0 & in_1990==0 & in_2000==1 & in_2010==1

    * Three decade patterns
    replace pattern = "1980, 1990, 2000" if in_1980==1 & in_1990==1 & in_2000==1 & in_2010==0
    replace pattern = "1980, 1990, 2010" if in_1980==1 & in_1990==1 & in_2000==0 & in_2010==1
    replace pattern = "1980, 2000, 2010" if in_1980==1 & in_1990==0 & in_2000==1 & in_2010==1
    replace pattern = "1990, 2000, 2010" if in_1980==0 & in_1990==1 & in_2000==1 & in_2010==1

    * All decades
    replace pattern = "All decades" if in_1980==1 & in_1990==1 & in_2000==1 & in_2010==1

    tab pattern, missing
restore

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

exit

sort ind year

* Create lagged migration status (previous period)
by ind: gen migrant_t1 = work_us[_n-1] if _n > 1   // previous period (t-1)
by ind: gen migrant_t2 = work_us                    // current period (t)

* Create simple change indicator
by ind: gen change_status = (work_us != work_us[_n-1]) if _n > 1

* Create detailed transition categories
gen transition = ""
replace transition = "Stay Non-migrant" if migrant_t1==0 & migrant_t2==0
replace transition = "Start Migration" if migrant_t1==0 & migrant_t2==1
replace transition = "Stop Migration" if migrant_t1==1 & migrant_t2==0
replace transition = "Stay Migrant" if migrant_t1==1 & migrant_t2==1
replace transition = "First Observation" if migrant_t1==.

* Quick check of our variables
list ind year work_us migrant_t1 migrant_t2 transition in 1/20 if change_status!=.

* Tabulate to see transitions
tab transition year, row


exit


* 1. Sankey-style transition visualization using connected lines
preserve
    * Calculate transition percentages
    collapse (mean) work_us, by(year)
    
    * Create connected line plot
    twoway (connected work_us year), ///
        title("Share of Population in US Over Time") ///
        ylabel(0(.1)1, format(%3.2f)) ///
        xlabel(1980(10)2010) ///
        ytitle("Share in US") ///
        xtitle("Year") ///
        connect(direct) // or use 'connect(stairstep)' for discrete jumps
        
    //graph export "migration_trend.png", replace
restore

* 2. Stacked bar chart of transition types by year
preserve
    * Calculate shares of each transition type
    collapse (count) n=ind, by(year transition)
    bysort year: egen total=sum(n)
    gen share = n/total
    
    * Create stacked bar graph
    graph bar (asis) share, over(transition) over(year) stack ///
        title("Migration Status Transitions Over Time") ///
        ytitle("Share of Population") ///
        legend(cols(1)) ///
        bar(1, color(navy)) bar(2, color(green)) ///
        bar(3, color(red)) bar(4, color(orange))
        
    graph export "transition_shares.png", replace
restore

* 3. Individual trajectory plot for a random sample
preserve
    * Keep a random sample of individuals for clearer visualization
    egen tag = tag(ind)
    gen random = runiform() if tag==1
    sort random
    keep if random <= 0.01 // Keep 1% random sample
    drop random tag
    
    * Create individual trajectory plot
    sort ind year
    twoway (connect work_us year, connect(direct)), ///
        by(ind, compact legend(off)) ///
        title("Individual Migration Trajectories") ///
        subtitle("1% Random Sample") ///
        ylabel(0 1) xlabel(1980(10)2010) ///
        ytitle("Migration Status")
        
    graph export "individual_trajectories.png", replace
restore

* 4. Transition matrix heatmap
preserve
    * Create transition matrix
    tab migrant_t1 migrant_t2, matcell(trans_matrix)
    
    * Convert to percentages
    mata: st_matrix("trans_pct", st_matrix("trans_matrix") :/ rowsum(st_matrix("trans_matrix")))
    
    * Create heatmap
    heatplot trans_pct, colors(white navy) ///
        title("Migration Status Transition Matrix") ///
        xlabel(0 "Non-Migrant" 1 "Migrant") ///
        ylabel(0 "Non-Migrant" 1 "Migrant", angle(0)) ///
        cell(format(%4.3f))
        
    graph export "transition_heatmap.png", replace
restore

* 5. Entry/Exit flow diagram
preserve
    * Calculate entries and exits by year
    bysort ind (year): gen entry = _n==1
    bysort ind (year): gen exit = _n==_N
    
    collapse (sum) entries=entry exits=exit, by(year)
    
    * Create combined entry/exit plot
    twoway (bar entries year, color(green)) ///
           (bar exits year, color(red)), ///
        title("Migration Flows Over Time") ///
        legend(label(1 "Entries") label(2 "Exits")) ///
        ylabel(, angle(0)) ///
        xtitle("Year") ytitle("Number of Individuals")
        
    graph export "migration_flows.png", replace
restore
