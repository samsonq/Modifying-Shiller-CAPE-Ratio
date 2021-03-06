# Modifying-Shiller-CAPE-Ratio

Developed by Nobel laureate Robert Shiller, and John Campbell, the Shiller Cyclically Adjusted P/E (CAPE) is one of the most established metrics used to evaluate whether the equity market is over-, under-, or fairly valued.  The framework gained popularity amidst the dot-com bubble when it correctly warned that the U.S. sto¬ck market had run too far ahead of itself.

The CAPE’s elegance lies in its simplicity.  The commonly used P/E ratio reflects only one year of earnings, which could provide false si¬gnals in periods of cyclical extremes.  For example, during economic recessions, the drop in corporate earnings could temporarily elevate the market P/E which normally could be interpreted as equities being overvalued.  To account for such cyclical aberrations, the CAPE takes the average of the last ten years of market earnings (adjusted for inflation), and then divides the current market index price by that adjusted earnings.  It thus attempts to capture relative valuations levels over business cycles rather than just one year of results.

Today, the CAPE is also widely used by investors for framing long-run equity market return expectations at any point in time.  Using data that spanned more than a century, Shiller & Campbell back-tested the CAPE by regressing 10-year real stock returns against the long-term average market CAPE ratio and found that the CAPE approach was statistically significant in predicting long-run equity returns.

Nevertheless, the CAPE does have its limitations.  At its core, the CAPE is mean reverting in orientation and is thus anchored on empirical norms.  Recent critiques of this framework posit that several important presumptions may no longer hold to a significant degree.  These include, but are not limited to, i) interest rate regimes; ii) accounting rules; iii) proposed changes to tax regulations and rates, especially as it pertains international income of U.S. multi-national corporations; and iv) the composition of market index.

The purpose of this exercise is to examine these and other significant changes, and to come up ways to account for the more meaningful ones such that a prototypical Adjusted CAPE could be more instructive on informing how much the U.S. equity market is over- or under-valued, and to provide a perspective on expected equity returns in the current environment.


## Usage Instructions

The code for performing the CAPE regressions and visualizing the results are located in the `notebooks` folder within the Jupyter Notebooks. Each of the 4 topics our team covered has a notebook to run: accounting, interest rates, market composition, taxes. Before running, please install the necessary Python libraries located in 'requirements.txt'. To do so, run: `$ pip install -r requirements.txt`. The data that we used to perform the analysis are all located in the `data` folder; this folder contains the S&P returns data and CAPE ratios that Shiller used in his original paper, along with any supplemental data that our team collected to compute an Adjusted CAPE value. This data is recorded in monthly intervals and ranges from 1871-2021.
