import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn")

df_acc = pd.read_pickle("../data/accounting.pkl")
df_ie = pd.read_pickle("../data/df.pkl")

def date_to_fraction(dt):
  return int(dt.year)+(int(dt.month)-1)/12+(int(dt.day))/30/12

df_acc["Date Fraction"]=df_acc["Date"].apply(date_to_fraction)

plt.plot(df_ie["Date Fraction"], df_ie["E"], label="(Nominal) E")
plt.plot(df_acc["Date Fraction"], df_acc["PFE"], label="PFE") # Nominal
plt.axvline(x=2000, color='r', linestyle='--')
plt.legend(); plt.savefig("Earning_series.png"); plt.clf()

# Marge the dataframes on Date-oFraction
# Use int(Date Fraction * 100) as a proxy to perform merge
initial_df_ie_len = len(df_ie)

df_ie["proxy"]=(df_ie["Date Fraction"]*100).astype(int)
df_acc["proxy"]=(df_acc["Date Fraction"]*100).astype(int)
df_acc.drop(columns=["Date Fraction"], inplace=True)  # Avoid Multiple columns
df_acc.drop_duplicates(subset=["proxy"], inplace=True)
df_ie = pd.merge(left=df_ie, right=df_acc, on="proxy", how="left")
df_ie.loc[df_ie.PFE.isna(), "PFE"]=df_ie.loc[df_ie.PFE.isna()]["E"]

assert initial_df_ie_len==len(df_ie)

df_ie["Real PFE"] = df_ie["PFE"]*(df_ie["Real Earnings"]/df_ie["E"])
# Plot the two Real Earning Serieses
plt.plot(df_ie["Date Fraction"], df_ie["Real Earnings"], label="(Nominal) E")
plt.plot(df_ie["Date Fraction"], df_ie["Real PFE"], label="PFE") # Real
plt.axvline(x=2000, color='r', linestyle='--')
plt.legend(); plt.savefig("Earning_series_real.png"); plt.clf()

df_ie["PFE10"] = df_ie["Real PFE"].rolling(window=120, min_periods=120).mean()
df_ie["Accounting Adjusted CAPE"] = df_ie["Real Price"]/df_ie["PFE10"]

plt.plot(df_ie["Date Fraction"], df_ie["CAPE"],label="CAPE (GAAP)")
plt.plot(df_ie["Date Fraction"], df_ie["Accounting Adjusted CAPE"], label="CAPE (Pro-Forma)")
plt.legend(); plt.savefig("CAPE_comparison.png"); plt.clf()

