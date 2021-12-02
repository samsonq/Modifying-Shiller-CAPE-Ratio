import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
mlp.style.use("seaborn")

df=pd.read_pickle("../data/df.pkl")
df['E10'] = df['Real Earnings'].rolling(window=120, min_periods=120).mean()
df["P/E10"] = df['Real Price'] / df['E10']
# Plot P
plt.plot(df["Date Fraction"], df["Real Price"])
plt.title("Historical S&P Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price, P")
plt.savefig("SPY.png")
plt.clf()
# Plot E
plt.plot(df["Date Fraction"], df["Real Earnings"], label="E")
plt.plot(df["Date Fraction"], df["E10"], label="E10")
plt.title("Historical S&P Earnings")
plt.xlabel("Date")
plt.ylabel("Earnings, E")
plt.legend()
plt.savefig("E.png")
plt.clf()
# Plot D
plt.plot(df["Date Fraction"], df["Real Dividend"])
plt.title("Historical S&P Dividends")
plt.xlabel("Date")
plt.ylabel("Dividends, D")
plt.savefig("D.png")
plt.clf()
# Plot CAPE
plt.plot(df["Date Fraction"], df["CAPE"], label="CAPE")
plt.plot(df["Date Fraction"], df["P/E10"], label="CAPE_Reconstructed")
plt.title("Historical S&P CAPE")
plt.xlabel("Date")
plt.ylabel("P/E10")
plt.legend()
plt.savefig("CAPE.png")
plt.clf()

# Plot log(CAPE) log(E10_t+1/E10_t)
df["10 yr. MAE growth"] = np.log(df.E10.shift(-12)/df.E10.shift())/10
plt.xscale('log')
plt.scatter(df["CAPE"], df["10 yr. MAE growth"])
plt.ylabel("10 Year MA(E) Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.savefig("E6A.png")
plt.clf()

# Plot log(CAPE) log(P_{t+10*12} /P_t)
df["10 yr. P growth"] = np.log(df.P.shift(-120)/df.P)/10
plt.xscale('log')
plt.scatter(df["CAPE"], df["10 yr. P growth"])
plt.ylabel("10 Year P Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B.png")
plt.clf()

# Plot Sparsely
df_sparse=df[::6]
df_sparse_old=df_sparse.loc[df_sparse["Date Fraction"]<=1996]
df_sparse_new=df_sparse.loc[df_sparse["Date Fraction"]>1996]
# plt.xscale('log')
plt.scatter(df_sparse_old["CAPE"], df_sparse_old["10 yr. MAE growth"], label="1881-2006")
plt.scatter(df_sparse_new["CAPE"], df_sparse_new["10 yr. MAE growth"], label="2007-2021")
plt.ylabel("10 Year MA(E) Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.legend()
plt.savefig("E6A_sparse.png")
plt.clf()
# plt.xscale('log')
plt.scatter(df_sparse_old["CAPE"], df_sparse_old["10 yr. P growth"], label="1881-2006")
plt.scatter(df_sparse_new["CAPE"], df_sparse_new["10 yr. P growth"], label="2007-2021")
plt.legend()
plt.ylabel("10 Year Price Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B_sparse.png")
plt.clf()

# Plot Sparsely With Regression Lines
# plt.xscale('log')
sns.regplot(x="CAPE", y="10 yr. MAE growth",data=df_sparse_old, label="1881-2006")
sns.regplot(x="CAPE", y="10 yr. MAE growth",data=df_sparse_new, label="2007-2021")
plt.ylabel("10 Year MA(E) Growth (Annualized)")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.legend()
plt.savefig("E6A_sparse_sns.png")
plt.clf()
# plt.xscale('log')
sns.regplot(x="CAPE", y="10 yr. P growth",data=df_sparse_old, label="1886-2006")
sns.regplot(x="CAPE", y="10 yr. P growth",data=df_sparse_new, label="2007-2021")
plt.legend()
plt.ylabel("10 Year Price Growth (Anualized)")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B_sparse_sns.png")
plt.clf()

# Plot In buckets
df["CAPE quintile"]=pd.qcut(df["CAPE"], 5, labels=False)
sns.boxplot(x="CAPE quintile", y="10 yr. P growth", data=df)
plt.savefig("CAPE_decile_all.png"); plt.clf()
df["time"]="1886-2006"
df.loc[df["Date Fraction"]>1996, "time"]="2007-2021"
sns.boxplot(x="CAPE quintile", y="10 yr. P growth", data=df, hue="time")
plt.savefig("CAPE_decile_hued.png"); plt.clf()
# Changing the decile to be time-wise
df.loc[df["Date Fraction"]<=1996, "CAPE quintile"]=pd.qcut(df.loc[df["Date Fraction"]<=1996]["CAPE"], 5, labels=False)
df.loc[df["Date Fraction"]>1996, "CAPE quintile"]=pd.qcut(df.loc[df["Date Fraction"]>1996]["CAPE"], 5, labels=False)
df_old=df.loc[df["Date Fraction"]<=1996]
df_new=df.loc[df["Date Fraction"]>1996]
sns.boxplot(x="CAPE quintile", y="10 yr. P growth", data=df_old); sns.regplot(x="CAPE quintile", y="10 yr. P growth", scatter=False, data=df_old); plt.title("1886-2006");plt.savefig("CAPE_decile_old.png"); plt.clf()
sns.boxplot(x="CAPE quintile", y="10 yr. P growth", data=df_new); sns.regplot(x="CAPE quintile", y="10 yr. P growth", scatter=False, data=df_new); plt.title("2007-2021");plt.savefig("CAPE_decile_new.png"); plt.clf()
sns.boxplot(x="CAPE quintile", y="10 yr. P growth", data=df, hue="time"); sns.regplot(x="CAPE quintile", y="10 yr. P growth", scatter=False, data=df_old);sns.regplot(x="CAPE quintile", y="10 yr. P growth", scatter=False, data=df_new); plt.savefig("CAPE_quintile_timewise_hued.png"); plt.clf()
