import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
mlp.style.use("seaborn")

df=pd.read_pickle("df.pkl")
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
df["10 yr. MAE growth"] = np.log(df.E10.shift(-12)/df.E10.shift())
plt.xscale('log')
plt.scatter(df["CAPE"], df["10 yr. MAE growth"])
plt.ylabel("10 Year MA(E) Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.savefig("E6A.png")
plt.clf()

# Plot log(CAPE) log(P_{t+10*12} /P_t)
df["10 yr. P growth"] = np.log(df.P.shift(-120)/df.P)
plt.xscale('log')
plt.scatter(df["CAPE"], df["10 yr. P growth"])
plt.ylabel("10 Year P Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B.png")
plt.clf()

# Plot Sparsely
df_sparse=df[::6]
df_sparse_old=df_sparse.loc[df_sparse["Date Fraction"]<=1991]
df_sparse_new=df_sparse.loc[df_sparse["Date Fraction"]>1991]
# plt.xscale('log')
plt.scatter(df_sparse_old["CAPE"], df_sparse_old["10 yr. MAE growth"], label="<=1991")
plt.scatter(df_sparse_new["CAPE"], df_sparse_new["10 yr. MAE growth"], label=">1991")
plt.ylabel("10 Year MA(E) Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.legend()
plt.savefig("E6A_sparse.png")
plt.clf()
# plt.xscale('log')
plt.scatter(df_sparse_old["CAPE"], df_sparse_old["10 yr. P growth"], label="<=1991")
plt.scatter(df_sparse_new["CAPE"], df_sparse_new["10 yr. P growth"], label=">1991")
plt.legend()
plt.ylabel("10 Year Price Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B_sparse.png")
plt.clf()

# Plot Sparsely With Regression Lines
df_sparse=df[::6]
df_sparse_old=df_sparse.loc[df_sparse["Date Fraction"]<=1991]
df_sparse_new=df_sparse.loc[df_sparse["Date Fraction"]>1991]
# plt.xscale('log')
sns.regplot(x="CAPE", y="10 yr. MAE growth",data=df_sparse_old, label="<=1991")
sns.regplot(x="CAPE", y="10 yr. MAE growth",data=df_sparse_new, label="<=1991")
plt.ylabel("10 Year MA(E) Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year MA(E) Growth vs CAPE")
plt.legend()
plt.savefig("E6A_sparse_sns.png")
plt.clf()
# plt.xscale('log')
sns.regplot(x="CAPE", y="10 yr. P growth",data=df_sparse_old, label="<=1991")
sns.regplot(x="CAPE", y="10 yr. P growth",data=df_sparse_new, label="<=1991")
plt.legend()
plt.ylabel("10 Year Price Growth")
plt.xlabel("CAPE, P/E10")
plt.title("Ten Year Price Growth vs CAPE")
plt.savefig("E6B_sparse_sns.png")
plt.clf()
