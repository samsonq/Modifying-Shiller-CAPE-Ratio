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
plt.legend(); plt.savefig("Earning_series.png"); plt.clf()

