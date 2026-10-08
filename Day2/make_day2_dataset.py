"""Generate the Day 2 'messy loan default' dataset with planted defects.
Usage: python make_day2_dataset.py [seed] [outfile]
Use a different seed per student or batch to get personalised variants."""
import sys
import numpy as np, pandas as pd

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
out = sys.argv[2] if len(sys.argv) > 2 else "loan_default_day2.csv"
rng = np.random.default_rng(seed)
n = 5000

first = ["Amit","Priya","Ravi","Kiran","Meera","Jignesh","Hetal","Nirav","Pooja","Bhavesh","Dipti","Mahesh","Kinjal","Sanjay","Rekha","Harsh","Nisha","Dhruv","Anita","Vijay"]
last = ["Patel","Joshi","Shah","Mehta","Parmar","Solanki","Vora","Bhatt","Rathod","Chauhan","Dave","Trivedi","Gohil","Zala","Jadeja"]
regions = ["Saurashtra","Kutch","North Gujarat","Central Gujarat","South Gujarat"]

age = np.clip(rng.normal(38, 10, n).round(), 21, 65).astype(int)
emp = rng.choice(["Salaried","Self-employed","Business","Daily wage"], n, p=[.40,.25,.20,.15])
emp_years = np.clip(age - 21 - rng.integers(0, 8, n), 0, None)
base_inc = {"Salaried":38000,"Self-employed":32000,"Business":55000,"Daily wage":14000}
income = np.array([rng.lognormal(np.log(base_inc[e]), 0.35) for e in emp]).round(-2)
loan_amount = (income * 12 * rng.uniform(0.3, 1.6, n)).round(-3)
credit = np.clip(rng.normal(680, 70, n), 300, 900).round().astype(int)
region = rng.choice(regions, n, p=[.30,.15,.20,.20,.15])
loan_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")

burden = np.clip(loan_amount / (income * 12), 0, 2)
z = (0.9*burden - (credit-680)/80*0.9 + np.where(emp=="Daily wage", .5, 0) - np.where(emp=="Salaried", .3, 0)
     + rng.normal(0, .6, n))
lo, hi = -10, 10
for _ in range(60):
    mid = (lo+hi)/2
    if (1/(1+np.exp(-(z+mid)))).mean() > 0.15: hi = mid
    else: lo = mid
default = (rng.uniform(0,1,n) < 1/(1+np.exp(-(z+mid)))).astype(int)

# --- leaky columns (built from the TRUE outcome) ---
collection_calls = np.where(default==1, rng.poisson(6, n)+1, rng.poisson(0.2, n))
recovery_amount = np.where(default==1, (loan_amount*rng.uniform(.1,.6,n)).round(-2), 0.0)
closed = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
has_close = np.where(default==1, rng.uniform(0,1,n) < .92, rng.uniform(0,1,n) < .03)
closed[has_close] = loan_date[has_close] + pd.to_timedelta(rng.integers(200, 400, has_close.sum()), unit="D")
risk_flag = np.where(default==1, np.where(rng.uniform(0,1,n)<.90,"high","low"),
                     np.where(rng.uniform(0,1,n)<.08,"high","low"))
late_but_repaid = np.where((default==0) & (rng.uniform(0,1,n) < .07), "Y", "N")

# --- label noise ---
label = default.copy()
flip = rng.uniform(0,1,n) < .05
label[flip] = 1 - label[flip]
rule_rows = (late_but_repaid=="Y")
in_strict = np.isin(region, ["Saurashtra","Kutch"])
label[rule_rows & in_strict & (rng.uniform(0,1,n) < .8)] = 1
label[rule_rows & ~in_strict & (rng.uniform(0,1,n) < .9)] = 0

# --- loan_id: defaulters cluster late ---
order = np.argsort(default*1.0 + rng.uniform(0, 1.5, n))
loan_id = np.empty(n, int); loan_id[order] = np.arange(1, n+1) + 10000

df = pd.DataFrame(dict(
    loan_id=loan_id,
    applicant_name=[f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)],
    age=age, employment_type=emp, employment_years=emp_years,
    monthly_income=income, loan_amount=loan_amount, loan_date=loan_date.strftime("%Y-%m-%d"),
    credit_score=credit, region=region, collection_calls=collection_calls,
    recovery_amount=recovery_amount,
    account_closed_date=closed.dt.strftime("%Y-%m-%d"), risk_flag=risk_flag,
    late_but_repaid=late_but_repaid, defaulted_12m=label))

# --- messy values ---
se = df.employment_type == "Self-employed"
variants = rng.choice(["self employed","Self-Employed","SE","Self employed"], n)
mask_se = se & (rng.uniform(0,1,n) < .35)
df.loc[mask_se, "employment_type"] = variants[mask_se]
df.loc[rng.choice(n, 15, replace=False), "age"] = 0
df.loc[rng.uniform(0,1,n) < .06, "employment_years"] = 999
p_miss = np.where(label==1, .35, .08)
df.loc[rng.uniform(0,1,n) < p_miss, "monthly_income"] = np.nan
lakh = rng.choice(df.index[df.monthly_income.notna()], 75, replace=False)
df.loc[lakh, "monthly_income"] = (df.loc[lakh, "monthly_income"]/100000).round(3)
df.loc[rng.choice(df.index[df.monthly_income.notna()], 1), "monthly_income"] = 9999999

# --- near duplicates (new id, different formatting) ---
nd = df.sample(int(.03*n), random_state=seed).copy()
nd["applicant_name"] = [m.upper() if rng.uniform() < .5 else f" {m.lower()} " for m in nd.applicant_name]
nd["monthly_income"] = (nd.monthly_income/1000).round()*1000
nd["loan_id"] = np.arange(df.loan_id.max()+1, df.loan_id.max()+1+len(nd))
# --- exact duplicates ---
ed = df.sample(int(.04*n), random_state=seed+1)
full = pd.concat([df, nd, ed], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
full.to_csv(out, index=False)
print(out, full.shape, "default rate", round(full.defaulted_12m.mean(), 3))
