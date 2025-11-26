"""Generate realistic Uganda-specific financial data for ML training"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Uganda-specific configurations
UGANDA_PROVIDERS = ['MTN Mobile Money', 'Airtel Money', 'M-Sente']
UGANDA_LOCATIONS = [
    'Kampala', 'Entebbe', 'Jinja', 'Mbarara', 'Gulu', 'Lira',
    'Mbale', 'Masaka', 'Hoima', 'Arua', 'Fort Portal', 'Soroti'
]

UGANDA_CATEGORIES = {
    'income': ['Salary', 'Business Income', 'Remittances', 'Agricultural Sales', 'Freelance Work'],
    'transport': ['Boda Boda', 'Matatu', 'Special Hire', 'Fuel'],
    'food': ['Supermarket', 'Local Market', 'Restaurant', 'Street Food'],
    'utilities': ['UMEME', 'NWSC', 'Airtime', 'Internet'],
    'education': ['School Fees', 'Books', 'Uniform', 'Transport to School'],
    'business': ['Stock Purchase', 'Business Supplies', 'Rent'],
    'healthcare': ['Clinic', 'Hospital', 'Pharmacy', 'Medical Insurance'],
    'entertainment': ['Mobile Data', 'DSTV', 'Cinema', 'Social'],
    'savings': ['SACCO', 'Bank Deposit', 'Investment']
}

UGANDA_MERCHANTS = {
    'Supermarket': ['Shoprite', 'Quality Supermarket', 'Capital Shoppers', 'Game Stores'],
    'Fuel': ['Total', 'Shell', 'Gapco'],
    'Restaurant': ['Java House', 'KFC', 'Cafe Javas', 'Pork Joint'],
    'Pharmacy': ['Vine Pharmacy', 'Medipharm', 'Rocket Health'],
    'Bank Deposit': ['Stanbic Bank', 'Centenary Bank', 'DFCU', 'Equity Bank'],
}

# Uganda economic ranges (in UGX)
INCOME_RANGES = {
    'low': (300_000, 800_000),      # Monthly
    'medium': (800_000, 2_500_000),
    'high': (2_500_000, 8_000_000)
}

def generate_user_profile(user_id, income_tier):
    """Generate a user profile with Uganda-specific characteristics"""

    income_min, income_max = INCOME_RANGES[income_tier]
    monthly_income = np.random.uniform(income_min, income_max)

    # Expense ratio varies by income tier
    expense_ratios = {'low': (0.85, 1.05), 'medium': (0.65, 0.85), 'high': (0.45, 0.65)}
    exp_min, exp_max = expense_ratios[income_tier]
    expense_ratio = np.random.uniform(exp_min, exp_max)

    return {
        'user_id': user_id,
        'monthly_income': monthly_income,
        'expense_ratio': expense_ratio,
        'location': random.choice(UGANDA_LOCATIONS),
        'provider': random.choice(UGANDA_PROVIDERS),
        'has_formal_job': income_tier in ['medium', 'high'] and random.random() > 0.3,
        'has_business': random.random() > 0.4,
        'financial_discipline': np.random.beta(2, 5) if income_tier == 'low' else np.random.beta(5, 2)
    }

def generate_transactions(profile, num_transactions=50):
    """Generate realistic Uganda mobile money transactions"""

    transactions = []
    current_balance = np.random.uniform(50_000, 500_000)
    start_date = datetime.now() - timedelta(days=180)

    for i in range(num_transactions):
        # Transaction timing (more frequent at month start for salaries)
        days_offset = np.random.exponential(3)
        transaction_date = start_date + timedelta(days=days_offset)
        start_date = transaction_date

        # Determine transaction type
        is_income = random.random() < 0.3  # 30% income transactions

        if is_income:
            # Income transaction
            category = random.choice(UGANDA_CATEGORIES['income'])

            if category == 'Salary' and profile['has_formal_job']:
                amount = profile['monthly_income'] * np.random.uniform(0.9, 1.1)
            elif category == 'Business Income' and profile['has_business']:
                amount = np.random.uniform(100_000, 2_000_000)
            elif category == 'Remittances':
                amount = np.random.uniform(50_000, 500_000)
            elif category == 'Agricultural Sales':
                amount = np.random.uniform(200_000, 3_000_000)
            else:
                amount = np.random.uniform(50_000, 500_000)

            current_balance += amount
            transaction_type = 'incoming'
            merchant = category

        else:
            # Expense transaction
            category_type = random.choice([
                'transport', 'food', 'utilities', 'education',
                'business', 'healthcare', 'entertainment', 'savings'
            ])
            category = random.choice(UGANDA_CATEGORIES[category_type])

            # Amount varies by category
            if category == 'Boda Boda':
                amount = np.random.uniform(2_000, 15_000)
            elif category == 'School Fees':
                amount = np.random.uniform(200_000, 2_000_000)
            elif category == 'Rent':
                amount = profile['monthly_income'] * np.random.uniform(0.2, 0.4)
            elif category == 'UMEME':
                amount = np.random.uniform(30_000, 300_000)
            elif category == 'Airtime':
                amount = np.random.uniform(5_000, 50_000)
            elif category in ['Supermarket', 'Local Market']:
                amount = np.random.uniform(20_000, 200_000)
            elif category in UGANDA_MERCHANTS:
                merchant_list = UGANDA_MERCHANTS[category]
                merchant = random.choice(merchant_list)
                amount = np.random.uniform(10_000, 500_000)
            else:
                amount = np.random.uniform(5_000, 200_000)

            # Apply financial discipline
            amount *= (1 + (1 - profile['financial_discipline']) * 0.5)

            current_balance -= amount
            transaction_type = 'outgoing'
            merchant = category if 'merchant' not in locals() else merchant

        # Ensure balance doesn't go too negative
        if current_balance < -100_000:
            current_balance += amount
            continue

        transactions.append({
            'transaction_id': f'TXN_{profile["user_id"]}_{i:04d}',
            'user_id': profile['user_id'],
            'timestamp': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'type': transaction_type,
            'amount': round(amount, 2),
            'balance': round(current_balance, 2),
            'category': category,
            'merchant_id': f'MERCH_{hash(merchant) % 1000:04d}',
            'merchant_name': merchant,
            'location': profile['location'],
            'provider': profile['provider']
        })

        merchant = None  # Reset for next iteration

    return transactions

def generate_airtime_purchases(profile, num_purchases=15):
    """Generate Uganda airtime purchase history"""

    purchases = []
    start_date = datetime.now() - timedelta(days=180)
    providers = ['MTN', 'Airtel']

    for i in range(num_purchases):
        days_offset = np.random.exponential(7)  # Average every 7 days
        purchase_date = start_date + timedelta(days=days_offset)
        start_date = purchase_date

        # Airtime amounts in Uganda (common denominations)
        amount = random.choice([5_000, 10_000, 20_000, 50_000, 100_000])

        purchases.append({
            'transaction_id': f'AIR_{profile["user_id"]}_{i:04d}',
            'user_id': profile['user_id'],
            'timestamp': purchase_date.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': amount,
            'provider': random.choice(providers)
        })

    return purchases

def generate_loan_history(profile, num_loans=3):
    """Generate Uganda loan history"""

    loans = []
    start_date = datetime.now() - timedelta(days=365)

    # Loan probability based on income tier and financial discipline
    loan_probability = (1 - profile['financial_discipline']) * 0.5

    if random.random() > loan_probability and num_loans > 0:
        num_loans = max(1, num_loans - 2)

    for i in range(num_loans):
        days_offset = np.random.uniform(30, 300)
        loan_date = start_date + timedelta(days=days_offset)

        # Loan amount relative to monthly income
        loan_amount = profile['monthly_income'] * np.random.uniform(0.5, 3.0)

        # Interest rates in Uganda (monthly)
        interest_rate = np.random.uniform(0.08, 0.15)  # 8-15% monthly

        # Loan term (months)
        term_months = random.choice([1, 2, 3, 6, 12])

        # Default probability
        default_prob = (1 - profile['financial_discipline']) * 0.3
        is_default = random.random() < default_prob

        # Repayment amount
        total_repayment = loan_amount * (1 + interest_rate) ** term_months

        loans.append({
            'loan_id': f'LOAN_{profile["user_id"]}_{i:04d}',
            'user_id': profile['user_id'],
            'loan_date': loan_date.strftime('%Y-%m-%d'),
            'loan_amount': round(loan_amount, 2),
            'interest_rate': round(interest_rate, 4),
            'term_months': term_months,
            'repayment_amount': round(total_repayment, 2),
            'is_default': int(is_default),
            'provider': random.choice(['Bank', 'SACCO', 'Microfinance', 'Mobile Loan'])
        })

    return loans

def generate_eligibility_labels(profile):
    """Generate loan eligibility label based on profile"""

    # Calculate score based on multiple factors
    score = 0

    # Income factor
    if profile['monthly_income'] > 1_500_000:
        score += 30
    elif profile['monthly_income'] > 800_000:
        score += 20
    else:
        score += 10

    # Expense ratio factor
    if profile['expense_ratio'] < 0.6:
        score += 25
    elif profile['expense_ratio'] < 0.8:
        score += 15
    else:
        score += 5

    # Financial discipline
    score += profile['financial_discipline'] * 30

    # Employment
    if profile['has_formal_job']:
        score += 15

    # Eligible if score > 60
    is_eligible = 1 if score > 60 else 0

    return {
        'user_id': profile['user_id'],
        'is_eligible': is_eligible,
        'credit_score': round(score, 2),
        'monthly_income': round(profile['monthly_income'], 2),
        'location': profile['location']
    }

# Generate data for 100 users
print("Generating Uganda-specific financial data...")
print("=" * 80)

num_users = 100
all_transactions = []
all_airtime = []
all_loans = []
all_eligibility = []

# Distribute users across income tiers
income_distribution = ['low'] * 40 + ['medium'] * 45 + ['high'] * 15

for i in range(num_users):
    user_id = f'user_{i}'
    income_tier = income_distribution[i]

    # Generate profile
    profile = generate_user_profile(user_id, income_tier)

    # Generate transactions
    num_trans = np.random.randint(40, 80)
    transactions = generate_transactions(profile, num_trans)
    all_transactions.extend(transactions)

    # Generate airtime
    num_airtime = np.random.randint(10, 25)
    airtime = generate_airtime_purchases(profile, num_airtime)
    all_airtime.extend(airtime)

    # Generate loans
    num_loans = np.random.randint(0, 5)
    loans = generate_loan_history(profile, num_loans)
    all_loans.extend(loans)

    # Generate eligibility
    eligibility = generate_eligibility_labels(profile)
    all_eligibility.append(eligibility)

    if (i + 1) % 20 == 0:
        print(f"Generated data for {i + 1} users...")

# Convert to DataFrames
df_transactions = pd.DataFrame(all_transactions)
df_airtime = pd.DataFrame(all_airtime)
df_loans = pd.DataFrame(all_loans)
df_eligibility = pd.DataFrame(all_eligibility)

# Save to CSV
print("\nSaving data to CSV files...")
df_transactions.to_csv('data/raw/mobile_money_transactions.csv', index=False)
df_airtime.to_csv('data/raw/airtime_purchases.csv', index=False)
df_loans.to_csv('data/raw/loan_history.csv', index=False)
df_eligibility.to_csv('data/raw/loan_eligibility.csv', index=False)

print("\n" + "=" * 80)
print("Uganda Financial Data Generation Complete!")
print("=" * 80)
print(f"Mobile Money Transactions: {len(df_transactions):,} records")
print(f"Airtime Purchases: {len(df_airtime):,} records")
print(f"Loan History: {len(df_loans):,} records")
print(f"Users: {len(df_eligibility)} profiles")
print("\nSample Statistics (UGX):")
print(f"  Average Transaction: {df_transactions['amount'].mean():,.0f}")
print(f"  Total Volume: {df_transactions['amount'].sum():,.0f}")
print(f"  Eligible Users: {df_eligibility['is_eligible'].sum()}/{len(df_eligibility)}")
print(f"  Average Income: {df_eligibility['monthly_income'].mean():,.0f}")
print("\nTop Locations:")
print(df_transactions['location'].value_counts().head())
print("\nTop Categories:")
print(df_transactions['category'].value_counts().head(10))
print("=" * 80)
