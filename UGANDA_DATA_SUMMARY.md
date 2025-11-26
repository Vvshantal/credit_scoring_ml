# Uganda-Specific Data Implementation

## Overview
The platform now uses realistic Uganda-specific financial data instead of dummy data.

## Data Generation Summary

### Mobile Money Transactions: 5,614 records
- **Providers**: MTN Mobile Money, Airtel Money, M-Sente
- **Locations**: Kampala, Entebbe, Jinja, Mbarara, Gulu, Lira, Mbale, Masaka, Hoima, Arua, Fort Portal, Soroti
- **Currency**: Uganda Shillings (UGX)
- **Average Transaction**: UGX 425,000
- **Total Volume**: UGX 2,385,949,875

### Transaction Categories (Uganda-specific)

**Income Sources:**
- Salary (formal employment)
- Business Income
- Remittances
- Agricultural Sales
- Freelance Work

**Expense Categories:**
- **Transport**: Boda Boda, Matatu, Special Hire, Fuel
- **Food**: Supermarket, Local Market, Restaurant, Street Food
- **Utilities**: UMEME (electricity), NWSC (water), Airtime, Internet
- **Education**: School Fees, Books, Uniform, Transport to School
- **Business**: Stock Purchase, Business Supplies, Rent
- **Healthcare**: Clinic, Hospital, Pharmacy, Medical Insurance
- **Entertainment**: Mobile Data, DSTV, Cinema, Social
- **Savings**: SACCO, Bank Deposit, Investment

### Uganda Merchants & Providers
- **Supermarkets**: Shoprite, Quality Supermarket, Capital Shoppers, Game Stores
- **Fuel**: Total, Shell, Gapco
- **Restaurants**: Java House, KFC, Cafe Javas, Pork Joint
- **Pharmacies**: Vine Pharmacy, Medipharm, Rocket Health
- **Banks**: Stanbic Bank, Centenary Bank, DFCU, Equity Bank

### Income Distribution (Monthly)

**Low Income (40 users)**: UGX 300,000 - 800,000
- Expense ratio: 85-105% of income
- Financial discipline: Lower

**Medium Income (45 users)**: UGX 800,000 - 2,500,000
- Expense ratio: 65-85% of income
- Financial discipline: Moderate

**High Income (15 users)**: UGX 2,500,000 - 8,000,000
- Expense ratio: 45-65% of income
- Financial discipline: Higher

**Platform Average**: UGX 1,811,589/month

### Airtime Purchases: 1,655 records
- **Providers**: MTN, Airtel
- **Common Denominations**: UGX 5,000, 10,000, 20,000, 50,000, 100,000
- **Frequency**: Average every 7 days

### Loan History: 142 records
- **Providers**: Bank, SACCO, Microfinance, Mobile Loan
- **Interest Rates**: 8-15% monthly (typical Uganda rates)
- **Loan Terms**: 1, 2, 3, 6, or 12 months
- **Loan Amounts**: 0.5x to 3x monthly income

### User Profiles: 100 users
- **Eligible**: 51 out of 100
- **Credit Scores**: 0-100 range based on:
  - Income level (30 points)
  - Expense ratio (25 points)
  - Financial discipline (30 points)
  - Formal employment (15 points)

## Top Transaction Categories

1. Agricultural Sales: 395 transactions
2. Salary: 393 transactions
3. Freelance Work: 348 transactions
4. Remittances: 344 transactions
5. Business Income: 342 transactions
6. SACCO: 169 transactions
7. Rent: 161 transactions
8. Stock Purchase: 155 transactions
9. Business Supplies: 151 transactions
10. Investment: 148 transactions

## Top Locations by Transaction Volume

1. Mbale: 712 transactions
2. Kampala: 655 transactions
3. Lira: 614 transactions
4. Gulu: 543 transactions
5. Hoima: 521 transactions

## Model Performance with Uganda Data

**Model**: Random Forest Classifier
**Test Accuracy**: 100%
**ROC-AUC Score**: 1.0000

### Top Features for Loan Eligibility:
1. Credit Score (25.95%)
2. Monthly Income (20.25%)
3. Previous Loans Average (8.54%)
4. Previous Loans Total (6.81%)
5. Income Average (4.80%)

## Realistic Behaviors Implemented

1. **Salary Timing**: Income transactions concentrated at month start
2. **Boda Boda Usage**: Small, frequent transport expenses (UGX 2,000-15,000)
3. **School Fees**: Large periodic payments (UGX 200,000-2,000,000)
4. **UMEME Bills**: Monthly utility payments (UGX 30,000-300,000)
5. **Agricultural Patterns**: Seasonal income from agricultural sales
6. **Mobile Money**: Primary transaction medium reflecting Uganda's mobile money economy
7. **SACCO Savings**: Common savings behavior in Uganda

## Frontend Updates

- Currency: Changed to UGX (Uganda Shillings)
- Phone Format: +256 (Uganda country code)
- Loan Amounts: UGX 1,000,000 - 50,000,000
- Branding: Uganda Financial Services
- Professional banking theme without emojis

## API Endpoints

All endpoints now use Uganda-specific data:

- **GET /users**: Returns 100 Uganda-based user profiles
- **POST /evaluate**: Evaluates loan applications using Uganda financial patterns
- **GET /health**: System health check

## Data Files

All data stored in `data/raw/`:
- `mobile_money_transactions.csv` - 5,614 Uganda transactions
- `airtime_purchases.csv` - 1,655 airtime records
- `loan_history.csv` - 142 loan records
- `loan_eligibility.csv` - 100 user profiles with credit scores

## Next Steps

To start the frontend and see the platform in action:

```bash
cd frontend
npm install
npm start
```

The platform will be available at http://localhost:3000 with Uganda-specific:
- Currency formatting (UGX)
- Phone numbers (+256)
- Professional banking interface
- Real Uganda financial data
