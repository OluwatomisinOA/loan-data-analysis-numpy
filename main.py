import numpy as np
import pandas as pd

def main():
    print("--- Loan Data Analysis with NumPy ---")

    # 1. Load the Data
    # We use pandas to load the CSV, then convert to numpy arrays immediately
    try:
        df = pd.read_csv('loan.csv')
        print("Data loaded successfully.\n")
    except FileNotFoundError:
        print("Error: loan.csv not found. Please ensure it is in the same directory.")
        return

    # Extract columns into NumPy arrays
    loan_amount = df['LoanAmount'].values
    applicant_income = df['ApplicantIncome'].values
    coapplicant_income = df['CoapplicantIncome'].values
    loan_status = df['Loan_Status'].values

    # --- Project 1: Data Cleaning & Imputation ---
    print("--- Project 1: Cleaning Missing Values ---")
    
    # Identify missing values
    nan_mask = np.isnan(loan_amount)
    num_missing = np.sum(nan_mask)
    print(f"Original missing LoanAmount values: {num_missing}")

    # Calculate median ignoring NaNs
    median_loan = np.nanmedian(loan_amount)
    print(f"Median Loan Amount: {median_loan}")

    # Fill missing values with the median
    filled_loan_amount = np.where(nan_mask, median_loan, loan_amount)
    print(f"Missing values after cleaning: {np.sum(np.isnan(filled_loan_amount))}\n")


    # --- Project 2: Income Analysis (Vectorization) ---
    print("--- Project 2: Income Analysis ---")

    # Calculate Total Income using vector addition
    total_income = applicant_income + coapplicant_income
    
    # Calculate basic stats
    mean_income = np.mean(total_income)
    std_income = np.std(total_income)
    
    print(f"Average Total Income: ${mean_income:.2f}")
    print(f"Income Standard Deviation: ${std_income:.2f}\n")


    # --- Project 3: Advanced Filtering ---
    print("--- Project 3: Approved vs Rejected Analysis ---")

    # Create boolean masks
    approved_mask = (loan_status == 'Y')
    rejected_mask = (loan_status == 'N')

    # Filter data based on masks
    approved_incomes = total_income[approved_mask]
    rejected_incomes = total_income[rejected_mask]

    print(f"Number of Approved Loans: {len(approved_incomes)}")
    print(f"Number of Rejected Loans: {len(rejected_incomes)}")
    
    print(f"Mean Income (Approved): ${np.mean(approved_incomes):.2f}")
    print(f"Mean Income (Rejected): ${np.mean(rejected_incomes):.2f}")

if __name__ == "__main__":
    main()
