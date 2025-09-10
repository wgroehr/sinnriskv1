import pandas as pd

def calculate_portfolio_metrics(filename):
    """Calculate portfolio metrics from a CSV file."""
    
    # Load your CSV file
    df = pd.read_csv(filename, skiprows=3)
    
    # Calculate DV01 for each bond 
    df['DV01'] = df ['Px Close'] * df['OAD'] * 0.0001
    # Portfolio-level calculations
    total_portfolio_value = df['Px Close'].sum()
    total_portfolio_dv01 = df['DV01'].sum()
    portfolio_duration = total_portfolio_dv01 / (total_portfolio_value * 0.0001)
    avg_convexity = df['Convex to Mty'].mean()
    
    # Print results
    print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
    print(f"Total Portfolio DV01: ${total_portfolio_dv01:.2f}")
    print(f"Portfolio Duration: {portfolio_duration:.2f} ")
    print(f"Number of Securities: {len(df)}")
    print(f"Average Convexity: {avg_convexity: .2f}")
    
    return df

# Usage
if __name__ == "__main__":
    filename = input("Enter the CSV filename (with extension): ")
    calculate_portfolio_metrics(filename)

