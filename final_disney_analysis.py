# final_disney_analysis_v3.py # Renaming implies versioning, let's stick to the original name or a new one if preferred
# final_disney_analysis.py

# ---- Imports ----
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, BaseLoader, Template # Use Template for string loading
from datetime import datetime, timedelta
from pytz import timezone
import numbers
import json
import os
import traceback
import webbrowser # For opening HTML

# ---- Configuration ----
TICKER = 'DIS'
INDEX_TICKER = '^GSPC'
FETCH_PERIOD = '2y'
COMPETITORS = ['NFLX', 'AMZN', 'CMCSA', 'WBD', 'PARA']
REPORT_FILENAME = 'disney_stock_analysis.html'
PLOT_DIR = 'plots'

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")

# ---- Part 1: Data Fetching ----
def fetch_stock_data(ticker, period='2y'):
    print(f"Fetching {period} data for {ticker}...")
    try:
        stock_data = yf.Ticker(ticker)
        # Use default progress for Ticker methods
        hist_data = yf.download(ticker, period=period) # Keep default progress here
        if hist_data.empty: print(f"Warning: No historical data returned for {ticker}."); return None, None
        if not isinstance(hist_data.index, pd.DatetimeIndex): hist_data.index = pd.to_datetime(hist_data.index)
        stock_data.ticker_symbol = ticker; return stock_data, hist_data
    except Exception as e: print(f"Error fetching stock data for {ticker}: {e}"); return None, None

def fetch_index_data(ticker, period='2y'):
    print(f"Fetching {period} data for {ticker}...")
    try:
        # Use default progress for Ticker methods
        data = yf.download(ticker, period=period) # Keep default progress here
        if data.empty: print(f"Warning: No historical data returned for {ticker}."); return None
        if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index)
        data.ticker_symbol = ticker; return data
    except Exception as e: print(f"Error fetching index data for {ticker}: {e}"); return None

# ---- Part 2: Technical Indicator Calculations ----
def calculate_technical_indicators(hist_data):
    if hist_data is None: return None
    print("Calculating technical indicators...")
    try:
        hist_data = hist_data.copy()
        if len(hist_data) >= 50: hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
        else: hist_data['MA50'] = np.nan
        if len(hist_data) >= 200: hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
        else: hist_data['MA200'] = np.nan
        delta = hist_data['Close'].diff(); gain = delta.where(delta > 0, 0).fillna(0); loss = -delta.where(delta < 0, 0).fillna(0)
        if len(hist_data) >= 14:
            avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
            # Handle division by zero for avg_loss
            rs = avg_gain.div(avg_loss.replace(0, np.nan)).fillna(np.inf) # Avoid division by zero warning and handle it
            hist_data['RSI'] = 100 - (100 / (1 + rs)); hist_data['RSI'] = hist_data['RSI'].replace([np.inf, -np.inf], 100) # Replace inf from fillna with 100
        else: hist_data['RSI'] = np.nan
        hist_data['EMA12'] = hist_data['Close'].ewm(span=12, adjust=False).mean(); hist_data['EMA26'] = hist_data['Close'].ewm(span=26, adjust=False).mean()
        hist_data['MACD'] = hist_data['EMA12'] - hist_data['EMA26']; hist_data['Signal'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
        hist_data['MACD_Hist'] = hist_data['MACD'] - hist_data['Signal']
        if len(hist_data) >= 20:
            hist_data['20d_SMA'] = hist_data['Close'].rolling(window=20).mean(); hist_data['20d_STD'] = hist_data['Close'].rolling(window=20).std().fillna(0)
            hist_data['Upper_Band'] = hist_data['20d_SMA'] + (hist_data['20d_STD'] * 2); hist_data['Lower_Band'] = hist_data['20d_SMA'] - (hist_data['20d_STD'] * 2)
        else: hist_data['20d_SMA'], hist_data['20d_STD'], hist_data['Upper_Band'], hist_data['Lower_Band'] = np.nan, np.nan, np.nan, np.nan
        return hist_data
    except Exception as e: print(f"Error calculating technical indicators: {e}"); traceback.print_exc(); return None

# ---- Helper for Part 3 ----
def safe_get_scalar(value):
    if isinstance(value, pd.Series):
        if not value.empty: val = value.iloc[0]; return val if pd.notna(val) else None
        else: return None
    elif isinstance(value, numbers.Number): return value if pd.notna(value) else None
    elif pd.isna(value): return None
    else:
        try: val = float(value); return val if pd.notna(val) else None
        except (ValueError, TypeError): return None

# ---- Part 3: Key Metrics and Returns Calculation ----
def calculate_key_metrics(hist_data):
    if hist_data is None: return None
    print("Calculating key metrics...")
    required_cols = ['Close', 'Volume'];
    if not all(col in hist_data.columns for col in required_cols): return None
    try:
        daily_returns = hist_data['Close'].pct_change(); annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
        if len(hist_data) >= 252: high_val_raw = hist_data['Close'].tail(252).max(); low_val_raw = hist_data['Close'].tail(252).min()
        else: high_val_raw = hist_data['Close'].max(); low_val_raw = hist_data['Close'].min()
        fifty_two_week_high = safe_get_scalar(high_val_raw); fifty_two_week_low = safe_get_scalar(low_val_raw)
        current_price_raw = hist_data['Close'].iloc[-1]; current_price = safe_get_scalar(current_price_raw)
        if current_price is None: return None
        current_to_high = ((current_price / fifty_two_week_high) - 1) * 100 if (fifty_two_week_high is not None and fifty_two_week_high != 0) else 0.0
        current_to_low = ((current_price / fifty_two_week_low) - 1) * 100 if (fifty_two_week_low is not None and fifty_two_week_low != 0) else 0.0
        ma50 = safe_get_scalar(hist_data['MA50'].iloc[-1]) if 'MA50' in hist_data.columns and not hist_data['MA50'].isnull().all() else None
        ma200 = safe_get_scalar(hist_data['MA200'].iloc[-1]) if 'MA200' in hist_data.columns and not hist_data['MA200'].isnull().all() else None
        rsi = safe_get_scalar(hist_data['RSI'].iloc[-1]) if 'RSI' in hist_data.columns and not hist_data['RSI'].isnull().all() else None
        if len(hist_data) >= 63: avg_vol_3m_raw = hist_data['Volume'].tail(63).mean()
        else: avg_vol_3m_raw = hist_data['Volume'].mean()
        if len(hist_data) >= 10: avg_vol_2w_raw = hist_data['Volume'].tail(10).mean()
        else: avg_vol_2w_raw = hist_data['Volume'].mean()
        avg_volume_3m = safe_get_scalar(avg_vol_3m_raw); avg_volume_2w = safe_get_scalar(avg_vol_2w_raw)
        volume_trend = (avg_volume_2w / avg_volume_3m - 1) * 100 if (avg_volume_3m is not None and avg_volume_3m != 0 and avg_volume_2w is not None) else 0.0
        return {'annualized_volatility': annualized_volatility, 'fifty_two_week_high': fifty_two_week_high, 'fifty_two_week_low': fifty_two_week_low,
                'current_to_high': current_to_high, 'current_to_low': current_to_low, 'current_price': current_price, 'ma50': ma50, 'ma200': ma200,
                'rsi': rsi, 'avg_volume_2w': avg_volume_2w, 'avg_volume_3m': avg_volume_3m, 'volume_trend': volume_trend}
    except Exception as e: print(f"Error calculating key metrics: {e}"); traceback.print_exc(); return None

def calculate_returns(data, period_label="Stock"):
    if data is None or len(data) < 2: return None, None, "N/A"
    # print(f"Calculating returns for {period_label}...") # Reduced verbosity
    try:
        current_raw = data['Close'].iloc[-1]; current = safe_get_scalar(current_raw)
        if current is None: return None, None, "Error"
        latest_date = data.index[-1]; first_date = data.index[0]
        start_of_year_dt = datetime(latest_date.year, 1, 1)
        if data.index.tz is not None: start_of_year_aware = timezone(str(data.index.tz)).localize(start_of_year_dt); year_data = data[data.index >= start_of_year_aware]
        else: year_data = data[data.index >= start_of_year_dt]
        if not year_data.empty:
            year_start_price_raw = year_data.iloc[0]['Close']; year_start_price = safe_get_scalar(year_start_price_raw)
            ytd_return = ((current / year_start_price) - 1) * 100 if (year_start_price is not None and year_start_price != 0) else None
        else: ytd_return = None
        period_start_price_raw = data.iloc[0]['Close']; period_start_price = safe_get_scalar(period_start_price_raw)
        period_return = ((current / period_start_price) - 1) * 100 if (period_start_price is not None and period_start_price != 0) else None
        time_delta_days = (latest_date - first_date).days
        if time_delta_days > 700: period_return_label = "2-Year Return"
        elif time_delta_days > 330: period_return_label = "1-Year Return"
        else: period_return_label = f"Period ({max(0, time_delta_days)} Days) Return"
        return ytd_return, period_return, period_return_label
    except Exception as e: print(f"Error calculating returns for {period_label}: {e}"); traceback.print_exc(); return None, None, "Error"

# ---- Part 4: Beta Calculation ----
def calculate_beta(stock_hist_data, index_hist_data):
    # Check if inputs are valid DataFrames
    if not isinstance(stock_hist_data, pd.DataFrame) or stock_hist_data.empty:
        print("Warning: Invalid or empty stock data passed to calculate_beta.")
        return None
    if not isinstance(index_hist_data, pd.DataFrame) or index_hist_data.empty:
        print("Warning: Invalid or empty index data passed to calculate_beta.")
        return None

    print("Calculating beta...")
    try:
        stock_ticker = getattr(stock_hist_data, 'ticker_symbol', TICKER)
        index_ticker = getattr(index_hist_data, 'ticker_symbol', INDEX_TICKER)

        stock_close_col = ('Close', stock_ticker)
        index_close_col = ('Close', index_ticker)

        # --- Use simpler Series selection and combination ---
        # Select the 'Close' column data as a pandas Series
        if stock_close_col in stock_hist_data.columns:
            stock_close_series = stock_hist_data[stock_close_col]
        elif 'Close' in stock_hist_data.columns: # Fallback to simple 'Close'
             stock_close_series = stock_hist_data['Close']
        else:
            print(f"Error: Cannot find 'Close' column ({stock_close_col} or 'Close') in stock data.")
            return None

        if index_close_col in index_hist_data.columns:
            index_close_series = index_hist_data[index_close_col]
        elif 'Close' in index_hist_data.columns: # Fallback to simple 'Close'
             index_close_series = index_hist_data['Close']
        else:
            print(f"Error: Cannot find 'Close' column ({index_close_col} or 'Close') in index data.")
            return None

        # Combine the two Series into a DataFrame, aligning by index automatically
        combined_data = pd.DataFrame({'stock_close': stock_close_series, 'index_close': index_close_series}).dropna()

        if len(combined_data) < 2:
            print("Warning: Not enough overlapping/valid data points between stock and index to calculate beta.")
            return None

        # Calculate percentage changes on the new DataFrame columns
        stock_returns = combined_data['stock_close'].pct_change()
        index_returns = combined_data['index_close'].pct_change()

        # Combine returns and drop initial NaN row from pct_change
        combined_returns = pd.DataFrame({'stock': stock_returns, 'index': index_returns}).dropna()

        if len(combined_returns) <= 1:
             print("Warning: Not enough valid return data points after dropping NaNs for beta.")
             return None

        # Use pandas cov/var
        covariance = combined_returns['stock'].cov(combined_returns['index'])
        index_variance = combined_returns['index'].var()

        if pd.isna(index_variance) or index_variance == 0 or pd.isna(covariance):
            print(f"Warning: Invalid variance ({index_variance}) or covariance ({covariance}) for beta calculation.")
            return None

        beta = covariance / index_variance
        print(f"Beta calculated successfully: {beta:.3f}")
        return beta

    except KeyError as ke:
        print(f"Error calculating Beta (KeyError): Problem accessing column '{ke}'. Check data.")
        # traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error calculating Beta (General Exception): {e}")
        traceback.print_exc()
        return None

# ---- Part 5: Competitor Data Fetching and Processing ----
def fetch_competitor_data(competitors_list):
    print("Fetching competitor data...")
    competitor_data_map = {}
    for comp_ticker_symbol in competitors_list:
        print(f"  Processing {comp_ticker_symbol}...")
        data_for_comp = {'Name': comp_ticker_symbol}
        try:
            comp_yf_ticker = yf.Ticker(comp_ticker_symbol)
            comp_info = comp_yf_ticker.info
            if not isinstance(comp_info, dict): comp_info = {}

            comp_hist = yf.download(comp_ticker_symbol, period='1y', progress=False)
            if comp_hist.empty: ytd_return, yearly_return = None, None
            else: ytd_return, yearly_return, _ = calculate_returns(comp_hist, period_label=comp_ticker_symbol)
            data_for_comp['YTD Return'] = ytd_return
            data_for_comp['Yearly Return'] = yearly_return

            comp_name = comp_info.get('shortName', comp_ticker_symbol)
            # <<< Clean name WITHOUT rstrip >>>
            cleaned_name = comp_name.replace(' (The)', '').replace('(The)', '').strip()
            data_for_comp['Name'] = cleaned_name

            data_for_comp['Market Cap'] = comp_info.get('marketCap')
            data_for_comp['PE Ratio'] = comp_info.get('trailingPE')
            data_for_comp['EV/EBITDA'] = comp_info.get('enterpriseToEbitda')
            data_for_comp['Price to Sales'] = comp_info.get('priceToSalesTrailing12Months')

            revenue = None; profit_margin_decimal = None
            revenue = comp_info.get('totalRevenue')
            profit_margin_decimal = comp_info.get('profitMargins')

            if revenue is None or profit_margin_decimal is None:
                try:
                    annual_financials = comp_yf_ticker.financials
                    if not annual_financials.empty:
                        if revenue is None and 'Total Revenue' in annual_financials.index and not annual_financials.loc['Total Revenue'].empty:
                            revenue = annual_financials.loc['Total Revenue'].iloc[0]
                        if profit_margin_decimal is None:
                             net_income_key = next((k for k in ['Net Income Continuous Operations', 'Net Income', 'Net Income From Continuing Ops', 'Net Income Common Stockholders'] if k in annual_financials.index), None)
                             if net_income_key and not annual_financials.loc[net_income_key].empty:
                                 net_income = annual_financials.loc[net_income_key].iloc[0]
                                 if revenue is not None and net_income is not None and revenue != 0: profit_margin_decimal = net_income / revenue
                except Exception as fin_err: print(f"    Warning: Could not fetch/process annual financials for {comp_ticker_symbol}: {fin_err}")

            data_for_comp['Revenue'] = revenue
            data_for_comp['Profit Margin'] = profit_margin_decimal # Store decimal

            competitor_data_map[comp_ticker_symbol] = data_for_comp
        except Exception as e:
            print(f"    Error processing data for {comp_ticker_symbol}: {e}")
            competitor_data_map[comp_ticker_symbol] = data_for_comp
            competitor_data_map[comp_ticker_symbol]['Error'] = str(e)
    return competitor_data_map

# ---- Part 6: Quarterly Metrics Calculation ----
def get_quarterly_metrics(ticker_obj, num_quarters=8):
    if ticker_obj is None: return []
    print("Fetching quarterly metrics...")
    quarterly_data_list = []
    try:
        # Fetch quarterly financials and income statement
        q_financials = ticker_obj.quarterly_financials
        q_income = ticker_obj.quarterly_income_stmt

        if q_financials.empty: print("  Warning: No quarterly financial data found."); return []

        # Get the available quarter dates (columns)
        quarter_dates = q_financials.columns.tolist()

        for quarter_date in quarter_dates[:num_quarters]:
            quarter_num = (quarter_date.month - 1) // 3 + 1
            quarter_label = f"{quarter_date.year}-Q{quarter_num}"

            revenue_q, net_income_q, op_income_q = None, None, None

            # Safely access data using .get(key) for index and .get(col) for column
            if 'Total Revenue' in q_financials.index:
                 revenue_q = safe_get_scalar(q_financials.loc['Total Revenue'].get(quarter_date))

            # Net income might be under different names, check common ones
            net_income_key = next((k for k in ['Net Income', 'Net Income From Continuing Ops', 'Net Income Common Stockholders'] if k in q_financials.index), None)
            if net_income_key:
                 net_income_q = safe_get_scalar(q_financials.loc[net_income_key].get(quarter_date))

            # Check quarterly income statement for operating income
            if not q_income.empty and 'Operating Income' in q_income.index:
                 op_income_q = safe_get_scalar(q_income.loc['Operating Income'].get(quarter_date))

            quarterly_data_list.append({
                'Quarter': quarter_label,
                'Date': quarter_date, # Keep original date for sorting
                'Revenue': revenue_q,
                'Net Income': net_income_q,
                'Operating Income': op_income_q
            })

        # Sort by date ascending before returning
        quarterly_data_list.sort(key=lambda x: x['Date'])
        return quarterly_data_list

    except Exception as e: print(f"Error getting quarterly data: {e}"); traceback.print_exc(); return []


# ---- Part 7: Fundamentals Data Fetching ----
def get_fundamentals(ticker_obj):
    if ticker_obj is None: return {}
    print("Fetching fundamentals...")
    fundamentals = {}
    try:
        info = ticker_obj.info;
        if not info or not isinstance(info, dict):
            print(f"Warning: No .info data for {getattr(ticker_obj, 'ticker', 'N/A')}."); return {}

        # --- Revised safe_get_info ---
        def safe_get_info(key, default=None):
            value = info.get(key, default)
            # Check for various forms of None/NaN
            if value is None or value == 'None' or pd.isna(value): return default
            # Check for infinity strings
            if isinstance(value, str) and value.lower() == 'infinity': return default
            # Allow standard types through
            if isinstance(value, (numbers.Number, str, bool)):
                # Ensure large integers (like Market Cap, Revenue) are treated as numbers
                if isinstance(value, int) and value > 1e15: # Heuristic for very large ints from yfinance sometimes being odd types
                     try: return float(value)
                     except: return default # Fallback if conversion fails
                return value
            # If it's not a recognized scalar type, return default
            else:
                # Optional: print a warning for unexpected types
                # print(f"Warning: Unexpected type for key '{key}': {type(value)}")
                return default

        # --- Assign Fundamentals using safe_get_info where appropriate ---
        fundamentals['Market Cap'] = safe_get_info('marketCap')
        fundamentals['Enterprise Value'] = safe_get_info('enterpriseValue')
        fundamentals['EV/EBITDA'] = safe_get_info('enterpriseToEbitda')
        fundamentals['Price to Book'] = safe_get_info('priceToBook')
        fundamentals['Price to Sales'] = safe_get_info('priceToSalesTrailing12Months')
        fundamentals['Dividend Yield'] = safe_get_info('dividendYield') # Raw decimal
        fundamentals['Payout Ratio'] = safe_get_info('payoutRatio') # Raw decimal

        roe = safe_get_info('returnOnEquity'); fundamentals['ROE'] = roe * 100 if isinstance(roe, numbers.Number) else roe
        roa = safe_get_info('returnOnAssets'); fundamentals['ROA'] = roa * 100 if isinstance(roa, numbers.Number) else roa
        # Get profit margin decimal, store as percentage
        profit_margin_decimal = safe_get_info('profitMargins')
        fundamentals['Profit Margin'] = profit_margin_decimal * 100 if isinstance(profit_margin_decimal, numbers.Number) else profit_margin_decimal
        # Get operating margin decimal, store as percentage
        op_margin_decimal = safe_get_info('operatingMargins')
        fundamentals['Operating Margin'] = op_margin_decimal * 100 if isinstance(op_margin_decimal, numbers.Number) else op_margin_decimal

        fundamentals['Quick Ratio'] = safe_get_info('quickRatio')
        fundamentals['Current Ratio'] = safe_get_info('currentRatio')
        fundamentals['Debt to Equity'] = safe_get_info('debtToEquity')
        fundamentals['Forward PE'] = safe_get_info('forwardPE')
        fundamentals['Trailing PE'] = safe_get_info('trailingPE')
        fundamentals['PEG Ratio'] = safe_get_info('pegRatio')
        fundamentals['Beta'] = safe_get_info('beta')
        fundamentals['52 Week High'] = safe_get_info('fiftyTwoWeekHigh')
        fundamentals['52 Week Low'] = safe_get_info('fiftyTwoWeekLow')
        fundamentals['Avg Volume 10 Day'] = safe_get_info('averageVolume10days')
        fundamentals['Avg Volume 3 Month'] = safe_get_info('averageDailyVolume3Month')
        fundamentals['targetMeanPrice'] = safe_get_info('targetMeanPrice')
        fundamentals['shortName'] = safe_get_info('shortName', ticker_obj.ticker)

        # --- Explicitly add totalRevenue ---
        # Use safe_get_info to handle potential None/NaN/etc. for revenue
        fundamentals['totalRevenue'] = safe_get_info('totalRevenue')

       
        return fundamentals
    except Exception as e:
        print(f"Error getting fundamentals data: {e}")
        traceback.print_exc()
        return {}

# ---- Part 8: Investment Recommendation ----
# <<< REVISED AGAIN for Strong Company + Good Price Opportunity >>>
def generate_recommendation(key_metrics, fundamentals, beta_val, returns_data, ticker_symbol='STOCK'):
    print("Generating recommendation (Strong Co. + Price Opportunity Focus)...")
    score = 0
    reasons_pro_company = [] # Reasons company looks fundamentally strong
    reasons_con_company = [] # Reasons company looks fundamentally weak/risky
    reasons_pro_price = []   # Reasons current price looks like an opportunity
    reasons_con_price = []   # Reasons current price looks unfavorable/expensive

    if not key_metrics or not fundamentals:
        # Return empty lists for all reasons
        return "ERROR", "Insufficient data for analysis.", [], [], [], [], 0

    # --- Extract Data ---
    current_price = key_metrics.get('current_price')
    ma200 = key_metrics.get('ma200')
    rsi = key_metrics.get('rsi')
    forward_pe = fundamentals.get('Forward PE')
    peg_ratio = fundamentals.get('PEG Ratio')
    debt_to_equity = fundamentals.get('Debt to Equity')
    profit_margin = fundamentals.get('Profit Margin')
    roe = fundamentals.get('ROE')
    ytd_return, period_return, period_label = returns_data if returns_data else (None, None, "N/A")
    long_term_return = period_return if period_label != "N/A" else None

    # --- Scoring Logic ---
    try:
        # === Company Strength Factors ===
        # 1. Profitability (Profit Margin & ROE) - WEIGHT: HIGH
        if profit_margin is not None and isinstance(profit_margin, numbers.Number):
             if profit_margin > 10: score += 2; reasons_pro_company.append(f"Strong Profit Margin ({profit_margin:.1f}% > 10%).")
             elif profit_margin > 3: score += 1; reasons_pro_company.append(f"Positive Profit Margin ({profit_margin:.1f}% > 3%).")
             else: score -= 2; reasons_con_company.append(f"Low/Negative Profit Margin ({profit_margin:.1f}% <= 3%).")
        if roe is not None and isinstance(roe, numbers.Number):
            if roe > 15: score += 2; reasons_pro_company.append(f"Healthy Return on Equity ({roe:.1f}% > 15%).")
            elif roe > 5: score += 1; reasons_pro_company.append(f"Positive Return on Equity ({roe:.1f}% > 5%).")
            else: score -= 1; reasons_con_company.append(f"Low Return on Equity ({roe:.1f}% <= 5%).")

        # 2. Financial Health (Debt/Equity) - WEIGHT: MEDIUM/HIGH
        if debt_to_equity is not None and isinstance(debt_to_equity, numbers.Number):
             debt_ratio = debt_to_equity / 100.0 if debt_to_equity > 5 else debt_to_equity
             if debt_ratio < 0.7: score += 1; reasons_pro_company.append(f"Manageable Debt-to-Equity ({debt_ratio:.2f} < 0.7).")
             elif debt_ratio > 1.5: score -= 2; reasons_con_company.append(f"High Debt-to-Equity ({debt_ratio:.2f} > 1.5).")
             else: reasons_con_company.append(f"Moderate Debt-to-Equity ({debt_ratio:.2f}).")

        # === Price Opportunity Factors ===
        # 3. Valuation (Forward PE & PEG) - WEIGHT: MEDIUM/HIGH
        valuation_score = 0
        if forward_pe is not None and isinstance(forward_pe, numbers.Number) and forward_pe > 0:
            if forward_pe < 22: # Adjusted threshold for "good value" PE
                 valuation_score += 1; reasons_pro_price.append(f"Forward P/E ({forward_pe:.1f}) seems reasonable (< 22).")
            elif forward_pe < 30:
                 reasons_con_price.append(f"Forward P/E ({forward_pe:.1f}) is somewhat high (22-30).")
            else:
                 valuation_score -= 1; reasons_con_price.append(f"Forward P/E ({forward_pe:.1f}) looks expensive (> 30).")
        if peg_ratio is not None and isinstance(peg_ratio, numbers.Number) and peg_ratio > 0:
             if peg_ratio < 1.2: valuation_score += 1; reasons_pro_price.append(f"PEG Ratio ({peg_ratio:.2f}) suggests good value for growth (< 1.2).")
             elif peg_ratio < 1.8: reasons_pro_price.append(f"PEG Ratio ({peg_ratio:.2f}) seems reasonable for growth (< 1.8).")
             else: valuation_score -= 1; reasons_con_price.append(f"PEG Ratio ({peg_ratio:.2f}) might indicate overvaluation for growth (> 1.8).")
        score += valuation_score

        # 4. Technical Price Level / Timing (MA200, RSI, Recent Return) - WEIGHT: LOW (More for context/timing)
        if current_price is not None and ma200 is not None:
            # Penalize if significantly below MA200 (broken trend), otherwise neutral or slight positive if near
            if current_price < ma200 * 0.95:
                 score -= 1 # Caution signal
                 reasons_con_price.append(f"Price is significantly below 200-Day MA (${ma200:.2f}), long-term trend weak.")
            # No points purely for being above MA200 anymore

        if rsi is not None:
            if rsi < 40: # Potential pullback buy opportunity
                score += 1 # Small timing bonus
                reasons_pro_price.append(f"RSI ({rsi:.1f} < 40) indicates potential pullback / oversold condition.")
            elif rsi > 70: # Avoid buying at short-term peak
                score -= 1 # Small timing penalty
                reasons_con_price.append(f"RSI ({rsi:.1f} > 70) indicates potentially overbought short-term.")

        if long_term_return is not None:
            # Note large moves for context, but don't penalize drops
            if long_term_return < -20: # Significant recent drop occurred
                 reasons_pro_price.append(f"Context: Price has seen a significant recent drop ({period_label} {long_term_return:.1f}%), potentially offering lower entry IF fundamentals strong.")
            elif long_term_return > 50: # Significant recent run-up
                 reasons_con_price.append(f"Context: Price has seen a very strong recent run-up ({period_label} {long_term_return:.1f}%), consider if value remains.")

        # Beta (Volatility context) - Still just context
        if beta_val is not None:
             context = f"Note: Beta ({beta_val:.2f}) indicates {'higher' if beta_val > 1.1 else 'lower' if beta_val < 0.9 else 'market-like'} volatility."
             if beta_val > 1.1 or beta_val < 0.9 : reasons_con_company.append(context) # Add as a risk/factor
             else: reasons_pro_company.append(context)

    except Exception as e:
        print(f"Error during recommendation scoring: {e}"); traceback.print_exc()
        return "ERROR", "Analysis error occurred.", [], [], [], [], 0

    # --- Determine Final Recommendation ---
    # Threshold requires solid fundamentals AND not overly negative price signals
    buy_threshold = 4 # Adjusted threshold based on revised scoring

    if score >= buy_threshold:
        recommendation = "BUY"
        summary = f"Analysis suggests {ticker_symbol} presents a potential <strong>{recommendation}</strong> opportunity for long-term investment (Score: {score} >= {buy_threshold}). The company shows positive fundamental indicators, and the current valuation/price level may offer a reasonable entry point."
    else:
        recommendation = "DON'T BUY"
        summary = f"Analysis suggests <strong>{recommendation}</strong> {ticker_symbol} for a long-term position at this time (Score: {score} < {buy_threshold}). Either fundamental concerns or unfavorable valuation/price action exist. Consider waiting for improvement or a better entry point."

    # Return all reason lists
    return recommendation, summary, reasons_pro_company, reasons_con_company, reasons_pro_price, reasons_con_price, score


# ---- Part 9: Plotting and HTML Report Generation ----

# --- Static Content ---
segment_revenue = { # Example data - UPDATE MANUALLY from latest earnings
    'Segments': ['Entertainment (DTC & Linear)', 'Sports (ESPN)', 'Experiences (Parks & Products)'], # Simplified names
    'Revenue (Billions)': [40.6, 17.1, 34.8], # Example Q4 FY23 Annualized - NEEDS UPDATE
    'Growth': [-8.0, -0.5, 16.0] # Example YoY Growth - NEEDS UPDATE
}
streaming_breakdown = { # Example data - UPDATE MANUALLY from latest earnings (e.g., Q2 FY24)
    'Service': ['Disney+ Core', 'Hulu (SVOD Only)', 'ESPN+'],
    'Subscribers (millions)': [117.6, 50.2, 24.8], # Example Q2 FY24 - NEEDS UPDATE
    'ARPU': [7.28, 11.84, 6.30], # Example Q2 FY24 - NEEDS UPDATE
    'YoY Growth (%)': [6.0, 1.0, -2.0] # Example YoY Growth - NEEDS UPDATE
}
# <<< CHANGE: Added explicit \n for line breaks >>>
investment_thesis = """Disney possesses a unique and powerful portfolio of intellectual property and brands, fueling its diverse revenue streams across media, theme parks, and consumer products.\nThe strategic shift towards direct-to-consumer (DTC) streaming, while costly initially, positions Disney for long-term growth in the evolving media landscape.\nSynergies between segments (e.g., movie characters driving park attendance and merchandise sales) create a significant competitive advantage."""
risk_factors = """- Intense competition in the streaming market requires significant ongoing investment in content and technology, pressuring margins.\n- Economic downturns can negatively impact discretionary spending, affecting theme park attendance and advertising revenue.\n- The decline of linear television continues to challenge the traditional cable network business model.\n- Integration risks and achieving profitability targets for the DTC segment remain key execution challenges."""
catalysts = """- Achieving sustained profitability in the combined streaming business (Disney+, Hulu, ESPN+).\n- Continued strong performance and pricing power in the Parks, Experiences, and Products segment.\n- Successful execution of content strategy, including major film releases and new streaming series leveraging core IP.\n- Strategic partnerships or further optimization of the media portfolio (e.g., ESPN developments)."""
conclusion = """Disney is navigating a complex transformation, balancing investment in its high-growth streaming future with the challenges in its traditional media businesses. The strength of its IP and Parks segment provides a solid foundation. Success hinges on achieving DTC profitability, managing costs effectively, and adapting to evolving consumer preferences. The stock offers potential long-term value but carries execution risks related to the streaming transition and macroeconomic factors."""
# <<< CHANGE: Using fundamentals dividend yield if available, fallback to manual >>>
# manual_dividend_yield_display = 1.02 # Kept as fallback

# --- Plotting Functions ---
# <<< CHANGE: Standardized figsize for single-column layout >>>
TARGET_FIGSIZE = (7, 3) # Define a consistent size (Width, Height)

def plot_stock_performance(hist_data, ticker, filename=f"{PLOT_DIR}/stock_performance.png"):
    if hist_data is None or not all(c in hist_data.columns for c in ['Close', 'MA50', 'MA200']): print(f"Skipping plot: {filename} - Missing data"); return
    print(f"Generating plot: {filename}")
    try:
        # <<< Use TARGET_FIGSIZE >>>
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=TARGET_FIGSIZE)
        plt.plot(hist_data.index, hist_data['Close'], label='Close Price', linewidth=2, color='royalblue')
        plt.plot(hist_data.index, hist_data['MA50'], label='50-Day MA', linewidth=1.5, alpha=0.8, color='orange')
        plt.plot(hist_data.index, hist_data['MA200'], label='200-Day MA', linewidth=1.5, alpha=0.8, color='firebrick')
        plt.title(f'{ticker} Stock Price Performance ({FETCH_PERIOD} Chart)', fontsize=14, fontweight='bold'); plt.ylabel('Price (USD)', fontsize=12); plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=10); plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.7); plt.minorticks_on(); plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')); plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)); plt.gcf().autofmt_xdate()
        plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"Error generating {filename}: {e}"); traceback.print_exc()

def plot_rsi(hist_data, ticker, current_rsi, filename=f"{PLOT_DIR}/rsi_chart.png"):
    if hist_data is None or 'RSI' not in hist_data.columns: print(f"Skipping plot: {filename} - Missing data"); return
    print(f"Generating plot: {filename}")
    try:
        # <<< Use TARGET_FIGSIZE >>>
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=TARGET_FIGSIZE)
        plt.plot(hist_data.index, hist_data['RSI'], color='purple', linewidth=1.5, label='RSI (14)')
        plt.axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)'); plt.axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold (30)'); plt.axhline(50, color='grey', linestyle=':', alpha=0.5)
        plt.fill_between(hist_data.index, 70, hist_data['RSI'], where=(hist_data['RSI'] >= 70), interpolate=True, color='red', alpha=0.2)
        plt.fill_between(hist_data.index, hist_data['RSI'], 30, where=(hist_data['RSI'] <= 30), interpolate=True, color='green', alpha=0.2)
        plt.title(f'{ticker} Relative Strength Index (RSI)', fontsize=14, fontweight='bold'); plt.ylabel('RSI Value', fontsize=12); plt.ylim(0, 100); plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=10); plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')); plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)); plt.gcf().autofmt_xdate()
        if current_rsi is not None: plt.text(hist_data.index[-1], current_rsi, f' {current_rsi:.1f}', verticalalignment='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"Error generating {filename}: {e}"); traceback.print_exc()

def plot_macd(hist_data, ticker, filename=f"{PLOT_DIR}/macd_chart.png"):
    if hist_data is None or not all(c in hist_data.columns for c in ['MACD', 'Signal', 'MACD_Hist']): print(f"Skipping plot: {filename} - Missing data"); return
    print(f"Generating plot: {filename}")
    try:
        # <<< Use TARGET_FIGSIZE >>>
        plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=TARGET_FIGSIZE)
        ax.plot(hist_data.index, hist_data['MACD'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(hist_data.index, hist_data['Signal'], label='Signal Line', color='red', linewidth=1.5, alpha=0.8)
        # Ensure MACD_Hist exists and is not all NaN before trying to iterate
        if 'MACD_Hist' in hist_data and not hist_data['MACD_Hist'].isnull().all():
            colors = ['g' if v >= 0 else 'r' for v in hist_data['MACD_Hist'].fillna(0)] # fillna for color check
            ax.bar(hist_data.index, hist_data['MACD_Hist'], label='MACD Histogram', color=colors, width=1.0, alpha=0.6)
        else:
             ax.bar(hist_data.index, [0]*len(hist_data.index), label='MACD Histogram (No Data)', color='grey', width=1.0, alpha=0.1) # Placeholder if no data

        ax.set_title(f'{ticker} MACD Indicator', fontsize=14, fontweight='bold'); ax.set_ylabel('MACD Value', fontsize=12); ax.set_xlabel('Date', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.7); ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); fig.autofmt_xdate()
        plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close(fig) # Close the figure object
    except Exception as e: print(f"Error generating {filename}: {e}"); traceback.print_exc()

def plot_bollinger(hist_data, ticker, filename=f"{PLOT_DIR}/bollinger_bands.png"):
    if hist_data is None or not all(c in hist_data.columns for c in ['Close', '20d_SMA', 'Upper_Band', 'Lower_Band']): print(f"Skipping plot: {filename} - Missing data"); return
    print(f"Generating plot: {filename}")
    try:
        # <<< Use TARGET_FIGSIZE >>>
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=TARGET_FIGSIZE)
        plt.plot(hist_data.index, hist_data['Close'], label='Close Price', color='navy', linewidth=1.5)
        plt.plot(hist_data.index, hist_data['20d_SMA'], label='20-Day SMA', color='darkorange', linewidth=1, linestyle='--')
        plt.plot(hist_data.index, hist_data['Upper_Band'], label='Upper Band', color='maroon', linewidth=1, linestyle=':')
        plt.plot(hist_data.index, hist_data['Lower_Band'], label='Lower Band', color='forestgreen', linewidth=1, linestyle=':')
        plt.fill_between(hist_data.index, hist_data['Upper_Band'], hist_data['Lower_Band'], color='grey', alpha=0.15)
        plt.title(f'{ticker} Bollinger Bands (20, 2)', fontsize=14, fontweight='bold'); plt.ylabel('Price (USD)', fontsize=12); plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=10); plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')); plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)); plt.gcf().autofmt_xdate()
        plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"Error generating {filename}: {e}"); traceback.print_exc()

def plot_quarterly_performance(quarterly_data, ticker, filename=f"{PLOT_DIR}/quarterly_performance.png"):
    if not quarterly_data: print(f"Skipping plot: {filename} - Missing data"); return
    print(f"Generating plot: {filename}")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        quarters = [q['Quarter'] for q in quarterly_data];
        revenues = [(q.get('Revenue') or 0) for q in quarterly_data]
        op_incomes = [(q.get('Operating Income') or 0) for q in quarterly_data]
        net_incomes = [(q.get('Net Income') or 0) for q in quarterly_data]

        revenues_b = [r / 1e9 for r in revenues]
        op_incomes_b = [oi / 1e9 for oi in op_incomes]
        net_incomes_b = [ni / 1e9 for ni in net_incomes]

        # <<< Use TARGET_FIGSIZE >>>
        fig, ax1 = plt.subplots(figsize=TARGET_FIGSIZE)

        bar_width = 0.4
        index = np.arange(len(quarters))

        revenue_bars = ax1.bar(index, revenues_b, bar_width, color='skyblue', alpha=0.8, label='Revenue') # Centered bars for single bar type
        ax1.set_xlabel('Quarter', fontsize=10)
        ax1.set_ylabel('Billions USD (Revenue)', fontsize=10, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=9)
        ax1.set_xticks(index)
        ax1.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
        ax1.margins(x=0.02)

        for bar in revenue_bars:
            height = bar.get_height()
            # Check if height is valid before formatting
            if pd.notna(height):
                ax1.annotate(f'{height:.1f}B', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        ax2 = ax1.twinx()
        # Check if income data is valid before plotting
        if any(pd.notna(oi) for oi in op_incomes_b):
             op_income_line = ax2.plot(index, op_incomes_b, color='darkgreen', marker='o', linestyle='-', linewidth=2, markersize=5, label='Operating Income')
        if any(pd.notna(ni) for ni in net_incomes_b):
             net_income_line = ax2.plot(index, net_incomes_b, color='purple', marker='s', linestyle='--', linewidth=1.5, markersize=4, label='Net Income')

        ax2.set_ylabel('Billions USD (Income)', fontsize=10, color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen', labelsize=9)

        # Calculate ylim based on valid income data only
        valid_incomes_b = [inc for inc in op_incomes_b + net_incomes_b if pd.notna(inc)]
        if valid_incomes_b: # Only set ylim if there's valid data
            min_income = min(valid_incomes_b)
            max_income = max(valid_incomes_b)
            ax2.set_ylim(min(0, min_income - abs(min_income)*0.1 if min_income != 0 else -0.1), # Adjust if min is 0
                         max(1, max_income * 1.1)) # Ensure ylim is reasonable

        # Combine legends only if lines were plotted
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles or handles2: # Check if any legend items exist
            ax2.legend(handles + handles2, labels + labels2, loc='upper left', fontsize=9)

        plt.title(f'{ticker} Quarterly Financial Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth='0.5', color='grey', alpha=0.7)
        ax2.grid(True, which='major', axis='y', linestyle=':', linewidth='0.5', color='grey', alpha=0.5)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure object
    except Exception as e: print(f"Error generating {filename}: {e}"); traceback.print_exc()


# Competitor comparison plot
def plot_competitor_comparison(df_competitors, ticker, filename=f"{PLOT_DIR}/competitor_comparison.png"):
    # Input validation... (keep as is)
    if not isinstance(df_competitors, pd.DataFrame) or df_competitors.empty:
        print(f"Skipping plot: {filename} - Missing or invalid competitor DataFrame."); # Placeholder generation...
        try: fig, ax = plt.subplots(figsize=(10, 2)); ax.text(0.5, 0.5, 'Comp Data Missing', ha='center', va='center'); ax.axis('off'); plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close(fig)
        except Exception: pass
        return

    print(f"Generating plot: {filename}")
    plot_metrics_order = ['Market Cap (Billions)', 'PE Ratio', 'EV/EBITDA','Profit Margin (%)', 'Price to Sales', 'Yearly Return (%)']
    valid_metrics = [m for m in plot_metrics_order if m in df_competitors.columns]
    if not valid_metrics: print(f"Warning: No valid metrics found for plotting {filename}."); return

    num_metrics_found = len(valid_metrics); num_cols = 2; num_rows = (num_metrics_found + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3.7 * num_rows), squeeze=False)
    axes = axes.flatten(); plt.style.use('seaborn-v0_8-darkgrid')

    try:
        plot_axis_index = 0
        for metric in plot_metrics_order:
             if metric in valid_metrics:
                if plot_axis_index >= len(axes): continue
                ax = axes[plot_axis_index]
                numeric_series = df_competitors[metric].dropna()

                if numeric_series.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, color='grey')
                    ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
                else:
                    plot_data = numeric_series.sort_values(ascending=False)
                    colors = ['royalblue' if idx == ticker else 'skyblue' for idx in plot_data.index]
                    bars = ax.barh(plot_data.index, plot_data.values, color=colors)
                    ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
                    ax.tick_params(axis='y', labelsize=9); ax.tick_params(axis='x', labelsize=8)
                    ax.grid(True, axis='x', linestyle='--', linewidth='0.5', alpha=0.5)

                    # --- Add labels to bars (Revised Alignment for Large Positives) ---
                    for bar in bars:
                        width = bar.get_width()
                        if pd.notna(width):
                            padding_factor = 0.03 # Slightly reduced padding factor
                            xlim = ax.get_xlim()
                            axis_range = xlim[1] - xlim[0] if xlim[1] > xlim[0] else 1.0
                            padding = axis_range * padding_factor

                            # Determine format string first
                            if 'Billions' in metric: format_str = '{:,.1f}B'
                            elif '%' in metric: format_str = '{:,.1f}%'
                            else: format_str = '{:,.1f}'
                            label_text = format_str.format(width)

                            # <<< Logic to adjust position/alignment >>>
                            if width >= 0:
                                # Default position: outside right
                                label_x_pos = width + padding
                                ha = 'left'
                                # Check if label goes beyond axis limit
                                # Estimate text width (very rough, depends on font/dpi)
                                estimated_text_width = len(label_text) * (axis_range / 100) # Rough estimate
                                if label_x_pos + estimated_text_width > xlim[1]:
                                    # If it overflows, place inside left
                                    label_x_pos = width - padding
                                    ha = 'right'
                                    # Optional: Change text color if placed inside bar
                                    # text_color = 'white' if bar color is dark else 'black'
                            else: # Negative bar
                                label_x_pos = width - padding # Position left of bar end
                                ha = 'left'   # Align LEFT edge for negative bars

                            # Add the text label
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2., label_text,
                                    va='center', ha=ha, fontsize=8) # Use calculated alignment

                    ax.invert_yaxis()
                plot_axis_index += 1

        for j in range(plot_axis_index, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.suptitle(f'{ticker} vs Competitors - Key Metrics', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    except Exception as e: print(f"Error generating competitor plot {filename}: {e}"); traceback.print_exc()
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

# <<< End of plot_competitor_comparison function >>>

# <<< FUNCTION plot_main_charts_grid - REVISED X-AXIS, VOLUME LABEL, RSI FILL >>>
def plot_main_charts_grid(hist_data, quarterly_data, key_metrics, ticker, filename=f"{PLOT_DIR}/main_charts_grid.png"):
    """Generates a 3x2 grid of key charts: Price/MA, Volume, RSI, MACD, Bollinger, Quarterly."""
    print(f"Generating plot grid: {filename}")

    if not isinstance(hist_data, pd.DataFrame) or hist_data.empty:
        print(f"Skipping plot grid {filename}: Missing or invalid historical data.")
        try: fig, ax = plt.subplots(figsize=(10, 2)); ax.text(0.5, 0.5, 'Hist. Data Missing', ha='center', va='center', fontsize=12, color='grey'); ax.axis('off'); plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close(fig)
        except Exception: pass
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), sharex=False)
    axes = axes.flatten()

    title_fontsize = 11
    label_fontsize = 9
    tick_fontsize = 8
    legend_fontsize = 8

    # --- Plot 1: Price / Moving Averages (axes[0]) ---
    ax = axes[0]
    price_col = ('Close', ticker) if ('Close', ticker) in hist_data.columns else 'Close'
    ma50_col = ('MA50', '') if ('MA50', '') in hist_data.columns else 'MA50'
    ma200_col = ('MA200', '') if ('MA200', '') in hist_data.columns else 'MA200'
    plot1_has_data = False
    if price_col in hist_data.columns and ma50_col in hist_data.columns and ma200_col in hist_data.columns:
        ax.plot(hist_data.index, hist_data[price_col], label='Close', linewidth=1.5, color='royalblue')
        ax.plot(hist_data.index, hist_data[ma50_col], label='MA50', linewidth=1, alpha=0.8, color='orange', linestyle='--')
        ax.plot(hist_data.index, hist_data[ma200_col], label='MA200', linewidth=1, alpha=0.8, color='firebrick', linestyle='--')
        ax.set_title(f'{ticker} Price & MAs', fontsize=title_fontsize, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=label_fontsize); ax.legend(fontsize=legend_fontsize); ax.tick_params(axis='y', labelsize=tick_fontsize)
        plot1_has_data = True
    else:
        ax.text(0.5, 0.5, 'Price/MA Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title(f'{ticker} Price & MAs', fontsize=title_fontsize, fontweight='bold')

    # --- Plot 2: RSI (axes[1]) ---
    ax = axes[1]
    current_rsi = key_metrics.get('rsi') if key_metrics else None
    rsi_col = ('RSI', '') if ('RSI', '') in hist_data.columns else 'RSI'
    plot2_has_data = False
    if rsi_col in hist_data.columns and not hist_data[rsi_col].isnull().all():
        rsi_series = hist_data[rsi_col] # Get the series once
        ax.plot(rsi_series.index, rsi_series, color='purple', linewidth=1.2, label='RSI (14)')
        ax.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Overbought (70)'); ax.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.6, label='Oversold (30)'); ax.axhline(50, color='grey', linestyle=':', linewidth=0.7, alpha=0.5)
        # <<< Add back fill_between >>>
        ax.fill_between(rsi_series.index, 70, rsi_series, where=(rsi_series >= 70), interpolate=True, color='red', alpha=0.15)
        ax.fill_between(rsi_series.index, rsi_series, 30, where=(rsi_series <= 30), interpolate=True, color='green', alpha=0.15)
        ax.set_ylim(0, 100); ax.set_title('Relative Strength Index (RSI)', fontsize=title_fontsize, fontweight='bold'); ax.set_ylabel('RSI Value', fontsize=label_fontsize); ax.legend(fontsize=legend_fontsize)
        if current_rsi is not None: ax.text(hist_data.index[-1], current_rsi, f' {current_rsi:.1f}', va='center', fontsize=tick_fontsize, fontweight='bold', color='purple')
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        plot2_has_data = True
    else:
        ax.text(0.5, 0.5, 'RSI Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title('Relative Strength Index (RSI)', fontsize=title_fontsize, fontweight='bold')

    # --- Plot 3: Volume (axes[2]) ---
    ax = axes[2]
    volume_col = ('Volume', ticker) if ('Volume', ticker) in hist_data.columns else 'Volume'
    plot3_has_data = False
    if volume_col in hist_data.columns and not hist_data[volume_col].isnull().all():
        volume_data = hist_data[volume_col]; avg_vol_3m = volume_data.rolling(window=63).mean()
        ax.bar(volume_data.index, volume_data, color='lightsteelblue', alpha=0.7, width=1.0, label='Volume')
        # <<< Updated Avg Vol Label >>>
        ax.plot(avg_vol_3m.index, avg_vol_3m, color='darkslateblue', linestyle='--', linewidth=1, label='3-Month Avg')
        ax.set_title('Trading Volume (with 3M Avg)', fontsize=title_fontsize, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=label_fontsize); ax.legend(fontsize=legend_fontsize); ax.tick_params(axis='y', labelsize=tick_fontsize)
        formatter = plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M' if x >= 1e6 else (f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))
        ax.yaxis.set_major_formatter(formatter)
        plot3_has_data = True
    else:
        ax.text(0.5, 0.5, 'Volume Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title('Trading Volume (with 3M Avg)', fontsize=title_fontsize, fontweight='bold')

    # --- Plot 4: MACD (axes[3]) ---
    # ... (MACD code remains the same - Assuming it's okay) ...
    ax = axes[3]
    macd_col = ('MACD', '') if ('MACD', '') in hist_data.columns else 'MACD'
    signal_col = ('Signal', '') if ('Signal', '') in hist_data.columns else 'Signal'
    hist_col = ('MACD_Hist', '') if ('MACD_Hist', '') in hist_data.columns else 'MACD_Hist'
    plot4_has_data = False
    if macd_col in hist_data.columns and signal_col in hist_data.columns and hist_col in hist_data.columns \
       and not hist_data[macd_col].isnull().all():
        ax.plot(hist_data.index, hist_data[macd_col], label='MACD', color='blue', linewidth=1.2); ax.plot(hist_data.index, hist_data[signal_col], label='Signal', color='red', linewidth=1.2, linestyle='--')
        macd_hist_vals = hist_data[hist_col].fillna(0); colors = ['g' if v >= 0 else 'r' for v in macd_hist_vals]
        ax.bar(hist_data.index, macd_hist_vals, label='Histogram', color=colors, width=1.0, alpha=0.5); ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.set_title('MACD Indicator', fontsize=title_fontsize, fontweight='bold'); ax.set_ylabel('MACD Value', fontsize=label_fontsize); ax.legend(fontsize=legend_fontsize); ax.tick_params(axis='y', labelsize=tick_fontsize)
        plot4_has_data = True
    else:
        ax.text(0.5, 0.5, 'MACD Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title('MACD Indicator', fontsize=title_fontsize, fontweight='bold')

    # --- Plot 5: Bollinger Bands (axes[4]) ---
    # ... (Bollinger code remains the same - Assuming it's okay, except for X-axis formatting below) ...
    ax = axes[4]
    close_col = ('Close', ticker) if ('Close', ticker) in hist_data.columns else 'Close'
    sma_col = ('20d_SMA', '') if ('20d_SMA', '') in hist_data.columns else '20d_SMA'
    upper_col = ('Upper_Band', '') if ('Upper_Band', '') in hist_data.columns else 'Upper_Band'
    lower_col = ('Lower_Band', '') if ('Lower_Band', '') in hist_data.columns else 'Lower_Band'
    plot5_has_data = False
    if close_col in hist_data.columns and sma_col in hist_data.columns and upper_col in hist_data.columns and lower_col in hist_data.columns and not hist_data[sma_col].isnull().all():
        ax.plot(hist_data.index, hist_data[close_col], label='Close', color='navy', linewidth=1.5, alpha=0.8); ax.plot(hist_data.index, hist_data[sma_col], label='SMA(20)', color='darkorange', linewidth=1, linestyle='--')
        ax.plot(hist_data.index, hist_data[upper_col], label='Upper', color='maroon', linewidth=1, linestyle=':'); ax.plot(hist_data.index, hist_data[lower_col], label='Lower', color='forestgreen', linewidth=1, linestyle=':')
        ax.fill_between(hist_data.index, hist_data[upper_col], hist_data[lower_col], color='grey', alpha=0.1)
        ax.set_title('Bollinger Bands (20, 2)', fontsize=title_fontsize, fontweight='bold'); ax.set_ylabel('Price (USD)', fontsize=label_fontsize); ax.legend(fontsize=legend_fontsize); ax.tick_params(axis='y', labelsize=tick_fontsize)
        plot5_has_data = True
    else:
        ax.text(0.5, 0.5, 'Bollinger Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title('Bollinger Bands (20, 2)', fontsize=title_fontsize, fontweight='bold')


    # --- Plot 6: Quarterly Performance (axes[5]) ---
    # ... (Quarterly code remains the same - Assuming it's okay) ...
    ax = axes[5]
    plot6_has_data = False
    if quarterly_data and isinstance(quarterly_data, list):
        try:
            quarters = [q['Quarter'] for q in quarterly_data]; revenues = [(q.get('Revenue') or 0) for q in quarterly_data]; op_incomes = [(q.get('Operating Income') or 0) for q in quarterly_data]
            revenues_b = [r / 1e9 for r in revenues]; op_incomes_b = [oi / 1e9 for oi in op_incomes]; x_index = np.arange(len(quarters))
            bar_width = 0.6; ax.bar(x_index, revenues_b, bar_width, color='skyblue', alpha=0.8, label='Revenue')
            ax.set_ylabel('Billions USD (Revenue)', fontsize=label_fontsize, color='steelblue'); ax.tick_params(axis='y', labelcolor='steelblue', labelsize=tick_fontsize)
            ax.set_xticks(x_index); ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=tick_fontsize)
            ax2 = ax.twinx(); valid_op_incomes = [oi for oi in op_incomes_b if pd.notna(oi)]
            if valid_op_incomes:
                ax2.plot(x_index, op_incomes_b, color='darkgreen', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='Op. Income')
                min_oi, max_oi = min(valid_op_incomes), max(valid_op_incomes); ax2.set_ylim(min(0, min_oi - abs(min_oi)*0.1 if min_oi != 0 else -0.1), max(1, max_oi * 1.1))
            else: ax2.set_ylim(0, 1)
            ax2.set_ylabel('Billions USD (Income)', fontsize=label_fontsize, color='darkgreen'); ax2.tick_params(axis='y', labelcolor='darkgreen', labelsize=tick_fontsize)
            lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2: ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_fontsize)
            ax.set_title('Quarterly Revenue & Op. Income', fontsize=title_fontsize, fontweight='bold'); ax.grid(True, which='major', axis='y', linestyle='--', linewidth='0.5', color='lightgrey', alpha=0.7)
            plot6_has_data = True
        except Exception as q_err:
             print(f"  Error processing quarterly data for plot: {q_err}")
             ax.text(0.5, 0.5, 'Quarterly Data Error', ha='center', va='center', transform=ax.transAxes, color='red'); ax.set_title('Quarterly Revenue & Op. Income', fontsize=title_fontsize, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Quarterly Data Missing', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_title('Quarterly Revenue & Op. Income', fontsize=title_fontsize, fontweight='bold')


    # --- Final Touches for the Grid ---
    plot_flags = [plot1_has_data, plot2_has_data, plot3_has_data, plot4_has_data, plot5_has_data, plot6_has_data] # Include plot 6 flag
    for i, ax_ in enumerate(axes):
        if not plot_flags[i]: # Skip empty placeholders or errored plots
             ax_.tick_params(axis='x', labelbottom=False, bottom=False); ax_.tick_params(axis='y', labelleft=False, left=False)
             continue
        # Add grids
        ax_.grid(True, which='major', axis='x', linestyle='--', linewidth='0.5', color='lightgrey', alpha=0.5)
        ax_.grid(True, which='major', axis='y', linestyle='--', linewidth='0.5', color='lightgrey', alpha=0.7)
        ax_.tick_params(axis='x', labelsize=tick_fontsize) # Base x tick size

        # Apply specific X-axis formatting ONLY to time series plots (0-4)
        is_time_series = i in [0, 1, 2, 3, 4]
        if is_time_series:
            try:
                # <<< Use MonthLocator and YearLocator for Bollinger/other date axes >>>
                if i == 4: # Specific formatting for Bollinger axis (bottom row)
                    ax_.xaxis.set_major_locator(mdates.YearLocator()) # Tick mark per year
                    ax_.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Show Year
                    ax_.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) # Minor ticks every 3 months
                    ax_.xaxis.set_minor_formatter(mdates.DateFormatter('%b')) # Show Month Abbrev
                    ax_.tick_params(axis='x', which='minor', labelsize=tick_fontsize-1) # Smaller minor labels
                else: # Formatting for upper row date axes (less detail needed)
                     locator = mdates.AutoDateLocator(minticks=3, maxticks=6) # Fewer ticks maybe
                     formatter = mdates.ConciseDateFormatter(locator)
                     ax_.xaxis.set_major_locator(locator)
                     ax_.xaxis.set_major_formatter(formatter)

            except Exception as fmt_err:
                print(f"  Warning: Error setting date formatter/locator for axis {i}: {fmt_err}")

    # Apply automatic date formatting for the whole figure AFTER individual formatting
    try:
        fig.autofmt_xdate(rotation=30, ha='right')
    except Exception as afd_err:
        print(f"  Warning: fig.autofmt_xdate() failed: {afd_err}")

    # Ensure x-axis labels are only shown on bottom row plots if they have data
    for i in [0, 1, 2, 3]: # Axes NOT on bottom row
         if plot_flags[i]: axes[i].tick_params(axis='x', labelbottom=False) # Hide labels if plot has data
         else: axes[i].tick_params(axis='x', labelbottom=False, bottom=False) # Also hide ticks if no data

    # Ensure bottom row labels/ticks only show if data exists
    if not plot5_has_data: axes[4].tick_params(axis='x', labelbottom=False, bottom=False)
    if not plot6_has_data: axes[5].tick_params(axis='x', labelbottom=False, bottom=False)


    fig.suptitle(f'{ticker} Key Chart Analysis', fontsize=title_fontsize+2, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved plot grid: {filename}")
    except Exception as save_err:
        print(f"ERROR saving plot grid {filename}: {save_err}")
    finally:
        plt.close(fig)


# --- HTML Template ---
# <<< REPLACE ENTIRE EXISTING html_template_string WITH THIS >>>
html_template_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <!-- <<< Corrected Title Order >>> -->
    <title>{{ company_name }} ({{ ticker }}) Stock Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 20px auto; padding: 0 20px; background-color: #f8f9fa; }
        .header { text-align: center; padding-bottom: 15px; border-bottom: 3px solid #1a73e8; margin-bottom: 25px; }
        /* <<< Corrected Title Order >>> */
        h1 { color: #1a73e8; margin-bottom: 5px; font-weight: 600;}
        h2 { color: #1967d2; border-bottom: 2px solid #e8f0fe; padding-bottom: 8px; margin-top: 35px; font-weight: 600; }
        h3 { color: #333; margin-top: 25px; font-weight: 600; border-bottom: 1px dashed #ccc; padding-bottom: 3px; }
        h4 { color: #1967d2; margin-top: 20px; font-weight: 600; }
        .recommendation { padding: 15px 20px; border-radius: 8px; font-weight: bold; font-size: 1.2em; text-align: center; margin: 25px 0; border: 1px solid; }
        .buy { background-color: #e6f4ea; border-color: #b7e1c7; color: #1e4620; }
        .holdneutral { background-color: #fff8e1; border-color: #ffecb3; color: #6d4c0d; }
        .avoidsell { background-color: #fdecea; border-color: #f5c6cb; color: #5a1620; }
        .error { background-color: #fff3cd; border-color: #ffeeba; color: #856404; }
        .summary { font-size: 1.05em; margin-bottom: 20px; padding: 15px; background-color: #e8f0fe; border-left: 5px solid #1a73e8; border-radius: 5px; }
        .qualitative-rec { background-color: #e6f4ea; border-left-color: #188038; margin-top: 20px; padding: 15px; border-radius: 5px; } /* Style for PDF's recommendation */
        .flex-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
        .metrics-box { flex: 1 1 280px; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #ddd;}
        .metrics-title { font-size: 1.15em; font-weight: bold; margin-bottom: 15px; color: #1967d2; border-bottom: 1px solid #eee; padding-bottom: 8px;}
        .metric-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px dotted #eee; font-size: 0.95em;}
        .metric-row:last-child { border-bottom: none; }
        .metric-row span:first-child { color: #5f6368; }
        .metric-row span:last-child { font-weight: 600; }
        .positive { color: #188038; font-weight: 600; }
        .negative { color: #d93025; font-weight: 600; }
        .neutral { color: #5f6368; }
        .chart-container { margin: 20px 0; text-align: center; background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #ddd;}
        .chart-img { max-width: 100%; height: auto; margin-bottom: 10px; border: 1px solid #eee; border-radius: 4px;}
        .chart-caption { font-size: 0.9em; color: #5f6368; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 0.9em; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; border: 1px solid #ddd;}
        th, td { padding: 10px 14px; border-bottom: 1px solid #ddd; text-align: left; }
        th { background-color: #e8f0fe; color: #1967d2; font-weight: 600; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #f1f3f4; }
        .rationale-list, .factors-list, .takeaways-list, .swot-list { list-style: none; padding-left: 5px; }
        .rationale-list li { background-color: #fff; margin-bottom: 8px; padding: 10px 15px; border-radius: 5px; border-left: 4px solid #1a73e8; font-size: 0.95em; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
        .takeaways-list li, .swot-list li, .trends-list li { list-style: disc; margin-left: 25px; margin-bottom: 10px; } /* Use standard bullets */
        .pros-cons-table { margin-top: 20px; }
        .pros-cons-table td { vertical-align: top; padding: 15px; }
        .pros-cons-table td:first-child { border-right: 1px solid #ddd; }
        .pros-cons-table h4 { margin-top: 0; border-bottom: none; }
        .education-section { background-color: #f0f4f8; padding: 20px; margin-top: 30px; border-radius: 8px; border: 1px solid #d1d9e0; }
        .education-section h3 { border-bottom: 1px solid #a6b7c8; color: #0d47a1; }
        .education-section p { margin-bottom: 15px; font-size: 0.95em;}
        .education-section .metric-explainer, .education-section .chart-explainer { margin-bottom: 25px; padding-left: 15px; border-left: 3px solid #a6b7c8; }
        .education-section strong { color: #1967d2; }
        .disclaimer { font-size: 0.85em; font-style: italic; margin-top: 40px; padding: 15px; background-color: #f1f1f1; border-radius: 5px; text-align: center; color: #5f6368; border-top: 1px solid #ddd;}
        .note {font-size: 0.85em; color: #5f6368; font-style: italic;}
    </style>
</head>
<body>
    <div class="header">
        <!-- <<< Corrected Title Order >>> -->
        <h1>{{ company_name }} ({{ ticker }}) Stock Analysis Report</h1>
        <p>Generated on: {{ current_date }}</p>
    </div>

    <!-- <<< NEW: Introduction Section from PDF >>> -->
<h2>Introduction</h2>
<p>The Walt Disney Company, a global leader in the media and entertainment industry, has demonstrated remarkable resilience and adaptability in the face of unprecedented challenges, particularly during the COVID-19 pandemic. This report provides a comprehensive analysis of Disney's financial performance, competitive positioning, and the broader industry trends from approximately 2020 to the present. By examining Disney's financial health, competitive landscape, and emerging industry trends alongside current quantitative stock data, this report aims to offer a holistic view to aid in making a long-term investment decision.</p>

    <!-- <<< NEW: Executive Summary Section from PDF >>> -->
    <h2>Executive Summary & Key Takeaways</h2>
    <p class="note">Insights below are based on an external analysis report covering 2020-2024.</p>
    <ul class="takeaways-list">
        <li><strong>Strong Financial Recovery:</strong> Disney has rebounded strongly from the COVID-19 pandemic, with revenue growth and profitability returning to pre-pandemic levels by 2024. Free cash flow and operating cash flow have also shown significant improvement, indicating strong financial health.</li>
        <li><strong>Diversified Business Model:</strong> Disney's diversified revenue streams—spanning media networks, theme parks, studio entertainment, and streaming services (Disney+)—provide resilience against economic downturns and sector-specific challenges.</li>
        <li><strong>Competitive Positioning:</strong> Disney remains a market leader in the entertainment industry, with a strong brand, extensive intellectual property (IP) library, and global reach. However, it faces intense competition from companies like Netflix, Comcast, and Warner Bros. Discovery.</li>
        <li><strong>Streaming Growth:</strong> Disney+ has seen rapid growth, but profitability in the streaming segment remains a challenge. The company is focusing on subscriber retention and cost management to improve margins.</li>
        <li><strong>Emerging Opportunities:</strong> Disney is well-positioned to capitalize on emerging markets (e.g., China, India), technological innovations (e.g., Al, AR/VR), and sustainability initiatives.</li>
        <li><strong>Risks and Challenges:</strong> Disney faces risks from economic downturns, piracy, technological disruption, and high operating costs, particularly in its theme parks and content production.</li>
    </ul>

    <!-- <<< NEW: Pros and Cons Section from PDF >>> -->
    <h2>Pros and Cons of Investing in Disney Stock</h2>
    <table class="pros-cons-table">
        <tr>
            <td width="50%">
                <h4>Pros</h4>
                <ul>
                    <li><strong>Strong Brand and IP:</strong> Disney owns iconic franchises like Marvel, Star Wars, and Pixar, which drive revenue across multiple segments.</li>
                    <li><strong>Diversified Revenue Streams:</strong> Multiple business segments reduce reliance on any single revenue source.</li>
                    <li><strong>Streaming Growth:</strong> Disney+ has rapidly grown its subscriber base, though profitability remains a work in progress.</li>
                    <li><strong>Global Reach:</strong> Disney's theme parks and content distribution span the globe, providing exposure to international markets.</li>
                    <li><strong>Strong Cash Flow:</strong> Improved cash flow generation in 2024 indicates financial stability and the ability to reinvest in growth.</li>
                </ul>
            </td>
            <td width="50%">
                <h4>Cons</h4>
                <ul>
                    <li><strong>High Operating Costs:</strong> Running theme parks and producing high-budget content are capital-intensive.</li>
                    <li><strong>Dependence on Blockbusters:</strong> The film segment's success is heavily tied to the performance of a few blockbuster movies each year.</li>
                    <li><strong>Streaming Competition:</strong> Intense competition from Netflix, Amazon Prime Video, and others could pressure margins.</li>
                    <li><strong>Economic Sensitivity:</strong> Disney's revenue is vulnerable to economic downturns, as consumers may cut back on discretionary spending.</li>
                    <li><strong>Piracy and IP Theft:</strong> Piracy remains a significant threat to Disney's revenue streams.</li>
                </ul>
            </td>
        </tr>
    </table>

    <!-- Quantitative Recommendation Section -->
    <h2>Quantitative Model Recommendation</h2>
    <div class="recommendation {{ rec_class }}">
        RECOMMENDATION: {{ recommendation }} (Score: {{ rec_score }})
    </div>
    <div class="summary">
        <p>{{ rec_summary }}</p>
        {% if key_metrics.current_price %}
            <p style="font-weight: 600;">Current Price: ${{ "%.2f"|format(key_metrics.current_price) }}
            {% if fundamentals.get('targetMeanPrice') %} | Avg. Analyst Target: ${{ "%.2f"|format(fundamentals.get('targetMeanPrice')) }} {% endif %}
             {% if key_metrics.fifty_two_week_low is not none and key_metrics.fifty_two_week_high is not none %}
                | 52-Wk Range: ${{ "%.2f"|format(key_metrics.fifty_two_week_low) }} - ${{ "%.2f"|format(key_metrics.fifty_two_week_high) }}
             {% endif %}
            </p>
        {% endif %}
    </div>
    <h4>Model Rationale (Score Breakdown)</h4>
    <!-- Final Rationale Section - Only show categories with reasons -->
    <div class="flex-container"> {# Use flex-box for side-by-side reasons #}
        <div class="metrics-box" style="margin-top: 10px;"> {# Left Box: Supporting Reasons #}
            <h4 style="color: #188038; margin-top:0; border-bottom: none;">Supporting Reasons (Pro-BUY)</h4>

            {# Only show Company Strength if list is not empty #}
            {% if reasons_pro_company %}
                <strong>Company Strength:</strong>
                <ul class="rationale-list" style="margin-top: 5px;">
                    {% for reason in reasons_pro_company %}<li>{{ reason }}</li>{% endfor %}
                </ul>
                <br> {# Add break only if section was shown #}
            {% endif %}

            {# Only show Price/Valuation Opportunity if list is not empty #}
            {% if reasons_pro_price %}
                <strong>Price/Valuation Opportunity:</strong>
                <ul class="rationale-list" style="margin-top: 5px;">
                    {% for reason in reasons_pro_price %}<li>{{ reason }}</li>{% endfor %}
                </ul>
            {% endif %}

            {# Add fallback if BOTH pro lists are empty #}
            {% if not reasons_pro_company and not reasons_pro_price %}
                 <p style="color: grey; font-style: italic;">No specific supporting reasons identified by the model for a 'BUY' signal at this time.</p>
            {% endif %}
        </div>

        <div class="metrics-box" style="margin-top: 10px;"> {# Right Box: Cautionary Reasons #}
             <h4 style="color: #d93025; margin-top:0; border-bottom: none;">Cautionary Reasons (Con-BUY)</h4>

             {# Only show Company Concerns if list is not empty #}
             {% if reasons_con_company %}
                 <strong>Company Concerns:</strong>
                <ul class="rationale-list" style="margin-top: 5px;">
                    {% for reason in reasons_con_company %}<li>{{ reason }}</li>{% endfor %}
                </ul>
                <br> {# Add break only if section was shown #}
            {% endif %}

             {# Only show Price/Valuation Concerns if list is not empty #}
            {% if reasons_con_price %}
                <strong>Price/Valuation Concerns:</strong>
                <ul class="rationale-list" style="margin-top: 5px;">
                    {% for reason in reasons_con_price %}<li>{{ reason }}</li>{% endfor %}
                </ul>
            {% endif %}

             {# Add fallback if BOTH con lists are empty #}
            {% if not reasons_con_company and not reasons_con_price %}
                 <p style="color: grey; font-style: italic;">No specific cautionary reasons identified by the model suggesting 'DON'T BUY' at this time.</p>
            {% endif %}
        </div>
    </div>
    <!-- End of Final Rationale Section -->

    <!-- Qualitative / Long-Term Recommendation Section (Keep as is) -->
    <h2>Qualitative Investment Outlook & Strategy Considerations</h2>
    <p class="note">The following outlook considers the quantitative data alongside broader business factors, drawing insights from an external analysis.</p>
    <div class="qualitative-rec">
        <strong>Overall Outlook: Buy (Cautious Optimism for Long-Term)</strong>
        <p>Disney's strong financial recovery, diversified business model, and leadership in the entertainment industry make it an attractive long-term investment. The company's ability to generate strong cash flow, coupled with its extensive IP library and global reach, positions it well for future growth.</p>
        <p>However, investors should be mindful of the following:</p>
        <ol>
            <li><strong>Streaming Profitability:</strong> While Disney+ has seen rapid growth, achieving sustained profitability in the streaming segment remains a key challenge. Monitor progress in reducing losses and improving margins.</li>
            <li><strong>Economic Sensitivity:</strong> Revenue remains susceptible to economic downturns (theme parks, advertising). Consider the broader economic environment.</li>
            <li><strong>Competition:</strong> Intense competition in streaming and entertainment could pressure market share and profitability.</li>
            <li><strong>Valuation:</strong> Analyze the current stock price and P/E ratio relative to peers and growth prospects to determine if the valuation is reasonable.</li>
        </ol>
    </div>
    <!-- End of Qualitative Section -->

    <h4>Investor Profile Suggestions:</h4>
     <ul>
        <li><strong>Long-Term Investors:</strong> The analysis suggests Disney is a potentially strong consideration for those believing in the company's ability to capitalize on its brand, IP, and global reach. The streaming segment's growth potential and the resilience from diversification are key factors, despite current unprofitability in streaming.</li>
        <li><strong>Short-Term Investors:</strong> The stock may experience volatility due to economic uncertainties and streaming competition. A cautious approach, focusing on quarterly earnings and subscriber metrics, is advised.</li>
    </ul>


    <!-- Key Metrics & Fundamentals Section -->
    <h2>Key Metrics & Fundamentals (Live Data)</h2>
    <div class="flex-container">
        <!-- Price Performance Box -->
        <div class="metrics-box">
            <div class="metrics-title">Price & Performance</div>
            <div class="metric-row"><span>Current Price</span><span>${{ "%.2f"|format(key_metrics.current_price) if key_metrics.current_price is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>52-Week Range</span><span>${{ "%.2f"|format(key_metrics.fifty_two_week_low) if key_metrics.fifty_two_week_low is number else 'N/A' }} - ${{ "%.2f"|format(key_metrics.fifty_two_week_high) if key_metrics.fifty_two_week_high is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Dist. from High</span><span class="{{ 'negative' if key_metrics.current_to_high is number and key_metrics.current_to_high < -5 else 'neutral' }}">{{ "%.1f"|format(key_metrics.current_to_high) if key_metrics.current_to_high is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Dist. from Low</span><span class="{{ 'positive' if key_metrics.current_to_low is number and key_metrics.current_to_low > 5 else 'neutral' }}">{{ "%.1f"|format(key_metrics.current_to_low) if key_metrics.current_to_low is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>YTD Return</span><span class="{{ 'positive' if ytd_return is number and ytd_return > 1 else 'negative' if ytd_return is number and ytd_return < -1 else 'neutral' }}">{{ "%.2f"|format(ytd_return) if ytd_return is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>{{ period_label }}</span><span class="{{ 'positive' if period_return is number and period_return > 1 else 'negative' if period_return is number and period_return < -1 else 'neutral' }}">{{ "%.2f"|format(period_return) if period_return is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Volatility (Ann.)</span><span>{{ "%.1f"|format(key_metrics.annualized_volatility) if key_metrics.annualized_volatility is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Beta (Calculated)</span><span>{{ "%.2f"|format(beta_val) if beta_val is number else 'N/A' }}</span></div>
        </div>
        <!-- Technical Indicators Box -->
        <div class="metrics-box">
            <div class="metrics-title">Technical Indicators</div>
            <div class="metric-row"><span>50-Day MA</span><span>${{ "%.2f"|format(key_metrics.ma50) if key_metrics.ma50 is number else 'N/A' }} {% if key_metrics.current_price is number and key_metrics.ma50 is number %}<span class="{{ 'positive' if key_metrics.current_price > key_metrics.ma50 else 'negative' }}">({{ 'Above' if key_metrics.current_price > key_metrics.ma50 else 'Below' }})</span>{% endif %}</span></div>
            <div class="metric-row"><span>200-Day MA</span><span>${{ "%.2f"|format(key_metrics.ma200) if key_metrics.ma200 is number else 'N/A' }} {% if key_metrics.current_price is number and key_metrics.ma200 is number %}<span class="{{ 'positive' if key_metrics.current_price > key_metrics.ma200 else 'negative' }}">({{ 'Above' if key_metrics.current_price > key_metrics.ma200 else 'Below' }})</span>{% endif %}</span></div>
            <div class="metric-row"><span>RSI (14)</span><span>{% if key_metrics.rsi is number %}<span class="{{ 'positive' if key_metrics.rsi < 35 else 'negative' if key_metrics.rsi > 65 else 'neutral' }}">{{ "%.1f"|format(key_metrics.rsi) }} ({{ 'Oversold Zone' if key_metrics.rsi < 35 else 'Overbought Zone' if key_metrics.rsi > 65 else 'Neutral' }})</span>{% else %}N/A{% endif %}</span></div>
            <div class="metric-row"><span>Volume Trend (2W/3M)</span><span class="{{ 'positive' if key_metrics.volume_trend is number and key_metrics.volume_trend > 10 else 'negative' if key_metrics.volume_trend is number and key_metrics.volume_trend < -10 else 'neutral' }}">{{ "%.1f"|format(key_metrics.volume_trend) if key_metrics.volume_trend is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Avg Vol (10d)</span><span>{{ "{:,.0f}".format(fundamentals.get('Avg Volume 10 Day')) if fundamentals.get('Avg Volume 10 Day') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Avg Vol (3m)</span><span>{{ "{:,.0f}".format(fundamentals.get('Avg Volume 3 Month')) if fundamentals.get('Avg Volume 3 Month') is number else 'N/A' }}</span></div>
        </div>
        <!-- Valuation Box -->
        <div class="metrics-box">
            <div class="metrics-title">Valuation</div>
            <div class="metric-row"><span>Market Cap</span><span>${{ "{:,.1f}".format(fundamentals.get('Market Cap', 0)/1e9) if fundamentals.get('Market Cap') is number else 'N/A' }} B</span></div>
            <div class="metric-row"><span>Enterprise Value</span><span>${{ "{:,.1f}".format(fundamentals.get('Enterprise Value', 0)/1e9) if fundamentals.get('Enterprise Value') is number else 'N/A' }} B</span></div>
            <div class="metric-row"><span>Trailing P/E</span><span>{{ "%.2f"|format(fundamentals.get('Trailing PE')) if fundamentals.get('Trailing PE') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Forward P/E</span><span>{{ "%.2f"|format(fundamentals.get('Forward PE')) if fundamentals.get('Forward PE') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>PEG Ratio</span><span>{{ "%.2f"|format(fundamentals.get('PEG Ratio')) if fundamentals.get('PEG Ratio') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Price/Sales (TTM)</span><span>{{ "%.2f"|format(fundamentals.get('Price to Sales')) if fundamentals.get('Price to Sales') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Price/Book</span><span>{{ "%.2f"|format(fundamentals.get('Price to Book')) if fundamentals.get('Price to Book') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>EV/EBITDA</span><span>{{ "%.2f"|format(fundamentals.get('EV/EBITDA')) if fundamentals.get('EV/EBITDA') is number else 'N/A' }}</span></div>
        </div>
        <!-- Financial Health Box -->
        <div class="metrics-box">
            <div class="metrics-title">Financial Health</div>
            <div class="metric-row"><span>Profit Margin</span><span>{{ "%.2f"|format(fundamentals.get('Profit Margin')) if fundamentals.get('Profit Margin') is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Operating Margin</span><span>{{ "%.2f"|format(fundamentals.get('Operating Margin')) if fundamentals.get('Operating Margin') is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Return on Equity</span><span>{{ "%.2f"|format(fundamentals.get('ROE')) if fundamentals.get('ROE') is number else 'N/A' }}%</span></div>
            <div class="metric-row"><span>Return on Assets</span><span>{{ "%.2f"|format(fundamentals.get('ROA')) if fundamentals.get('ROA') is number else 'N/A' }}%</span></div>
            <!-- <<< Corrected Debt/Equity Display >>> -->
            <div class="metric-row"><span>Debt/Equity</span><span>{% set de_ratio = fundamentals.get('Debt to Equity') %}{% if de_ratio is number %}{{ "%.1f"|format(de_ratio if de_ratio > 5 else de_ratio * 100) }}%{% else %}N/A{% endif %}</span></div>
            <div class="metric-row"><span>Current Ratio</span><span>{{ "%.2f"|format(fundamentals.get('Current Ratio')) if fundamentals.get('Current Ratio') is number else 'N/A' }}</span></div>
            <div class="metric-row"><span>Quick Ratio</span><span>{{ "%.2f"|format(fundamentals.get('Quick Ratio')) if fundamentals.get('Quick Ratio') is number else 'N/A' }}</span></div>
             <!-- <<< Corrected Dividend Yield Display >>> -->
            <div class="metric-row"><span>Fwd Dividend Yield</span><span>{% if live_dividend_yield_decimal is number %}{{ "%.2f"|format(live_dividend_yield_decimal) }}%{% else %}N/A{% endif %}</span></div>
        </div>
    </div>

    <!-- <<< REVISED: Educational Section for Metrics >>> -->
    <div class="education-section">
        <h3>Understanding Key Metrics (For Long-Term Investing)</h3>
        <div class="metric-explainer">
            <h4>Is the Stock Price Reasonable? (Valuation Ratios)</h4>
            <p>These ratios help us decide if the stock price is high or low compared to the company's earnings, sales, or growth potential.</p>
            <p><strong>P/E Ratio (Price-to-Earnings):</strong> Compares the stock price to the company's profit per share.
                <ul><li>Lower P/E can mean the stock is cheaper relative to earnings (potentially good value).</li><li>Higher P/E can mean the stock is more expensive, possibly because investors expect high future growth (or it might be overvalued).</li></ul>
                For context, Disney's Trailing P/E (based on past earnings) is <strong>{{ "%.1f"|format(fundamentals.get('Trailing PE')) if fundamentals.get('Trailing PE') is number else 'N/A' }}</strong>. The Forward P/E (based on expected future earnings) is <strong>{{ "%.1f"|format(fundamentals.get('Forward PE')) if fundamentals.get('Forward PE') is number else 'N/A' }}</strong>. Comparing these to competitors (like NFLX, CMCSA in the table below) and Disney's own history helps judge if the current P/E is reasonable.</p>
            <p><strong>PEG Ratio:</strong> Compares the P/E Ratio to the company's expected earnings growth rate.
                <ul><li>A PEG around or below 1 suggests the price is reasonable given the expected growth (generally desirable).</li><li>A PEG significantly above 1 or 2 might suggest the stock price is high compared to expected growth.</li></ul>
                Disney's PEG is <strong>{{ "%.2f"|format(fundamentals.get('PEG Ratio')) if fundamentals.get('PEG Ratio') is number else 'N/A' }}</strong>. (Note: Growth estimates can vary).</p>
            <p><strong>P/S Ratio (Price-to-Sales):</strong> Compares the stock price (via market cap) to the company's total revenue. Useful when earnings are volatile. Lower is generally seen as cheaper relative to sales. Disney's P/S is <strong>{{ "%.2f"|format(fundamentals.get('Price to Sales')) if fundamentals.get('Price to Sales') is number else 'N/A' }}</strong>.</p>
            <p><strong>EV/EBITDA:</strong> Compares the company's total value (Enterprise Value) to its operational earnings (EBITDA). Lower values are often preferred, suggesting the company might be cheaper relative to its core earnings power.</p>
        </div>
        <div class="metric-explainer">
            <h4>Is the Company Financially Healthy? (Health Ratios)</h4>
            <p>These ratios help us understand if the company manages its money well and isn't taking on too much risk.</p>
            <p><strong>Debt/Equity (D/E):</strong> Shows how much debt the company uses compared to its own funds (equity).
                <ul><li>Lower D/E (e.g., below 100% or 1.0) generally means less financial risk.</li><li>Very high D/E means the company relies heavily on borrowing, which can be risky if business slows down.</li></ul>
                Disney's D/E is <strong>{% set de_ratio = fundamentals.get('Debt to Equity') %}{% if de_ratio is number %}{{ "%.1f"|format(de_ratio if de_ratio > 5 else de_ratio * 100) }}%{% else %}N/A{% endif %}</strong>.</p>
            <p><strong>Current Ratio:</strong> Compares short-term assets (cash, receivables) to short-term debts.
                <ul><li>Ratio > 1 means the company likely has enough liquid resources to cover immediate obligations (Generally good).</li><li>Ratio < 1 means short-term debts exceed easily accessible assets, which could indicate liquidity risk.</li></ul>
                Disney's is <strong>{{ "%.2f"|format(fundamentals.get('Current Ratio')) if fundamentals.get('Current Ratio') is number else 'N/A' }}</strong>.</p>
            <p><strong>Quick Ratio:</strong> Similar to Current Ratio, but excludes inventory (harder to sell quickly). A stricter test of immediate liquidity. Disney's is <strong>{{ "%.2f"|format(fundamentals.get('Quick Ratio')) if fundamentals.get('Quick Ratio') is number else 'N/A' }}</strong>.</p>
        </div>
         <div class="metric-explainer">
            <h4>How Well is the Company Performing? (Profitability/Performance)</h4>
            <p>These metrics show how good the company is at making money from its business.</p>
            <p><strong>Profit Margin:</strong> How much profit the company keeps for every dollar of sales. Higher percentages are better, showing efficiency. Disney's margin is <strong>{{ "%.1f"|format(fundamentals.get('Profit Margin')) if fundamentals.get('Profit Margin') is number else 'N/A' }}%</strong>.</p>
             <p><strong>ROE (Return on Equity):</strong> Shows how much profit the company generates using the money invested by shareholders. Higher percentages are generally better, indicating good returns for owners. Disney's ROE is <strong>{{ "%.1f"|format(fundamentals.get('ROE')) if fundamentals.get('ROE') is number else 'N/A' }}%</strong>.</p>
             <p><strong>Dividend Yield:</strong> The annual dividend payment as a percentage of the stock price. This is direct cash return to shareholders. A higher yield means more income relative to price, but ensure the company can sustain the payments. Disney's forward yield is <strong>{% if live_dividend_yield_decimal is number %}{{ "%.2f"|format(live_dividend_yield_decimal) }}%{% else %}N/A{% endif %}</strong>.</p>
         </div>
    </div>


    <!-- Charts Section -->
    <h2>Charts & Visual Analysis</h2>
    <!-- Single Grid Image -->
    <div class="chart-container" style="flex-basis: 100%;">
        <img src="{{ plot_dir }}/main_charts_grid.png" alt="Main Charts Grid" class="chart-img">
        <p class="chart-caption">Grid view: Price/MAs, RSI, Volume, MACD, Bollinger Bands, Quarterly Performance.</p>
    </div>
    <!-- Competitor Chart -->
     <div class="chart-container" style="flex-basis: 100%;">
        <img src="{{ plot_dir }}/competitor_comparison.png" alt="Competitor Comparison Chart" class="chart-img">
        <p class="chart-caption">Comparison with Key Competitors across select metrics.</p>
    </div>

    <!-- <<< REVISED: Educational Section for Charts with Simple Dynamic Comments >>> -->
    <div class="education-section">
        <h3>Understanding the Charts</h3>
        <div class="chart-explainer">
            <h4>Price & Moving Averages (MAs)</h4>
            <p>This chart shows the stock's closing price history (blue line). The orange and red dashed lines are Moving Averages (MAs), which smooth out price action to show the trend.</p>
                <ul>
                    <li><strong>50-Day MA (Orange):</strong> Shows the average price over the last 50 trading days (approx. 2-3 months) - indicates the shorter-term trend.</li>
                    <li><strong>200-Day MA (Red):</strong> Shows the average price over the last 200 trading days (approx. 9-10 months) - indicates the longer-term trend.</li>
                    <li><strong>What to look for:</strong> As a long-term investor, seeing the price consistently above the 200-day MA is often considered a positive sign of a healthy long-term uptrend. When the price dips near or below the 200-MA, it might represent a potential buying opportunity *if* the company fundamentals are still strong, but it also signals trend weakness.</li>
                    <!-- <<< Dynamic Comment based on key_metrics >>> -->
                    {% if key_metrics.current_price is number and key_metrics.ma200 is number %}
                        <li><strong>Current Status:</strong> The price (${{ "%.2f"|format(key_metrics.current_price) }}) is currently <strong>{{ 'Above' if key_metrics.current_price > key_metrics.ma200 else 'Below' }}</strong> the 200-day MA (${{ "%.2f"|format(key_metrics.ma200) }}), suggesting {{ 'positive' if key_metrics.current_price > key_metrics.ma200 else 'negative or weakening' }} long-term momentum based on this indicator.</li>
                    {% endif %}
                </ul>
        </div>
         <div class="chart-explainer">
            <h4>RSI (Relative Strength Index)</h4>
            <p>This chart helps gauge if the stock is potentially "overbought" (maybe too high, too fast) or "oversold" (maybe too low, too fast). It's a momentum indicator shown by the purple line (0-100).</p>
                 <ul>
                    <li><strong>Above 70 (Red Line):</strong> Often considered "Overbought". Suggests recent price gains were strong; caution might be needed as a pullback is possible. Might not be the best time to buy immediately.</li>
                    <li><strong>Below 30 (Green Line):</strong> Often considered "Oversold". Suggests recent price drops were sharp; a bounce might be possible. Could indicate a potential entry point, especially if the drop seems overdone.</li>
                    <li><strong>What to look for:</strong> For buying opportunities, an RSI dipping towards or below 30 (entering the green shaded area) can signal a potential short-term low, but always check company fundamentals too.</li>
                    <!-- <<< Dynamic Comment based on key_metrics >>> -->
                    {% if key_metrics.rsi is number %}
                        {% set rsi_zone = 'Overbought (>70)' if key_metrics.rsi > 70 else 'Oversold (<30)' if key_metrics.rsi < 30 else 'Neutral (30-70)' %}
                        <li><strong>Current Status:</strong> The current RSI is <strong>{{ "%.1f"|format(key_metrics.rsi) }}</strong>, which is in the <strong>{{ rsi_zone }}</strong> zone.</li>
                    {% endif %}
                </ul>
        </div>
         <div class="chart-explainer">
            <h4>Trading Volume</h4>
             <p>Shows how many shares were traded each day (blue bars). Volume helps confirm price trends. The dashed line shows the 3-Month Average volume.</p>
                 <ul>
                    <li><strong>High Volume on Up Days:</strong> Suggests strong buying interest.</li>
                    <li><strong>High Volume on Down Days:</strong> Suggests strong selling pressure.</li>
                    <li><strong>Compare to Average:</strong> Look if recent bars are significantly taller (higher volume) or shorter (lower volume) than the dashed average line. High volume makes the price action on that day more significant.</li>
                    <!-- Note: Dynamic comment comparing current to average volume requires passing avg vol -->
                </ul>
        </div>
        <div class="chart-explainer">
            <h4>MACD (Moving Average Convergence Divergence)</h4>
             <p>Another momentum indicator using moving averages to show trend changes.</p>
                 <ul>
                    <li><strong>MACD Line (Blue) vs Signal Line (Red):</strong> When Blue crosses Above Red = Bullish signal. When Blue crosses Below Red = Bearish signal.</li>
                    <li><strong>Histogram (Green/Red Bars):</strong> Shows difference between lines. Growing bars = strengthening momentum.</li>
                    <li><strong>Zero Line Crossover:</strong> MACD crossing above zero = broader positive momentum shift. Below zero = negative shift.</li>
                     <!-- Note: Dynamic comment requires passing latest MACD/Signal values -->
                </ul>
        </div>
        <div class="chart-explainer">
            <h4>Bollinger Bands®</h4>
            <p>Shows price volatility relative to a moving average.</p>
                 <ul>
                    <li><strong>Wide Bands:</strong> High volatility. <strong>Narrow Bands ("Squeeze"):</strong> Low volatility (can precede a breakout).</li>
                    <li><strong>Price vs Bands:</strong> Touching lower band = potentially oversold relative to recent volatility. Touching upper band = potentially overbought relative to recent volatility.</li>
                     <!-- Note: Dynamic comment requires passing band values/width -->
                </ul>
        </div>
         <div class="chart-explainer">
            <h4>Quarterly Performance</h4>
             <p>Shows reported Revenue (blue bars, left axis) and Operating Income (green line, right axis) over the past ~2 years.</p>
                 <ul>
                    <li><strong>Look for:</strong> Growth trends in bars (Revenue) and line (Income). Is income growing with revenue (stable/improving margins)? Any seasonality?</li>
                     <!-- Note: Dynamic comment requires analyzing quarterly_data list -->
                </ul>
        </div>
        <!-- <<< NEW: Explanation for Competitor Subplots >>> -->
         <div class="chart-explainer">
            <h4>Competitor Comparison Charts</h4>
            <p>These 6 small charts compare Disney (dark blue bar) against peers on key metrics:</p>
                 <ul>
                    <li><strong>Market Cap:</strong> Company size based on stock price.</li>
                    <li><strong>P/E Ratio:</strong> Valuation relative to earnings (lower often better, N/A if unprofitable).</li>
                    <li><strong>EV/EBITDA:</strong> Valuation relative to operating earnings (lower often better).</li>
                    <li><strong>Profit Margin (%):</strong> Profitability (higher is better).</li>
                    <li><strong>Price to Sales:</strong> Valuation relative to revenue (lower often better).</li>
                    <li><strong>Yearly Return (%):</strong> Recent stock performance (higher is better, but past performance doesn't guarantee future results).</li>
                </ul>
             <p>Quickly see if Disney is trading at a premium/discount, is more/less profitable, or has performed better/worse than competitors recently.</p>
        </div>
    </div>

    <!-- Business Segments Section -->
    <h2>Business Overview</h2>
    <h3>Segment Performance</h3>
    {% if segment_revenue and 'Segments' in segment_revenue and segment_revenue['Segments'] %}
    <table>
        <thead><tr><th>Segment</th><th>Revenue (Billions USD)</th><th>YoY Growth</th><th>Contribution</th></tr></thead>
        <tbody>
        {% set total_rev = segment_revenue['Revenue (Billions)'] | sum %}
        {% for i in range(segment_revenue['Segments']|length) %}
        <tr>
            <td>{{ segment_revenue['Segments'][i] }}</td>
            <td>${{ "%.2f"|format(segment_revenue['Revenue (Billions)'][i]) }}</td>
            <td class="{{ 'positive' if segment_revenue['Growth'][i] > 0 else 'negative' if segment_revenue['Growth'][i] < 0 else 'neutral' }}">{{ "%.1f"|format(segment_revenue['Growth'][i]) }}%</td>
            <td>{{ "%.1f"|format(segment_revenue['Revenue (Billions)'][i] / total_rev * 100 if total_rev else 0) }}%</td>
        </tr>
        {% endfor %}
                </tbody>
    </table>
    <!-- <<< NEW Interpretation Note >>> -->
    <p><strong>Interpretation:</strong> This breakdown (using illustrative data) highlights Disney's diversification. Experiences (Parks) shows strong growth (+16.0%), becoming a major revenue driver (~38%). Entertainment (including streaming & traditional TV) is the largest segment (~44%) but faces challenges (negative growth). Sports (ESPN) is relatively stable. Ideally, we want to see growth across most segments, or stabilization in declining ones alongside strong growth in others (like Experiences and potentially future streaming profitability).</p>
<p class="note"><i>Note: Segment data shown is based on prior analysis and is illustrative. It requires manual updates from the latest official earnings reports for current accuracy.</i></p>    {% else %}
    <p>Segment revenue data not available or formatted incorrectly.</p>
    {% endif %}

    <h3>Streaming Services Snapshot</h3>
    {% if streaming_breakdown and 'Service' in streaming_breakdown and streaming_breakdown['Service'] %}
     <table>
         <thead><tr><th>Service</th><th>Subscribers (Millions)</th><th>Avg. Revenue Per User (ARPU)</th><th>YoY Sub Growth</th></tr></thead>
         <tbody>
         {% for i in range(streaming_breakdown['Service']|length) %}
        <tr>
            <td>{{ streaming_breakdown['Service'][i] }}</td>
            <td>{{ "%.1f"|format(streaming_breakdown['Subscribers (millions)'][i]) }} M</td>
            <td>${{ "%.2f"|format(streaming_breakdown['ARPU'][i]) }}</td>
            <td class="{{ 'positive' if streaming_breakdown['YoY Growth (%)'][i] > 0 else 'negative' if streaming_breakdown['YoY Growth (%)'][i] < 0 else 'neutral' }}">{{ "%.1f"|format(streaming_breakdown['YoY Growth (%)'][i]) }}%</td>
        </tr>
        {% endfor %}
                </tbody>
    </table>
     <!-- <<< NEW Interpretation Note >>> -->
     <p><strong>Interpretation:</strong> This snapshot (using illustrative data) shows strong subscriber numbers for Disney+ Core (117.6M) and Hulu (50.2M). ARPU (Average Revenue Per User) is a key metric for profitability – higher is better. Disney+ Core's ARPU ($7.28) is lower than Hulu's ($11.84), reflecting different pricing tiers and markets. Positive subscriber growth (Disney+, Hulu) is good, but the decline for ESPN+ (-2.0%) and the overall challenge of making streaming profitable are key areas investors watch closely.</p>
<p class="note"><i>Note: Streaming data shown is based on prior analysis and is illustrative. It requires manual updates from the latest official earnings reports for current accuracy.</i></p>    {% else %}
    <p>Streaming breakdown data not available or formatted incorrectly.</p>
    {% endif %}

    <!-- Competitor Table Section -->
    <h2>Competitor Comparison (Live Data)</h2>
    {% if competitors_data is defined and competitors_data is not none and competitors_data %}
    <table>
        <thead><tr><th>Company</th><th>Mkt Cap (B)</th><th>Revenue (B)</th><th>Profit Margin (%)</th><th>PE Ratio</th><th>EV/EBITDA</th><th>Price/Sales</th><th>1-Yr Return (%)</th></tr></thead>
        <tbody>
        {% for idx, comp in competitors_data.items() %}
        <tr {% if idx == ticker %}style="font-weight: bold; background-color: #e8f0fe;"{% endif %}>
            <td>{{ comp.get('Name', idx) }}</td>
            <td>{% if comp.get('Market Cap (Billions)') is number %}${{ "%.1f"|format(comp.get('Market Cap (Billions)')) }}{% else %}N/A{% endif %}</td>
            <td>{% if comp.get('Revenue (Billions)') is number %}${{ "%.1f"|format(comp.get('Revenue (Billions)')) }}{% else %}N/A{% endif %}</td>
            <td>{% if comp.get('Profit Margin (%)') is number %}{{ "%.1f"|format(comp.get('Profit Margin (%)')) }}%{% else %}N/A{% endif %}</td>
            <td>{% if comp.get('PE Ratio') is number %}{{ "%.1f"|format(comp.get('PE Ratio')) }}{% else %}N/A{% endif %}</td>
            <td>{% if comp.get('EV/EBITDA') is number %}{{ "%.1f"|format(comp.get('EV/EBITDA')) }}{% else %}N/A{% endif %}</td>
            <td>{% if comp.get('Price to Sales') is number %}{{ "%.2f"|format(comp.get('Price to Sales')) }}{% else %}N/A{% endif %}</td> {# Added Price/Sales #}
            {% set yr_return = comp.get('Yearly Return (%)') %}
            <td class="{{ 'positive' if yr_return is number and yr_return > 1 else 'negative' if yr_return is number and yr_return < -1 else 'neutral' }}">{% if yr_return is number %}{{ "%.1f"|format(yr_return) }}%{% else %}N/A{% endif %}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}<p>Competitor data could not be fully processed or displayed.</p>{% endif %}
        </tbody>
    </table>
    <!-- <<< NEW Interpretation Note >>> -->
    <p><strong>Interpretation:</strong> Comparing Disney to peers:
        <ul>
            <li><strong>Size (Market Cap):</strong> Disney ($ {{ "%.1f"|format(competitors_data[ticker]['Market Cap (Billions)']) if ticker in competitors_data and competitors_data[ticker]['Market Cap (Billions)'] is number else 'N/A' }}B) is large, but smaller than giants like Amazon and Netflix in market value.</li>
            <li><strong>Valuation:</strong> Disney's P/E ({{ "%.1f"|format(competitors_data[ticker]['PE Ratio']) if ticker in competitors_data and competitors_data[ticker]['PE Ratio'] is number else 'N/A' }}) and EV/EBITDA ({{ "%.1f"|format(competitors_data[ticker]['EV/EBITDA']) if ticker in competitors_data and competitors_data[ticker]['EV/EBITDA'] is number else 'N/A' }}) appear more moderate than Netflix's but higher than Comcast's. Its Price/Sales ({{ "%.2f"|format(competitors_data[ticker]['Price to Sales']) if ticker in competitors_data and competitors_data[ticker]['Price to Sales'] is number else 'N/A' }}) is relatively low compared to Netflix.</li>
            <li><strong>Profitability:</strong> Disney's Profit Margin ({{ "%.1f"|format(competitors_data[ticker]['Profit Margin (%)']) if ticker in competitors_data and competitors_data[ticker]['Profit Margin (%)'] is number else 'N/A' }}%) is positive, unlike WBD and Paramount, but significantly lower than Netflix and Comcast.</li>
            <li><strong>Performance:</strong> Disney's 1-Year Return ({{ "%.1f"|format(competitors_data[ticker]['Yearly Return (%)']) if ticker in competitors_data and competitors_data[ticker]['Yearly Return (%)'] is number else 'N/A' }}%) lags behind Netflix significantly but is better than Comcast's recent performance.</li>
        </ul>
        This comparison helps assess if Disney is attractively valued or performing well <strong>relative</strong> to similar companies.
    </p>

    <!-- <<< NEW: Historical Financial Table from PDF >>> -->
    <h2>Historical Financial Performance (2020-2024)</h2>
    <p class="note">Data below based on external analysis report, using reported fiscal year end dates.</p>
    <table>
        <thead>
            <tr>
                <th>Financial Metric</th>
                <th>FY 2020 (Oct 3)</th>
                <th>FY 2021 (Oct 2)</th>
                <th>FY 2022 (Oct 1)</th>
                <th>FY 2023 (Sep 30)</th>
                <th>FY 2024 (Sep 28, Est.)</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Revenue (Billion USD)</td><td>65.388</td><td>67.418</td><td>82.722</td><td>88.898</td><td>91.361</td></tr>
            <tr><td>Gross Profit (Billion USD)</td><td>21.508</td><td>22.287</td><td>28.321</td><td>29.697</td><td>32.663</td></tr>
            <tr><td>Operating Income (Billion USD)</td><td>4.609</td><td>3.005</td><td>6.533</td><td>5.100</td><td>8.319</td></tr>
            <tr><td>Net Income (Billion USD)</td><td>-2.442</td><td>2.536</td><td>3.553</td><td>3.390</td><td>5.773</td></tr>
            <tr><td>Total Assets (Billion USD)</td><td>201.549</td><td>203.609</td><td>203.631</td><td>205.579</td><td>196.219</td></tr>
            <tr><td>Total Liabilities (Billion USD)</td><td>113.286</td><td>110.598</td><td>104.752</td><td>101.622</td><td>90.697</td></tr>
            <tr><td>Total Equity (Billion USD)</td><td>88.263</td><td>93.011</td><td>98.879</td><td>103.957</td><td>105.522</td></tr>
            <tr><td>Net Cash from Ops (Billion USD)</td><td>7.616</td><td>5.566</td><td>6.002</td><td>9.866</td><td>13.971</td></tr>
            <tr><td>Free Cash Flow (Billion USD)</td><td>3.594</td><td>1.988</td><td>1.059</td><td>4.897</td><td>8.559</td></tr>
        </tbody>
    </table>
        </tbody>
    </table>
    <!-- <<< NEW Interpretation Note >>> -->
    <p><strong>Interpretation:</strong> This historical view (based on external analysis) shows Disney's recovery post-COVID (FY2020 had a net loss). Key observations:
        <ul>
            <li><strong>Revenue Growth:</strong> Consistent top-line growth from $65B in FY20 to $91B in FY24 (Est.).</li>
            <li><strong>Profitability Recovery:</strong> Net Income turned positive in FY21 and grew significantly by FY24, though Operating Income dipped in FY23 before recovering.</li>
            <li><strong>Cash Flow Strength:</strong> Both Operating Cash Flow and Free Cash Flow (cash left after investments) show strong improvement, especially in FY23 and FY24, indicating better cash generation and financial flexibility.</li>
            <li><strong>Balance Sheet:</strong> Total Assets have remained relatively stable, while Total Liabilities have decreased, leading to steady growth in Total Equity (the company's net worth).</li>
        </ul>
        Overall, the historical trend suggests a company that navigated the pandemic, returned to growth and profitability, and improved its cash flow situation significantly by FY24.
    </p>

     <!-- <<< NEW: SWOT Analysis from PDF >>> -->
    <h2>SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)</h2>
    <p class="note">Based on external analysis.</p>
    <div class="flex-container">
        <div class="metrics-box">
            <div class="metrics-title">Strengths</div>
            <ul class="swot-list">
                <li><strong>Strong Brand and IP:</strong> Iconic franchises (Marvel, Star Wars, Pixar) drive revenue across segments.</li>
                <li><strong>Diversified Revenue Streams:</strong> Resilience from multiple business segments (media, parks, products, streaming).</li>
                <li><strong>Global Reach:</strong> Theme parks and content distribution provide exposure to international markets.</li>
                <li><strong>Strong Cash Flow:</strong> Improved cash generation indicates financial stability and reinvestment ability.</li>
            </ul>
        </div>
        <div class="metrics-box">
            <div class="metrics-title">Weaknesses</div>
             <ul class="swot-list">
                <li><strong>High Operating Costs:</strong> Capital-intensive nature of theme parks and content production.</li>
                <li><strong>Dependence on Blockbusters:</strong> Film segment success tied to performance of major movie releases.</li>
                <li><strong>Streaming Profitability:</strong> Disney+ has yet to achieve sustained profitability.</li>
            </ul>
        </div>
        <div class="metrics-box">
            <div class="metrics-title">Opportunities</div>
             <ul class="swot-list">
                 <li><strong>Emerging Markets:</strong> Growth potential in middle-class populations (e.g., China, India).</li>
                 <li><strong>Streaming Growth:</strong> Continued expansion of Disney+ and Hulu for subscribers and content.</li>
                 <li><strong>Technological Innovations:</strong> Leveraging VR, AR, and AI to enhance offerings.</li>
                 <li><strong>Sustainability Initiatives:</strong> Attracting environmentally conscious consumers and investors via CSR programs.</li>
            </ul>
        </div>
        <div class="metrics-box">
            <div class="metrics-title">Threats</div>
             <ul class="swot-list">
                 <li><strong>Intense Competition:</strong> Major media conglomerates and streaming services vie for market share.</li>
                 <li><strong>Economic Downturns:</strong> Impact on consumer discretionary spending affecting parks and advertising.</li>
                 <li><strong>Piracy and IP Theft:</strong> Significant threat to revenue streams.</li>
                 <li><strong>Technological Disruption:</strong> Rapid changes requiring constant adaptation and innovation.</li>
            </ul>
        </div>
    </div>

    <!-- <<< NEW: Emerging Trends from PDF >>> -->
    <h2>Emerging Industry Trends (2024-2025)</h2>
     <ul class="trends-list">
        <li><strong>Streaming Platforms:</strong> Dominance of OTT services (Disney+, Netflix, Prime Video) continues.</li>
        <li><strong>Connected TV (CTV):</strong> Growing significance as an advertising platform.</li>
        <li><strong>Artificial Intelligence (AI):</strong> Influencing content creation, personalization, and recommendations.</li>
        <li><strong>Social Video Platforms:</strong> TikTok, Instagram Reels capturing attention, requiring engagement strategies.</li>
        <li><strong>Immersive Content:</strong> AR, VR, and metaverse offering new audience engagement avenues.</li>
        <li><strong>eSports and Niche Sports:</strong> Growing popularity reflecting evolving entertainment preferences.</li>
    </ul>


    <!-- Original Investment Thesis / Risks / Conclusion replaced by integrated sections -->
    <!-- Remove old variables 'investment_thesis', 'risk_factors', 'catalysts', 'conclusion' from template_vars in Python if desired -->


    <!-- Disclaimer -->
    <div class="disclaimer">
        <p><strong>DISCLAIMER:</strong> This report is auto-generated using quantitative data from Yahoo Finance and incorporates qualitative insights from external analysis for educational purposes. It does not constitute financial advice or a recommendation to buy or sell any stock. Data may contain errors or be subject to delays. Financial models and recommendations are based on algorithms and historical data, which may not predict future performance. Static data (e.g., Segments, Streaming, Historical Table, SWOT) requires manual verification and updates from official company reports for current accuracy. Always conduct your own thorough research and consult with a qualified financial advisor before making any investment decisions.</p>
    </div>
</body>
</html>
"""

# ---- Main Execution Block ----
if __name__ == "__main__":
    print(f"--- Starting Stock Analysis for {TICKER} ---")
    plt.ioff() # Turn off interactive plotting

    # 1. Fetch Data
    print("\n--- Fetching Data ---")
    stock_data_obj, hist_data_raw = fetch_stock_data(TICKER, period=FETCH_PERIOD)
    index_hist_data = fetch_index_data(INDEX_TICKER, period=FETCH_PERIOD)
    competitor_data_map = fetch_competitor_data(COMPETITORS) # Fetches competitor Tickers
    quarterly_metrics = []
    fundamentals = {}
    if stock_data_obj:
        print(f"\n--- Fetching Financials/Fundamentals for {TICKER} ---")
        quarterly_metrics = get_quarterly_metrics(stock_data_obj)
        fundamentals = get_fundamentals(stock_data_obj)
    else:
        print(f"Warning: Could not fetch Ticker object for {TICKER}, financials/fundamentals skipped.")

    # Clean company name
    company_name_raw = fundamentals.get('shortName', TICKER)
    company_name_cleaned = company_name_raw.replace(' (The)', '').replace('(The)', '').strip()
    fundamentals['shortNameClean'] = company_name_cleaned # Store cleaned name

    # 2. Process Data & Calculate Metrics
    print("\n--- Processing Data & Calculating Metrics ---")
    hist_data_processed = None
    key_metrics = None
    main_stock_returns = (None, None, "N/A") # Default value

    if hist_data_raw is not None:
        hist_data_raw.ticker_symbol = TICKER
        hist_data_processed = calculate_technical_indicators(hist_data_raw)
    else:
        print("Warning: Raw historical data is missing, skipping technical indicators.")

    if hist_data_processed is not None:
        hist_data_processed.ticker_symbol = TICKER
        key_metrics = calculate_key_metrics(hist_data_processed)
        main_stock_returns = calculate_returns(hist_data_processed, period_label=TICKER)
    else:
         print("Warning: Processed historical data is missing, skipping key metrics and returns.")


    # Calculate Beta - Requires processed stock data and index data
    print("\n--- Preparing for Beta Calculation ---")
    beta_value = None
    # Ensure index data also has ticker attached if needed by beta function
    if isinstance(index_hist_data, pd.DataFrame): index_hist_data.ticker_symbol = INDEX_TICKER

    if isinstance(hist_data_processed, pd.DataFrame) and not hist_data_processed.empty:
        print(f"Columns in hist_data_processed: {hist_data_processed.columns.tolist()}")
        # Attach ticker symbol again just before calling beta (safer)
        hist_data_processed.ticker_symbol = TICKER
    else:
        print("hist_data_processed is not a valid DataFrame for beta.")

    if isinstance(index_hist_data, pd.DataFrame) and not index_hist_data.empty:
        print(f"Columns in index_hist_data: {index_hist_data.columns.tolist()}")
        # Attach ticker symbol again just before calling beta (safer)
        index_hist_data.ticker_symbol = INDEX_TICKER
    else:
        print("index_hist_data is not a valid DataFrame for beta.")


    if isinstance(hist_data_processed, pd.DataFrame) and not hist_data_processed.empty and \
       isinstance(index_hist_data, pd.DataFrame) and not index_hist_data.empty:
        # Call beta calculation
        beta_value = calculate_beta(hist_data_processed, index_hist_data)
    else:
        print("Skipping Beta calculation due to missing/invalid input data.")


    # Add calculated beta to fundamentals if info beta is missing or desired
    # <<< FIX: Conditional formatting for print statement >>>
    info_beta = fundamentals.get('Beta')
    calculated_beta_str = f"{beta_value:.3f}" if beta_value is not None else "N/A" # Format only if not None

    if beta_value is not None and info_beta is None:
        print(f"Using calculated Beta ({calculated_beta_str}) as 'Beta' fundamental.")
        fundamentals['Beta'] = beta_value # Store the actual number
    elif info_beta is not None:
         print(f"Using Beta from yfinance .info ({info_beta}). Calculated Beta was {calculated_beta_str}.")
    else:
         print(f"Beta fundamental not available from .info and calculation failed/skipped (Calculated: {calculated_beta_str}).")
    # <<< END FIX >>>


    # 3. Generate Recommendation
    print("\n--- Generating Recommendation ---")
    # Initialize variables with default values
    rec = "ERROR"; rec_sum = "Error during analysis."
    reasons_pro_comp, reasons_con_comp, reasons_pro_price, reasons_con_price = [], [], [], []
    rec_score = 0

    if key_metrics and fundamentals and main_stock_returns[0] is not None:
        # <<< Unpack all 7 return values from the updated function >>>
        rec, rec_sum, reasons_pro_comp, reasons_con_comp, reasons_pro_price, reasons_con_price, rec_score = generate_recommendation(
            key_metrics=key_metrics, fundamentals=fundamentals, beta_val=beta_value,
            returns_data=main_stock_returns, ticker_symbol=TICKER )
        print(f"Recommendation generated: {rec} (Score: {rec_score})")
    else:
        print("Warning: Could not generate recommendation due to missing key_metrics, fundamentals, or returns data.")

    # 4. Prepare Competitor Data
    print("\n--- Preparing Competitor Data ---")
    competitors_dict_for_template = None
    df_competitors = None # DataFrame for plot/template

    if isinstance(competitor_data_map, dict) and competitor_data_map:
        # Prepare main ticker data safely (as dictionary first)
        fundamentals_revenue = fundamentals.get('totalRevenue')
        disney_revenue = fundamentals_revenue if isinstance(fundamentals_revenue, numbers.Number) and pd.notna(fundamentals_revenue) else None
        fundamentals_pm_pct = fundamentals.get('Profit Margin')
        disney_profit_margin_decimal = (fundamentals_pm_pct / 100.0) if isinstance(fundamentals_pm_pct, numbers.Number) and pd.notna(fundamentals_pm_pct) else None

        disney_comp_data = {
             'Name': company_name_cleaned,
             'Revenue': disney_revenue, # Raw value passed here
             'Profit Margin': disney_profit_margin_decimal, # Use DECIMAL value here
             'PE Ratio': fundamentals.get('Trailing PE'),
             'EV/EBITDA': fundamentals.get('EV/EBITDA'),
             'Market Cap': fundamentals.get('Market Cap'),
             'Price to Sales': fundamentals.get('Price to Sales'),
             'YTD Return': main_stock_returns[0] if main_stock_returns[0] is not None else None,
             'Yearly Return': main_stock_returns[1] if main_stock_returns[1] is not None and 'Year Return' in main_stock_returns[2] else None
        }
        # Combine with competitor data (which now also has decimal Profit Margin)
        competitor_data_map_with_dis = {TICKER: disney_comp_data, **competitor_data_map}

        # Create initial DataFrame from combined data
        df_comp_raw = pd.DataFrame.from_dict(competitor_data_map_with_dis, orient='index')
        print("Raw competitor data fetched (with decimal PM).")
        

        # --- Clean and prepare the final DataFrame 'df_competitors' ---
        df_competitors = pd.DataFrame(index=df_comp_raw.index) # Start fresh

        # Helper for safe float conversion
        def safe_float(value):
            if pd.isna(value): return None
            if isinstance(value, str) and value.lower() in ['none', 'na', 'n/a', '']: return None
            if isinstance(value, (int, float)): return float(value)
            try: return float(value)
            except (ValueError, TypeError): return None

        # Process columns one by one directly onto df_competitors
        df_competitors['Name'] = df_comp_raw.get('Name', pd.Series(dtype='str')).astype(str)
        market_cap_float = df_comp_raw.get('Market Cap', pd.Series(dtype='object')).apply(safe_float)
        revenue_float = df_comp_raw.get('Revenue', pd.Series(dtype='object')).apply(safe_float)
        profit_margin_float = df_comp_raw.get('Profit Margin', pd.Series(dtype='object')).apply(safe_float)
        df_competitors['PE Ratio'] = df_comp_raw.get('PE Ratio', pd.Series(dtype='object')).apply(safe_float)
        df_competitors['EV/EBITDA'] = df_comp_raw.get('EV/EBITDA', pd.Series(dtype='object')).apply(safe_float)
        df_competitors['Price to Sales'] = df_comp_raw.get('Price to Sales', pd.Series(dtype='object')).apply(safe_float)
        df_competitors['Yearly Return Float'] = df_comp_raw.get('Yearly Return', pd.Series(dtype='object')).apply(safe_float)

        # Create scaled/formatted columns for display/plotting
        df_competitors['Market Cap (Billions)'] = market_cap_float.apply(lambda x: x / 1e9 if x is not None else None)
        df_competitors['Revenue (Billions)'] = revenue_float.apply(lambda x: x / 1e9 if x is not None else None)        
        df_competitors['Profit Margin (%)'] = profit_margin_float.apply(lambda x: x * 100 if x is not None else None)
        df_competitors['Yearly Return (%)'] = df_competitors['Yearly Return Float']

        # Select and order the final columns needed
        final_columns_ordered = [
            'Name', 'Market Cap (Billions)', 'Revenue (Billions)',
            'Profit Margin (%)', 'PE Ratio', 'EV/EBITDA', 'Price to Sales',
            'Yearly Return (%)'
        ]
        df_competitors = df_competitors.reindex(columns=final_columns_ordered)

        print("Cleaned Competitor DataFrame prepared:")
        # print(df_competitors.to_string()) # Uncomment to check final df

        # --- Create dictionary for template FROM the final cleaned DataFrame ---
        competitors_dict_for_template = df_competitors.where(pd.notna(df_competitors), None).to_dict(orient='index')
        
        print("Competitor dictionary for template prepared using .to_dict().")

    else:
        print("Warning: Competitor data map is missing or empty. Skipping competitor table/plot.")
        df_competitors = pd.DataFrame()
        competitors_dict_for_template = {}
    
    
    # 5. Generate Plots
    print("\n--- Generating Plots ---")
    # Call the new grid function instead of individual plots
    plot_main_charts_grid(hist_data_processed, quarterly_metrics, key_metrics, TICKER)
    # Keep the competitor comparison plot call
    plot_competitor_comparison(df_competitors, TICKER)

    # 6. Render HTML Report
    print("\n--- Generating HTML Report ---")
    try:
        rec_class = rec.lower().replace(" ", "").replace("'", "").replace("/", "")
        live_dividend_yield_decimal = fundamentals.get('Dividend Yield')

        # --- Final Aggressive Cleaning for Template ---
        cleaned_key_metrics = {}
        if isinstance(key_metrics, dict):
            for k, v in key_metrics.items():
                if isinstance(v, numbers.Number) and pd.notna(v): cleaned_key_metrics[k] = float(v)
                elif isinstance(v, str): cleaned_key_metrics[k] = v
                else: cleaned_key_metrics[k] = None
        else: cleaned_key_metrics = {}

        cleaned_fundamentals = {}
        if isinstance(fundamentals, dict):
             for k, v in fundamentals.items():
                if isinstance(v, numbers.Number) and pd.notna(v): cleaned_fundamentals[k] = float(v)
                elif isinstance(v, (str, bool)): cleaned_fundamentals[k] = v
                else: cleaned_fundamentals[k] = None
        else: cleaned_fundamentals = {}

        template_vars = {
            "company_name": company_name_cleaned, # Correct order for title
            "ticker": TICKER,
            "current_date": datetime.now().strftime("%B %d, %Y"),
            "rec_class": rec_class,
            "recommendation": rec, # The final BUY or DON'T BUY
            "rec_score": rec_score,
            "rec_summary": rec_sum, # The summary text matching the recommendation
            # <<< Add the four new reason lists >>>
            "reasons_pro_company": reasons_pro_comp,
            "reasons_con_company": reasons_con_comp,
            "reasons_pro_price": reasons_pro_price,
            "reasons_con_price": reasons_con_price,
            # <<< Remove the old 'rec_reasons' if it was still there >>>
            "key_metrics": cleaned_key_metrics,
            "fundamentals": cleaned_fundamentals,
            "beta_val": float(beta_value) if beta_value is not None else None,
            "ytd_return": float(main_stock_returns[0]) if main_stock_returns[0] is not None else None,
            "period_return": float(main_stock_returns[1]) if main_stock_returns[1] is not None else None,
            "period_label": main_stock_returns[2],
            "live_dividend_yield_decimal": float(live_dividend_yield_decimal) if live_dividend_yield_decimal is not None else None,
            "segment_revenue": segment_revenue,
            "streaming_breakdown": streaming_breakdown,
            "competitors_data": competitors_dict_for_template,
            "plot_dir": PLOT_DIR
            # Note: Removed old thesis/risk/catalyst/conclusion text vars as they are in the template now
        }
        # <<< Update HTML Template Title String Reference >>>
        html_template_string_updated = html_template_string.replace(
             "<title>{{ ticker }} Stock Analysis Report</title>",
             "<title>{{ company_name }} ({{ ticker }}) Stock Analysis Report</title>"
        ).replace(
             "<h1>{{ ticker }} ({{ company_name }}) Stock Analysis Report</h1>",
             "<h1>{{ company_name }} ({{ ticker }}) Stock Analysis Report</h1>"
        )

        template = Template(html_template_string_updated) # Use updated string
        html_output = template.render(template_vars)
        with open(REPORT_FILENAME, 'w', encoding='utf-8') as f: f.write(html_output)
        print(f"Successfully generated HTML report: {REPORT_FILENAME}")
        try: webbrowser.open(f'file://{os.path.realpath(REPORT_FILENAME)}')
        except Exception as wb_e: print(f"Could not automatically open report: {wb_e}")

    except Exception as e: print(f"Error during HTML generation: {e}"); traceback.print_exc()

    print(f"\n--- Analysis Complete for {TICKER} ---")