import streamlit as st

import pandas as pd

import numpy as np

import time

import warnings

from datetime import datetime



# --- CONFIG ---

st.set_page_config(page_title="Jim Simons Quant V7", layout="wide")



# Import core libraries (install if missing is maintained in requirements.txt or manually)

try:

    from tvDatafeed import TvDatafeed, Interval

    import yfinance as yf

    from sklearn.ensemble import RandomForestClassifier

except ImportError:

    st.error("⚠️ **Missing Libraries**")

    st.markdown("""

    To fix this, you need to install the required packages.

    

    **If running locally:**

    1. Open your terminal.

    2. Run this command:

    ```bash

    pip install -r requirements.txt

    ```

    3. Refresh this page.

    """)

    st.stop()



warnings.filterwarnings('ignore')



SYMBOL_TV = "XAUUSD"

EXCHANGE_TV = "OANDA"

SYMBOL_YF = "GC=F"



# --- SIDEBAR & CONFIG ---

st.sidebar.header("⚙️ Configuration")



# Credentials Input

st.sidebar.subheader("🔐 TradingView Login (Optional)")

st.sidebar.info("Leave empty to use Guest Mode (delayed data). Login for real-time.")

tv_user = st.sidebar.text_input("Username", value="", help="Your TradingView Username")

tv_pass = st.sidebar.text_input("Password", value="", type="password", help="Your TradingView Password")



mode = st.sidebar.radio("Select Mode", ["Live Auto-Loop", "History Search"])



# --- CORE CLASSES ---

class DataLoader:

    def __init__(self):

        # Cache the TvDatafeed instance to avoid re-login

        if 'tv' not in st.session_state:

            self._login()

        

        # Check if credentials changed

        if tv_user and tv_pass and (getattr(st.session_state, 'tv_user', '') != tv_user):

             self._login()

             

        self.tv = st.session_state.tv

        

    def _login(self):

        try:

            if tv_user and tv_pass:

                st.session_state.tv = TvDatafeed(username=tv_user, password=tv_pass)

                st.session_state.tv_user = tv_user

                st.toast(f"Logged in as {tv_user}!", icon="✅")

            else:

                st.session_state.tv = TvDatafeed()

                st.session_state.tv_user = ""

        except Exception as e:

            st.error(f"Login Failed: {e}")

            st.session_state.tv = TvDatafeed() # Fallback to guest



    def fetch_live(self, n_bars=1000):

        try:

            df = self.tv.get_hist(symbol=SYMBOL_TV, exchange=EXCHANGE_TV, interval=Interval.in_15_minute, n_bars=n_bars)

            if df is not None: 

                df.columns = [c.lower() for c in df.columns]

                return df

        except: pass

        return None



    def fetch_history(self, target_date=None):

        # Hybrid Fetcher (YF Fallback)

        try:

            df = yf.download(SYMBOL_YF, period="59d", interval="15m", progress=False)

            if self._valid(df, target_date): return self._clean(df), "15m"

        except: pass



        try:

            df = yf.download(SYMBOL_YF, period="729d", interval="1h", progress=False)

            if self._valid(df, target_date): return self._clean(df), "1h"

        except: pass

        

        return None, None



    def _valid(self, df, target_date):

        if df is None or df.empty: return False

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        if target_date:

            dates = df.index.strftime('%Y-%m-%d').unique()

            return target_date in dates

        return True



    def _clean(self, df):

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        df.rename(columns={'date': 'datetime'}, inplace=True)

        return df



class FeatureEngineer:

    def add_features(self, df):

        if df is None or df.empty: return None

        df = df.copy()

        if 'datetime' in df.columns:

            df['datetime'] = pd.to_datetime(df['datetime'])

            df.set_index('datetime', inplace=True)

            

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        df['vol_10'] = df['log_ret'].rolling(10).std()

        df['hour'] = df.index.hour

        df.dropna(inplace=True)

        return df



class Strategy:

    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        self.fe_cols = []

    def train(self, df):

        df['target'] = np.where(df['log_ret'].shift(-1) > 0, 1, 0)

        data = df.dropna()

        self.fe_cols = [c for c in data.columns if c not in ['target', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'adj close', 'is_copy']]

        self.model.fit(data[self.fe_cols], data['target'])

    def predict(self, df):

        return self.model.predict_proba(df.iloc[[-1]][self.fe_cols])[0][1]



class RiskManager:

    def get_sl_tp(self, df, action, price):

        if 'high' not in df.columns or 'low' not in df.columns:

            atr = df['close'].rolling(14).std() * 2 

        else:

            tr = df['high'] - df['low']

            atr = tr.rolling(14).mean().iloc[-1]

        

        if np.isnan(atr): atr = 1.0

        

        sl_pips = atr * 2.0

        tp_pips = atr * 3.0

        

        if action == "BUY":

            sl = price - sl_pips

            tp = price + tp_pips

        else:

            sl = price + sl_pips

            tp = price - tp_pips

            

        return round(sl, 2), round(tp, 2)



# --- APP LAYOUT ---

st.title("⚡ Quant System Dashboard")

st.markdown("### XAUUSD (Gold) | Random Forest | Volatility Scalping")



# Initialize

dl = DataLoader()

fe = FeatureEngineer()

strat = Strategy()

rm = RiskManager()



if mode == "Live Auto-Loop":

    st.subheader("📡 Live Auto-Loop")

    col1, col2 = st.columns(2)

    with col1: start_btn = st.button("▶ START System", type="primary")

    with col2: stop_btn = st.button("⏹ STOP System")



    if 'running' not in st.session_state: st.session_state.running = False

    if start_btn: st.session_state.running = True

    if stop_btn: st.session_state.running = False

    

    status = st.empty()

    last_update = st.empty()

    

    if st.session_state.running:

        status.success("Status: 🟢 RUNNING")

        

        with st.spinner("Initializing Model..."):

            raw = dl.fetch_live(3000)

            if raw is None: raw, _ = dl.fetch_history()

            

            if raw is not None:

                df = fe.add_features(raw)

                strat.train(df)

            else:

                st.error("No Data. check credentials.")

                st.session_state.running = False

                st.stop()

        

        while st.session_state.running:

            try:

                raw_live = dl.fetch_live(100)

                if raw_live is not None:

                    df = fe.add_features(raw_live)

                    prob = strat.predict(df)

                    row = df.iloc[-1]

                    price = row['close']

                    

                    action="WAIT"; sl=0.0; tp=0.0

                    if prob > 0.55: 

                        action="BUY"; sl, tp = rm.get_sl_tp(df, "BUY", price)

                    elif prob < 0.45: 

                        action="SELL"; sl, tp = rm.get_sl_tp(df, "SELL", price)

                        

                    with last_update.container():

                        m1, m2, m3 = st.columns(3)

                        m1.metric("Price", f"{price:.2f}")

                        m2.metric("Probability", f"{prob:.2f}")

                        m3.metric("Signal", action, delta_color="normal" if action=="WAIT" else "inverse")

                        

                        if action != "WAIT":

                            st.info(f"Signal: {action} | SL: {sl} | TP: {tp}")

                

                time.sleep(60)

            except Exception as e:

                st.error(f"Error: {e}")

                time.sleep(10)

    else:

        status.warning("Status: 🔴 IDLE")



elif mode == "History Search":

    st.subheader("📅 History Search")

    d_date = st.date_input("Select Date")

    sens = st.selectbox("Sensitivity", ["Low Risk (0.55)", "Medium (0.52)", "High Activity (0.51)"], index=1)

    

    if st.button("🔍 Search"):

        d_str = d_date.strftime("%Y-%m-%d")

        raw, tf = dl.fetch_history(d_str)

        

        if raw is None:

            st.error("No data found.")

        else:

            st.success(f"Data: {tf}")

            df = fe.add_features(raw)

            strat.train(df)

            df['prob'] = strat.model.predict_proba(df[strat.fe_cols])[:, 1]

            res = df[df.index.astype(str).str.startswith(d_str)]

            

            if res.empty:

                st.warning("Market closed.")

            else:

                thresh = float(sens.split('(')[1][:-1])

                trades = []

                pnl = 0

                wins = 0

                

                for i in range(len(res)-1):

                    r = res.iloc[i]; nr = res.iloc[i+1]

                    p=0; s="-"

                    if r['prob'] > thresh: s="BUY"; p=nr['close']-r['close']

                    elif r['prob'] < (1-thresh): s="SELL"; p=r['close']-nr['close']

                    

                    if s!="-":

                        pnl+=p; wins+=(1 if p>0 else 0)

                        trades.append({"Time": r.name.strftime('%H:%M'), "Price": r['close'], "Sig": s, "Prob": r['prob'], "PnL": p})

                        

                st.metric("Net PnL", f"${pnl:.2f}")

                if trades: st.dataframe(trades)

                else: st.info("No trades (Choppy).")
