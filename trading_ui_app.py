"""
Flask Web UI for Elite BankNifty Trading System
Real-time algorithmic trading recommendations with live refresh
"""

from flask import Flask, render_template, jsonify, request
from BankNifty_Elite_Unified_System import EliteBankNiftyUnifiedSystem, OptionsRecommender, PaperTradeTracker
import pandas as pd
import numpy as np
from threading import Lock
from datetime import datetime
import json
import webbrowser
import time

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global state for recommendations
recommendations_cache = {
    'balanced': {
        'recommendations': [],
        'timestamp': None,
        'backtest_stats': {},
        'mode': 'balanced',
        'vix': 20.0
    },
    'aggressive': {
        'recommendations': [],
        'timestamp': None,
        'backtest_stats': {},
        'mode': 'aggressive',
        'vix': 20.0
    },
    'ultra-selective': {
        'recommendations': [],
        'timestamp': None,
        'backtest_stats': {},
        'mode': 'ultra-selective',
        'vix': 20.0
    }
}
recommendations_lock = Lock()

# Initialize system
system = None
last_update = None


def generate_recommendations(mode='both'):
    """Generate fresh recommendations from the trading system for specified mode(s)"""
    global system, last_update
    
    try:
        modes_to_generate = ['balanced', 'aggressive', 'ultra-selective'] if mode == 'both' else [mode]
        
        # Fetch data once
        print("[SYSTEM] Fetching data and calculating indicators...")
        temp_system = EliteBankNiftyUnifiedSystem(initial_capital=100000, mode='balanced')
        
        # Check if data was fetched
        df = temp_system.fetch_data(period='2y')
        if df is None or len(df) == 0:
            print("[ERROR] Failed to fetch data")
            return []
        
        vix = temp_system.vix_value
        print(f"[SYSTEM] Data fetched. VIX: {vix:.1f}")
        
        # Calculate indicators once
        try:
            print("[SYSTEM] Calculating technical indicators...")
            indicators = temp_system.calculate_technical_indicators(df)
            print(f"[SYSTEM] Indicators calculated. Shape: {indicators.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to calculate indicators: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        try:
            print("[SYSTEM] Calculating sentiment indicators...")
            sentiment = temp_system.calculate_sentiment_indicators(df, indicators)
            print(f"[SYSTEM] Sentiment calculated. Shape: {sentiment.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to calculate sentiment: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Prepare features
        try:
            print("[SYSTEM] Preparing features...")
            X, y = temp_system.prepare_features(df, indicators, sentiment, lookahead=5)
            print(f"[SYSTEM] Features prepared. X shape: {X.shape}, y shape: {y.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to prepare features: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Train ensemble
        try:
            print("[SYSTEM] Training ensemble...")
            ensemble = temp_system.train_ensemble(X, y)
            print(f"[SYSTEM] Ensemble trained successfully")
        except Exception as e:
            print(f"[ERROR] Failed to train ensemble: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        for current_mode in modes_to_generate:
            try:
                print(f"[SYSTEM] Generating {current_mode} recommendations...")
                
                # Initialize system for this mode
                system = EliteBankNiftyUnifiedSystem(initial_capital=100000, mode=current_mode)
                system.vix_value = vix  # Use same VIX for consistency
                system.scaler = temp_system.scaler  # Use the same fitted scaler from temp_system
                
                # Generate signals based on mode
                if current_mode == 'balanced':
                    signals, confidences, individual_probs = system.generate_signals_balanced(indicators, sentiment, ensemble)
                elif current_mode == 'aggressive':
                    signals, confidences = system.generate_signals_aggressive(indicators, sentiment, ensemble)
                    individual_probs = None
                elif current_mode == 'ultra-selective':
                    signals, confidences = system.generate_signals_ultra_selective(indicators, sentiment, {}, {}, ensemble)
                    individual_probs = None
                
                print(f"[SYSTEM] Signals generated. Total signals: {len(np.where(signals != 0)[0])}")
                
                # Get recommendations
                recs = system.get_latest_options_recommendations(
                    indicators, sentiment, df, ensemble, mode=current_mode, top_n=10
                )
                
                print(f"[SYSTEM] Recommendations retrieved: {len(recs)} recs for {current_mode}")
                
                # Format recommendations for UI
                formatted_recs = []
                
                for idx, rec in enumerate(recs, 1):
                    try:
                        recommender = OptionsRecommender()
                        rsi = rec['technical_setup']['RSI']
                        signal_type = 1 if 'CALL' in rec['recommendation'] else -1
                        
                        # Use mode-specific stops and targets
                        stop_loss, target1, target2 = recommender.calculate_stops_and_targets(
                            rec['entry_price'], signal_type, rsi=rsi, mode=current_mode
                        )
                        
                        strike = recommender.get_nearest_strike(rec['entry_price'])
                        contract_type = 'PE' if signal_type == -1 else 'CE'
                        
                        # Calculate option premium (estimated)
                        # For BankNifty options, premium is typically 2-8% of strike for ATM options
                        current_price = rec['entry_price']
                        # Estimate premium as percentage of strike
                        premium_pct = 0.05 if abs(strike - current_price) < 100 else 0.035
                        estimated_premium = strike * premium_pct
                        
                        # BankNifty lot size = 40 contracts
                        lot_size = 40
                        premium_per_lot = estimated_premium * lot_size
                        
                        # Get and format expiry date
                        expiry_str = rec['expiry']
                        # Parse expiry format and convert to display format (e.g., "28 APR")
                        try:
                            if 'Weekly' in expiry_str or 'Monthly' in expiry_str:
                                # Calculate actual expiry date
                                from datetime import timedelta
                                today = datetime.now()
                                if 'Weekly' in expiry_str:
                                    days_to_expiry = (4 - today.weekday()) % 7
                                    if days_to_expiry == 0:
                                        days_to_expiry = 7
                                else:  # Monthly
                                    # Last Thursday of the month
                                    last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                                    days_to_expiry = (last_day - today).days
                                
                                expiry_date = today + timedelta(days=days_to_expiry)
                                expiry_display = expiry_date.strftime('%d %b').lstrip('0')
                            else:
                                expiry_display = expiry_str
                        except:
                            expiry_display = expiry_str
                        
                        # Calculate expected profit/loss percentages
                        profit_pct = ((target1 - rec['entry_price']) / rec['entry_price']) * 100
                        loss_pct = ((stop_loss - rec['entry_price']) / rec['entry_price']) * 100
                        
                        # Win prediction based on consensus and ML confidence
                        win_prediction = (rec['consensus'] * rec['confidence']) / 100 * 100
                        
                        formatted_rec = {
                            'id': idx,
                            'timestamp': rec['timestamp'],
                            'action': rec['recommendation'].split()[0],  # BUY or SELL
                            'contract': f"BANKNIFTY {strike}{contract_type}",
                            'strike': strike,
                            'contract_type': contract_type,
                            'entry_price': round(rec['entry_price'], 2),
                            'stop_loss': round(stop_loss, 2),
                            'target_1': round(target1, 2),
                            'target_2': round(target2, 2),
                            'profit_pct_1': round(profit_pct, 2),
                            'profit_pct_2': round(profit_pct * 2, 2),
                            'loss_pct': round(abs(loss_pct), 2),
                            'win_prediction': round(win_prediction, 1),
                            'ml_confidence': round(rec['confidence'] * 100, 1),
                            'consensus': round(rec['consensus'], 1),
                            'algorithm_votes': rec['algorithm_votes'],
                            'signal_strength': round(rec['signal_strength'], 0),
                            'action_level': rec['action'],
                            'rsi': round(rec['technical_setup']['RSI'], 1),
                            'macd': round(rec['technical_setup']['MACD_Histogram'], 2),
                            'trend': rec['technical_setup']['Trend'],
                            'expiry': expiry_display,  # Formatted expiry like "28 APR"
                            'premium_per_lot': round(premium_per_lot, 0),  # Premium for 1 lot (40 contracts)
                            'estimated_premium': round(estimated_premium, 2),  # Premium per contract
                            'mode': current_mode,
                            'vix': round(vix, 1)
                        }
                        
                        formatted_recs.append(formatted_rec)
                    except Exception as e:
                        print(f"[ERROR] Error formatting recommendation: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"[SYSTEM] Formatted {len(formatted_recs)} recommendations for {current_mode}")
                
                # Backtest stats
                try:
                    win_rate, pnl = system.backtest(df, signals, confidences, 
                                                   stop_loss=0.75 if current_mode == 'aggressive' else 1.0, 
                                                   take_profit=6.5 if current_mode == 'aggressive' else 3.5)
                except Exception as e:
                    print(f"[WARNING] Backtest failed: {e}")
                    win_rate = 0
                    pnl = 0
                
                backtest_stats = {
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(pnl, 2),
                    'active_signals': len(np.where(signals != 0)[0]),
                    'total_trades': len(signals),
                    'vix': round(vix, 1),
                    'mode': current_mode
                }
                
                with recommendations_lock:
                    recommendations_cache[current_mode]['recommendations'] = formatted_recs
                    recommendations_cache[current_mode]['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    recommendations_cache[current_mode]['backtest_stats'] = backtest_stats
                    recommendations_cache[current_mode]['vix'] = round(vix, 1)
                    last_update = datetime.now()
                
                print(f"[SUCCESS] Generated {len(formatted_recs)} {current_mode} recommendations (VIX: {vix:.1f})")
            
            except Exception as e:
                print(f"[ERROR] Error generating {current_mode} recommendations: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return []
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@app.route('/')
def index():
    """Serve the UI dashboard"""
    return render_template('trading_dashboard.html')


@app.route('/api/recommendations')
def api_recommendations():
    """Get current recommendations as JSON"""
    mode = request.args.get('mode', 'balanced')  # Can be 'balanced', 'aggressive', or 'both'
    
    with recommendations_lock:
        if mode == 'both':
            return jsonify(recommendations_cache)
        else:
            return jsonify(recommendations_cache.get(mode, recommendations_cache['balanced']))


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Refresh recommendations"""
    mode = request.json.get('mode', 'both') if request.json else 'both'
    generate_recommendations(mode)
    
    with recommendations_lock:
        if mode == 'both':
            return jsonify({
                'status': 'success',
                'balanced': {
                    'timestamp': recommendations_cache['balanced']['timestamp'],
                    'count': len(recommendations_cache['balanced']['recommendations']),
                    'vix': recommendations_cache['balanced']['vix']
                },
                'aggressive': {
                    'timestamp': recommendations_cache['aggressive']['timestamp'],
                    'count': len(recommendations_cache['aggressive']['recommendations']),
                    'vix': recommendations_cache['aggressive']['vix']
                },
                'ultra-selective': {
                    'timestamp': recommendations_cache['ultra-selective']['timestamp'],
                    'count': len(recommendations_cache['ultra-selective']['recommendations']),
                    'vix': recommendations_cache['ultra-selective']['vix']
                }
            })
        else:
            return jsonify({
                'status': 'success',
                'timestamp': recommendations_cache[mode]['timestamp'],
                'count': len(recommendations_cache[mode]['recommendations']),
                'vix': recommendations_cache[mode]['vix']
            })


@app.route('/api/recommendation/<int:rec_id>')
def api_recommendation_detail(rec_id):
    """Get detailed view of a specific recommendation"""
    mode = request.args.get('mode', 'balanced')
    
    with recommendations_lock:
        cache_entry = recommendations_cache.get(mode, recommendations_cache['balanced'])
        for rec in cache_entry.get('recommendations', []):
            if rec['id'] == rec_id:
                return jsonify(rec)
    
    return jsonify({'error': 'Recommendation not found'}), 404


@app.route('/api/stats')
def api_stats():
    """Get system statistics"""
    mode = request.args.get('mode', 'both')  # Can be 'balanced', 'aggressive', or 'both'
    
    with recommendations_lock:
        if mode == 'both':
            return jsonify({
                'balanced': recommendations_cache['balanced']['backtest_stats'],
                'aggressive': recommendations_cache['aggressive']['backtest_stats']
            })
        else:
            return jsonify(recommendations_cache.get(mode, recommendations_cache['balanced'])['backtest_stats'])


if __name__ == '__main__':
    print("[STARTUP] Generating initial recommendations for all modes...")
    generate_recommendations('both')
    print("[STARTUP] Starting Flask server...")
    print("[INFO] Opening browser to: http://localhost:5001")
    
    # Open browser after a short delay to ensure server is ready
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:5001')
    
    from threading import Thread
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
