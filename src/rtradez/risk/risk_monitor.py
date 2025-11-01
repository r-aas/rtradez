"""
Real-time risk monitoring and alerting system.

Provides continuous monitoring of portfolio risk metrics with configurable
alerts and automatic risk mitigation actions for options trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import threading
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Risk alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of risk alerts."""
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    POSITION_SIZE = "position_size"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MARGIN_CALL = "margin_call"
    LIQUIDITY = "liquidity"
    SYSTEM = "system"

@dataclass
class RiskAlert:
    """Risk alert message."""
    alert_type: AlertType
    level: AlertLevel
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    strategy_affected: Optional[str] = None
    recommended_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskThreshold:
    """Risk monitoring threshold configuration."""
    metric_name: str
    threshold_value: float
    alert_level: AlertLevel
    enabled: bool = True
    lookback_period: Optional[int] = None  # Minutes for rolling calculations
    cooldown_period: int = 300  # Seconds between repeated alerts

class RealTimeRiskMonitor:
    """Real-time portfolio risk monitoring system."""
    
    def __init__(self, update_interval: int = 60,
                 max_alert_history: int = 1000,
                 enable_auto_actions: bool = False):
        """
        Initialize real-time risk monitor.
        
        Args:
            update_interval: Update interval in seconds
            max_alert_history: Maximum alerts to keep in history
            enable_auto_actions: Enable automatic risk mitigation actions
        """
        self.update_interval = update_interval
        self.max_alert_history = max_alert_history
        self.enable_auto_actions = enable_auto_actions
        
        # Risk thresholds
        self.thresholds: Dict[str, RiskThreshold] = {}
        self.custom_monitors: Dict[str, Callable] = {}
        
        # Alert management
        self.alert_history: deque = deque(maxlen=max_alert_history)
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Real-time data storage
        self.current_portfolio_data: Dict[str, Any] = {}
        self.price_history: Dict[str, deque] = {}
        self.pnl_history: deque = deque(maxlen=10000)  # Keep 10000 data points
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        self.auto_action_callbacks: List[Callable[[RiskAlert], None]] = []
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
    def _initialize_default_thresholds(self):
        """Initialize default risk thresholds."""
        # Portfolio drawdown thresholds
        self.add_threshold(RiskThreshold(
            "max_drawdown", 0.05, AlertLevel.WARNING, lookback_period=60
        ))
        self.add_threshold(RiskThreshold(
            "max_drawdown", 0.10, AlertLevel.CRITICAL, lookback_period=60
        ))
        self.add_threshold(RiskThreshold(
            "max_drawdown", 0.15, AlertLevel.EMERGENCY, lookback_period=60
        ))
        
        # VaR thresholds
        self.add_threshold(RiskThreshold(
            "daily_var", 0.03, AlertLevel.WARNING
        ))
        self.add_threshold(RiskThreshold(
            "daily_var", 0.05, AlertLevel.CRITICAL
        ))
        
        # Position concentration
        self.add_threshold(RiskThreshold(
            "max_position_pct", 0.20, AlertLevel.WARNING
        ))
        self.add_threshold(RiskThreshold(
            "max_position_pct", 0.30, AlertLevel.CRITICAL
        ))
        
        # Volatility spike detection
        self.add_threshold(RiskThreshold(
            "volatility_spike", 2.0, AlertLevel.WARNING, lookback_period=30
        ))
        
    def add_threshold(self, threshold: RiskThreshold):
        """Add a risk monitoring threshold."""
        key = f"{threshold.metric_name}_{threshold.alert_level.value}"
        self.thresholds[key] = threshold
        logger.info(f"Added risk threshold: {threshold.metric_name} @ {threshold.threshold_value}")
        
    def remove_threshold(self, metric_name: str, alert_level: AlertLevel):
        """Remove a risk threshold."""
        key = f"{metric_name}_{alert_level.value}"
        if key in self.thresholds:
            del self.thresholds[key]
            logger.info(f"Removed risk threshold: {metric_name}")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def add_auto_action_callback(self, callback: Callable[[RiskAlert], None]):
        """Add callback function for automatic actions."""
        self.auto_action_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start real-time risk monitoring."""
        if self.is_monitoring:
            logger.warning("Risk monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started risk monitoring with {self.update_interval}s intervals")
        
    def stop_monitoring(self):
        """Stop real-time risk monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped risk monitoring")
        
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                self._update_risk_metrics()
                self._check_all_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(self.update_interval)
                
    def update_portfolio_data(self, portfolio_data: Dict[str, Any]):
        """Update current portfolio data for monitoring."""
        self.current_portfolio_data = portfolio_data.copy()
        
        # Update PnL history
        if 'total_pnl' in portfolio_data:
            self.pnl_history.append({
                'timestamp': datetime.now(),
                'pnl': portfolio_data['total_pnl']
            })
        
        # Update price history for volatility monitoring
        if 'positions' in portfolio_data:
            for strategy, position_data in portfolio_data['positions'].items():
                if 'market_value' in position_data:
                    if strategy not in self.price_history:
                        self.price_history[strategy] = deque(maxlen=1000)
                    
                    self.price_history[strategy].append({
                        'timestamp': datetime.now(),
                        'value': position_data['market_value']
                    })
    
    def _update_risk_metrics(self):
        """Update calculated risk metrics."""
        if not self.current_portfolio_data:
            return
        
        # Calculate current drawdown
        if len(self.pnl_history) > 1:
            pnl_values = [p['pnl'] for p in self.pnl_history]
            pnl_series = pd.Series(pnl_values)
            peak = pnl_series.expanding().max()
            current_drawdown = abs((pnl_series.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1])
            self.current_portfolio_data['current_drawdown'] = current_drawdown
        
        # Calculate position concentration
        if 'positions' in self.current_portfolio_data:
            position_values = [abs(pos.get('market_value', 0)) 
                             for pos in self.current_portfolio_data['positions'].values()]
            if position_values:
                total_value = sum(position_values)
                max_position_pct = max(position_values) / total_value if total_value > 0 else 0
                self.current_portfolio_data['max_position_pct'] = max_position_pct
        
        # Calculate volatility metrics
        self._calculate_volatility_metrics()
        
    def _calculate_volatility_metrics(self):
        """Calculate volatility-based risk metrics."""
        for strategy, price_data in self.price_history.items():
            if len(price_data) < 10:
                continue
            
            # Calculate rolling volatility
            prices = [p['value'] for p in price_data]
            price_series = pd.Series(prices)
            returns = price_series.pct_change().dropna()
            
            if len(returns) > 5:
                current_vol = returns.rolling(min(len(returns), 20)).std().iloc[-1]
                long_term_vol = returns.std()
                
                if long_term_vol > 0:
                    vol_spike = current_vol / long_term_vol
                    self.current_portfolio_data[f'{strategy}_volatility_spike'] = vol_spike
    
    def _check_all_thresholds(self):
        """Check all configured risk thresholds."""
        current_time = datetime.now()
        new_alerts = []
        
        for threshold_key, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
            
            # Check cooldown period
            if (threshold_key in self.last_alert_times and 
                (current_time - self.last_alert_times[threshold_key]).total_seconds() < threshold.cooldown_period):
                continue
            
            # Get current value for this metric
            current_value = self._get_metric_value(threshold.metric_name, threshold.lookback_period)
            
            if current_value is None:
                continue
            
            # Check if threshold is breached
            if current_value > threshold.threshold_value:
                alert = self._create_alert(threshold, current_value, current_time)
                new_alerts.append(alert)
                self.last_alert_times[threshold_key] = current_time
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
    
    def _get_metric_value(self, metric_name: str, lookback_period: Optional[int]) -> Optional[float]:
        """Get current value for a specific metric."""
        if metric_name in self.current_portfolio_data:
            return self.current_portfolio_data[metric_name]
        
        # Handle time-series metrics
        if metric_name == "max_drawdown" and len(self.pnl_history) > 1:
            return self.current_portfolio_data.get('current_drawdown', 0)
        
        # Handle volatility spike metrics
        if metric_name == "volatility_spike":
            vol_spikes = [v for k, v in self.current_portfolio_data.items() 
                         if k.endswith('_volatility_spike')]
            return max(vol_spikes) if vol_spikes else 0
        
        return None
    
    def _create_alert(self, threshold: RiskThreshold, current_value: float, 
                     timestamp: datetime) -> RiskAlert:
        """Create a risk alert."""
        # Determine alert type based on metric name
        if "drawdown" in threshold.metric_name:
            alert_type = AlertType.DRAWDOWN
        elif "var" in threshold.metric_name:
            alert_type = AlertType.VAR_BREACH
        elif "position" in threshold.metric_name:
            alert_type = AlertType.POSITION_SIZE
        elif "volatility" in threshold.metric_name:
            alert_type = AlertType.VOLATILITY
        else:
            alert_type = AlertType.SYSTEM
        
        # Generate message and recommended action
        message = f"{threshold.metric_name} breach: {current_value:.3f} > {threshold.threshold_value:.3f}"
        
        recommended_actions = {
            AlertType.DRAWDOWN: "Consider reducing position sizes or closing losing positions",
            AlertType.VAR_BREACH: "Review portfolio risk and consider hedging",
            AlertType.POSITION_SIZE: "Reduce position concentration",
            AlertType.VOLATILITY: "Monitor for market stress and consider hedging",
            AlertType.SYSTEM: "Review system parameters"
        }
        
        return RiskAlert(
            alert_type=alert_type,
            level=threshold.alert_level,
            message=message,
            current_value=current_value,
            threshold=threshold.threshold_value,
            timestamp=timestamp,
            recommended_action=recommended_actions.get(alert_type, "Review and take appropriate action"),
            metadata={'metric_name': threshold.metric_name}
        )
    
    def _process_alert(self, alert: RiskAlert):
        """Process a new risk alert."""
        # Add to history and active alerts
        self.alert_history.append(alert)
        alert_key = f"{alert.alert_type.value}_{alert.level.value}"
        self.active_alerts[alert_key] = alert
        
        # Log the alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.EMERGENCY: logger.critical
        }
        log_level[alert.level](f"RISK ALERT: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Execute automatic actions if enabled
        if self.enable_auto_actions and alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            for callback in self.auto_action_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in auto-action callback: {e}")
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[RiskAlert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def clear_alert(self, alert_type: AlertType, level: AlertLevel):
        """Manually clear an active alert."""
        alert_key = f"{alert_type.value}_{level.value}"
        if alert_key in self.active_alerts:
            del self.active_alerts[alert_key]
            logger.info(f"Cleared alert: {alert_key}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data."""
        current_time = datetime.now()
        
        return {
            'monitoring_status': self.is_monitoring,
            'last_update': current_time.isoformat(),
            'active_alerts': {
                'total': len(self.active_alerts),
                'critical': len([a for a in self.active_alerts.values() 
                               if a.level == AlertLevel.CRITICAL]),
                'emergency': len([a for a in self.active_alerts.values() 
                                if a.level == AlertLevel.EMERGENCY]),
                'alerts': [a.__dict__ for a in self.get_active_alerts()]
            },
            'recent_alerts_24h': len(self.get_alert_history(24)),
            'portfolio_metrics': self.current_portfolio_data,
            'threshold_status': {
                key: {
                    'enabled': threshold.enabled,
                    'threshold': threshold.threshold_value,
                    'current_value': self._get_metric_value(threshold.metric_name, threshold.lookback_period)
                }
                for key, threshold in self.thresholds.items()
            }
        }

# Utility functions for common monitoring scenarios
def create_basic_risk_monitor(portfolio_value: float) -> RealTimeRiskMonitor:
    """Create a basic risk monitor with sensible defaults."""
    monitor = RealTimeRiskMonitor(update_interval=30)
    
    # Add portfolio-specific thresholds
    monitor.add_threshold(RiskThreshold(
        "daily_pnl_pct", 0.02, AlertLevel.WARNING  # 2% daily loss
    ))
    monitor.add_threshold(RiskThreshold(
        "daily_pnl_pct", 0.05, AlertLevel.CRITICAL  # 5% daily loss
    ))
    
    return monitor

def setup_alert_notifications(monitor: RealTimeRiskMonitor, 
                             email_alerts: bool = False,
                             slack_webhook: Optional[str] = None):
    """Setup common alert notification methods."""
    
    def console_alert_handler(alert: RiskAlert):
        """Simple console alert handler."""
        print(f"\n🚨 {alert.level.value.upper()} ALERT: {alert.message}")
        print(f"   Recommended Action: {alert.recommended_action}")
        print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    monitor.add_alert_callback(console_alert_handler)
    
    # Add email/slack handlers if configured
    # Implementation would depend on specific email/messaging services