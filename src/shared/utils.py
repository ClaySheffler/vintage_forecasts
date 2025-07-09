"""
shared/utils.py
Common utility functions for both wide and long format modules.
"""
import pandas as pd

FICO_BANDS = {
    '600-649': {'min': 600, 'max': 649, 'risk_grade': 5, 'label': 'Very High Risk'},
    '650-699': {'min': 650, 'max': 699, 'risk_grade': 4, 'label': 'High Risk'},
    '700-749': {'min': 700, 'max': 749, 'risk_grade': 3, 'label': 'Medium Risk'},
    '750-799': {'min': 750, 'max': 799, 'risk_grade': 2, 'label': 'Low Risk'},
    '800+':    {'min': 800, 'max': 850, 'risk_grade': 1, 'label': 'Very Low Risk'}
}

def assign_fico_band(fico_score: int) -> str:
    if fico_score >= 800:
        return '800+'
    elif fico_score >= 750:
        return '750-799'
    elif fico_score >= 700:
        return '700-749'
    elif fico_score >= 650:
        return '650-699'
    elif fico_score >= 600:
        return '600-649'
    else:
        return '<600' 