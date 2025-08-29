import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class Visualizer:
    """Creates interactive visualizations for fraud detection analysis."""
    
    def __init__(self):
        self.color_palette = {
            'fraud': '#FF4B4B',
            'legitimate': '#00C851',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }
    
    def plot_fraud_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing fraud vs legitimate transaction distribution."""
        try:
            fraud_counts = df['is_fraud'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraud'],
                values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
                hole=0.4,
                marker_colors=[self.color_palette['legitimate'], self.color_palette['fraud']],
                textinfo='label+percent+value',
                textfont=dict(size=14)
            )])
            
            fig.update_layout(
                title={
                    'text': 'Transaction Distribution: Fraud vs Legitimate',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure if visualization fails
            fig = go.Figure()
            fig.add_annotation(text=f"Visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_amount_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create overlapping histograms for transaction amounts by fraud status."""
        try:
            fraud_amounts = df[df['is_fraud'] == 1]['amount']
            legitimate_amounts = df[df['is_fraud'] == 0]['amount']
            
            fig = go.Figure()
            
            # Add histogram for legitimate transactions
            fig.add_trace(go.Histogram(
                x=legitimate_amounts,
                name='Legitimate',
                opacity=0.7,
                marker_color=self.color_palette['legitimate'],
                nbinsx=50
            ))
            
            # Add histogram for fraud transactions
            fig.add_trace(go.Histogram(
                x=fraud_amounts,
                name='Fraud',
                opacity=0.7,
                marker_color=self.color_palette['fraud'],
                nbinsx=50
            ))
            
            fig.update_layout(
                title={
                    'text': 'Transaction Amount Distribution by Fraud Status',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title='Transaction Amount ($)',
                yaxis_title='Frequency',
                barmode='overlay',
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_fraud_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create a timeline showing fraud transactions over time."""
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("Timestamp column not found")
            
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Daily fraud counts
            daily_fraud = df[df['is_fraud'] == 1].groupby('date').size().reset_index(name='fraud_count')
            daily_total = df.groupby('date').size().reset_index(name='total_count')
            
            daily_stats = pd.merge(daily_fraud, daily_total, on='date', how='right').fillna(0)
            daily_stats['fraud_rate'] = daily_stats['fraud_count'] / daily_stats['total_count'] * 100
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Fraud Count', 'Daily Fraud Rate (%)'),
                vertical_spacing=0.15
            )
            
            # Fraud count plot
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['fraud_count'],
                    mode='lines+markers',
                    name='Fraud Count',
                    line=dict(color=self.color_palette['fraud']),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Fraud rate plot
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['fraud_rate'],
                    mode='lines+markers',
                    name='Fraud Rate (%)',
                    line=dict(color=self.color_palette['primary']),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title={
                    'text': 'Fraud Transaction Timeline',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                height=600,
                showlegend=False,
                font=dict(size=12)
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Timeline visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap for numerical features."""
        try:
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title={
                    'text': 'Feature Correlation Heatmap',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                width=600,
                height=600,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Heatmap visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_fraud_by_merchant(self, df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create a bar chart showing fraud count by merchant category."""
        try:
            if 'merchant_category' not in df.columns:
                raise ValueError("Merchant category column not found")
            
            fraud_by_merchant = df[df['is_fraud'] == 1]['merchant_category'].value_counts().head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=fraud_by_merchant.index,
                    y=fraud_by_merchant.values,
                    marker_color=self.color_palette['fraud'],
                    text=fraud_by_merchant.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': f'Top {top_n} Merchants by Fraud Count',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title='Merchant Category',
                yaxis_title='Fraud Count',
                height=400,
                font=dict(size=12)
            )
            
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Merchant visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_fraud_by_hour(self, df: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing fraud distribution by hour of day."""
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("Timestamp column not found")
            
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            fraud_by_hour = df[df['is_fraud'] == 1]['hour'].value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=fraud_by_hour.index,
                    y=fraud_by_hour.values,
                    marker_color=self.color_palette['fraud'],
                    text=fraud_by_hour.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Fraud Distribution by Hour of Day',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title='Hour of Day',
                yaxis_title='Fraud Count',
                height=400,
                font=dict(size=12)
            )
            
            fig.update_xaxes(dtick=1)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Hourly visualization error: {str(e)}", x=0.5, y=0.5)
            return fig
    
    def plot_amount_vs_fraud_probability(self, df: pd.DataFrame, probabilities: Optional[List] = None) -> go.Figure:
        """Create scatter plot of transaction amount vs fraud probability."""
        try:
            if probabilities is None:
                # Use is_fraud as binary probability
                probabilities = df['is_fraud'].values
            
            colors = ['red' if fraud else 'green' for fraud in df['is_fraud']]
            
            fig = go.Figure(data=go.Scatter(
                x=df['amount'],
                y=probabilities,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6
                ),
                text=[f"Amount: ${amt:.2f}<br>Fraud: {'Yes' if fraud else 'No'}" 
                      for amt, fraud in zip(df['amount'], df['is_fraud'])],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'Transaction Amount vs Fraud Probability',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title='Transaction Amount ($)',
                yaxis_title='Fraud Probability',
                height=400,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Scatter plot error: {str(e)}", x=0.5, y=0.5)
            return fig
