import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def plot_feature_importance(feature_importance):
        """Plot feature importance"""
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        df = df.sort_values('Importance', ascending=True)

        fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Blues')

        fig.update_layout(height=500)
        return fig

    @staticmethod
    def plot_correlation_matrix(df):
        """Plot correlation matrix for numeric columns only"""
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        fig = px.imshow(corr,
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r')

        fig.update_layout(title='Feature Correlation Matrix')
        return fig

    @staticmethod
    def plot_metrics_comparison(cv_results):
        """Plot model comparison metrics"""
        models = list(cv_results.keys())
        scores = [result['mean_cv_score'] for result in cv_results.values()]
        errors = [result['std_cv_score'] for result in cv_results.values()]

        fig = go.Figure(data=[
            go.Bar(name='Score',
                  x=models,
                  y=scores,
                  error_y=dict(type='data', array=errors))
        ])

        fig.update_layout(title='Model Performance Comparison',
                         xaxis_title='Model',
                         yaxis_title='Cross-validation Score')
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Rejected', 'Approved'],
            y=['Rejected', 'Approved'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False))

        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500,
            height=500
        )
        return fig