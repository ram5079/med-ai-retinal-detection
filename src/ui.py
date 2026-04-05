import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def get_custom_css():
    return """
    <style>
    /* Main body background to soft medical gray/blue */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide top header of Streamlit */
    header {visibility: hidden;}
    
    /* Modern Card Layouts */
    .stCard {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
        margin-bottom: 20px;
    }
    .stCard:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Typography */
    .dashboard-title {
        color: #1e293b;
        font-weight: 800;
        letter-spacing: -0.5px;
        padding-top: 10px;
        margin-bottom: 5px;
    }
    .dashboard-subtitle {
        color: #64748b;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Animated Fade In */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    .medical-insight-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(to right, #eff6ff, #ffffff);
        border-left: 5px solid #3b82f6;
        color: #1e293b;
        margin-top: 15px;
    }
    </style>
    """

def render_gauge_chart(value, title, color_str, max_val=100):
    """
    Returns a Plotly Gauge figure.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color_str},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val*0.4], 'color': '#f1f5f9'},
                {'range': [max_val*0.4, max_val*0.8], 'color': '#e2e8f0'},
                {'range': [max_val*0.8, max_val], 'color': '#cbd5e1'}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_probabilities_chart(probs, class_names):
    """
    Returns an animated/styled Plotly Bar chart for classification probabilities.
    """
    df = pd.DataFrame({'Stage': class_names, 'Probability': probs * 100})
    
    # Color map
    color_discrete_map = {
        "No DR": "#22c55e",
        "Mild": "#84cc16",
        "Moderate": "#eab308",
        "Severe": "#f97316",
        "Proliferative DR": "#ef4444"
    }
    
    fig = px.bar(df, x='Probability', y='Stage', orientation='h',
                 color='Stage', color_discrete_map=color_discrete_map,
                 text='Probability')
                 
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    # Lock axes
    fig.update_xaxes(range=[0, 110])
    return fig
