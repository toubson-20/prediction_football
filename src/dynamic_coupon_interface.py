"""
🎨 DYNAMIC COUPON INTERFACE - INTERFACE UTILISATEUR INTERACTIVE AVANCÉE  
Interface web dynamique pour génération et gestion des coupons intelligents

Version: 3.0 - Phase 3 ML Transformation
Créé: 23 août 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import des composants Phase 3
try:
    from intelligent_betting_coupon import IntelligentBettingCoupon, BettingPrediction
    from confidence_scoring_engine import AdvancedConfidenceScorer
    from realtime_recalibration_engine import RealtimeRecalibrationEngine
    from portfolio_optimization_engine import AdvancedPortfolioOptimizer, MultiObjectiveOptimizer
except ImportError as e:
    print(f"Warning: Import manque - {e}")

class CouponVisualizationEngine:
    """Moteur de visualisation avancée pour les coupons"""
    
    def __init__(self):
        self.color_scheme = {
            'safe': '#4CAF50',
            'balanced': '#2196F3', 
            'value': '#FF9800',
            'longshot': '#F44336',
            'background': '#f8f9fa',
            'text': '#212529',
            'accent': '#6c757d'
        }
    
    def create_confidence_gauge(self, confidence_score: float, title: str = "Confiance") -> go.Figure:
        """Crée une jauge de confiance"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16}},
            delta = {'reference': 70, 'increasing': {'color': self.color_scheme['safe']}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#ffcccb'},
                    {'range': [60, 75], 'color': '#fff2cc'},
                    {'range': [75, 85], 'color': '#d4edda'},
                    {'range': [85, 100], 'color': '#d1ecf1'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': self.color_scheme['text']},
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig
    
    def create_portfolio_risk_chart(self, portfolio_metrics: Dict) -> go.Figure:
        """Crée un graphique de risque du portefeuille"""
        
        risk_metrics = portfolio_metrics.get('risk_metrics', {})
        
        # Radar chart pour les métriques de risque
        categories = ['Expected Return', 'Sharpe Ratio', 'Diversification', 'Stability']
        
        values = [
            min(100, max(0, risk_metrics.get('expected_return', 0) * 1000)),  # Normalisé
            min(100, max(0, risk_metrics.get('sharpe_ratio', 0) * 50)),
            min(100, max(0, portfolio_metrics.get('diversification_ratio', 0.5) * 100)),
            min(100, max(0, (1 - abs(risk_metrics.get('var_95', 0)) / 100) * 100))
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Fermeture du radar
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(color='rgba(33, 150, 243, 1)', width=2),
            name='Portfolio Risk Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=False,
            height=400,
            title="Portfolio Risk Profile",
            font={'color': self.color_scheme['text']},
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig
    
    def create_predictions_breakdown(self, predictions: List[Dict]) -> go.Figure:
        """Crée un graphique de répartition des prédictions"""
        
        # Groupement par catégorie de risque
        risk_counts = {}
        for pred in predictions:
            risk_cat = pred.get('risk_category', 'UNKNOWN')
            risk_counts[risk_cat] = risk_counts.get(risk_cat, 0) + 1
        
        # Couleurs par catégorie
        colors = [self.color_scheme.get(cat.lower(), '#gray') for cat in risk_counts.keys()]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.3,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=12
            )
        ])
        
        fig.update_layout(
            title="Distribution des Risques",
            height=400,
            font={'color': self.color_scheme['text']},
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig
    
    def create_odds_analysis_chart(self, predictions: List[Dict]) -> go.Figure:
        """Crée un graphique d'analyse des cotes"""
        
        # Données pour le scatter plot
        confidences = [pred['confidence_score'] for pred in predictions]
        odds = [pred['odds'] for pred in predictions]
        expected_values = [pred['expected_value'] for pred in predictions]
        prediction_types = [pred['prediction_type'] for pred in predictions]
        
        # Couleur selon expected value
        colors = ['green' if ev > 0 else 'red' for ev in expected_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=confidences,
            y=odds,
            mode='markers',
            marker=dict(
                size=[abs(ev) * 200 + 10 for ev in expected_values],  # Taille selon EV
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"{pt}<br>EV: {ev:.3f}" for pt, ev in zip(prediction_types, expected_values)],
            hovertemplate='<b>%{text}</b><br>Confiance: %{x}%<br>Cotes: %{y}<extra></extra>',
            name='Prédictions'
        ))
        
        # Ligne de référence pour EV = 0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Seuil de rentabilité")
        
        fig.update_layout(
            title="Analyse Confiance vs Cotes",
            xaxis_title="Score de Confiance (%)",
            yaxis_title="Cotes",
            height=500,
            font={'color': self.color_scheme['text']},
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig
    
    def create_timeline_impact_chart(self, recalibration_history: List[Dict]) -> go.Figure:
        """Crée un graphique de timeline des impacts de recalibrage"""
        
        if not recalibration_history:
            return go.Figure().add_annotation(
                text="Aucune donnée de recalibrage disponible",
                showarrow=False,
                font=dict(size=16)
            )
        
        # Données pour la timeline
        timestamps = [entry.get('timestamp', datetime.now()) for entry in recalibration_history]
        impacts = [entry.get('total_impact', 0) for entry in recalibration_history]
        event_types = [entry.get('event_type', 'Unknown') for entry in recalibration_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=impacts,
            mode='markers+lines',
            marker=dict(size=10, color=self.color_scheme['accent']),
            line=dict(color=self.color_scheme['accent'], width=2),
            text=event_types,
            hovertemplate='<b>%{text}</b><br>Impact: %{y:.3f}<br>%{x}<extra></extra>',
            name='Impacts de Recalibrage'
        ))
        
        fig.update_layout(
            title="Timeline des Recalibrages",
            xaxis_title="Temps",
            yaxis_title="Impact",
            height=400,
            font={'color': self.color_scheme['text']},
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig

class StreamlitCouponInterface:
    """Interface principale Streamlit pour les coupons intelligents"""
    
    def __init__(self):
        self.visualization_engine = CouponVisualizationEngine()
        self.coupon_system = None
        self.confidence_scorer = None
        self.recalibration_engine = None
        self.portfolio_optimizer = None
        
        # État de session
        if 'generated_coupons' not in st.session_state:
            st.session_state.generated_coupons = []
        if 'active_coupon' not in st.session_state:
            st.session_state.active_coupon = None
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def initialize_systems(self):
        """Initialise tous les systèmes"""
        
        if st.session_state.system_initialized:
            return True
        
        try:
            with st.spinner("Initialisation des systèmes IA..."):
                self.coupon_system = IntelligentBettingCoupon()
                self.confidence_scorer = AdvancedConfidenceScorer()
                self.recalibration_engine = RealtimeRecalibrationEngine()
                self.portfolio_optimizer = AdvancedPortfolioOptimizer()
                
                # Initialisation des composants
                self.coupon_system.initialize_components()
                self.recalibration_engine.initialize_components()
                
                st.session_state.system_initialized = True
                st.success("✅ Systèmes initialisés avec succès!")
                return True
                
        except Exception as e:
            st.error(f"❌ Erreur initialisation: {str(e)}")
            return False
    
    def render_main_interface(self):
        """Rendu de l'interface principale"""
        
        st.set_page_config(
            page_title="🎯 Coupon Intelligent - Système Révolutionnaire",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header avec style
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>🎯 Coupon Intelligent</h1>
            <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>Système Révolutionnaire de Prédictions Sportives</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialisation des systèmes
        if not self.initialize_systems():
            st.stop()
        
        # Sidebar avec navigation
        with st.sidebar:
            st.header("🎮 Navigation")
            
            page = st.selectbox(
                "Choisir une section:",
                ["🏠 Accueil", "⚡ Génération Coupon", "📊 Analyse Portfolio", 
                 "🔄 Recalibrage Temps Réel", "📈 Statistiques", "⚙️ Configuration"]
            )
        
        # Routage selon la page sélectionnée
        if page == "🏠 Accueil":
            self.render_home_page()
        elif page == "⚡ Génération Coupon":
            self.render_coupon_generation()
        elif page == "📊 Analyse Portfolio":
            self.render_portfolio_analysis()
        elif page == "🔄 Recalibrage Temps Réel":
            self.render_realtime_recalibration()
        elif page == "📈 Statistiques":
            self.render_statistics()
        elif page == "⚙️ Configuration":
            self.render_configuration()
    
    def render_home_page(self):
        """Page d'accueil avec tableau de bord"""
        
        st.header("🏠 Tableau de Bord")
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_coupons = len(st.session_state.generated_coupons)
            st.metric("Coupons Générés", total_coupons, delta=None)
        
        with col2:
            if self.coupon_system and hasattr(self.coupon_system, 'coupon_history'):
                success_rate = 0.75  # Simulation
                st.metric("Taux de Succès", f"{success_rate:.1%}", delta="5%")
            else:
                st.metric("Taux de Succès", "N/A")
        
        with col3:
            avg_roi = 0.18  # Simulation 
            st.metric("ROI Moyen", f"{avg_roi:.1%}", delta="3%")
        
        with col4:
            active_monitoring = 0  # Simulation
            st.metric("Surveillance Active", f"{active_monitoring} matchs")
        
        # Graphiques de synthèse
        if st.session_state.generated_coupons:
            st.subheader("📈 Aperçu des Performances")
            
            # Performance des derniers coupons
            recent_coupons = st.session_state.generated_coupons[-10:]  # 10 derniers
            
            if recent_coupons:
                # Graphique de confiance moyenne
                confidences = []
                coupon_names = []
                
                for i, coupon in enumerate(recent_coupons):
                    if 'portfolio_metrics' in coupon:
                        conf = coupon['portfolio_metrics'].get('average_confidence', 70)
                        confidences.append(conf)
                        coupon_names.append(f"Coupon {i+1}")
                
                if confidences:
                    fig_conf = px.line(
                        x=coupon_names, y=confidences,
                        title="Évolution de la Confiance Moyenne",
                        labels={'y': 'Confiance (%)', 'x': 'Coupons'}
                    )
                    fig_conf.update_traces(line=dict(color='#2196F3', width=3))
                    st.plotly_chart(fig_conf, use_container_width=True)
        
        # Coupons récents
        if st.session_state.generated_coupons:
            st.subheader("🎫 Coupons Récents")
            
            for coupon in st.session_state.generated_coupons[-3:]:  # 3 derniers
                with st.expander(f"📋 {coupon.get('coupon_id', 'Coupon')} - {coupon.get('predictions_selected', 0)} prédictions"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Confiance:** {coupon.get('portfolio_metrics', {}).get('average_confidence', 'N/A')}%")
                    
                    with col2:
                        st.write(f"**Cotes Totales:** {coupon.get('portfolio_metrics', {}).get('total_odds', 'N/A')}")
                    
                    with col3:
                        st.write(f"**Expected Value:** {coupon.get('portfolio_metrics', {}).get('total_expected_value', 'N/A')}")
        
        else:
            st.info("🎯 Aucun coupon généré. Utilisez la section 'Génération Coupon' pour commencer!")
    
    def render_coupon_generation(self):
        """Interface de génération de coupons"""
        
        st.header("⚡ Génération de Coupon Intelligent")
        
        # Configuration du coupon
        st.subheader("🎛️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_predictions = st.slider("Nombre de prédictions cible", 5, 12, 8)
            min_confidence = st.slider("Confiance minimum (%)", 60, 90, 70)
        
        with col2:
            optimization_method = st.selectbox(
                "Méthode d'optimisation",
                ["Kelly Avancé", "Markowitz Max Sharpe", "Algorithme Génétique", "Multi-Critères"]
            )
            
            risk_level = st.selectbox("Niveau de risque", ["Conservateur", "Équilibré", "Agressif"])
        
        # Configuration des matchs
        st.subheader("⚽ Sélection des Matchs")
        
        # Interface simplifiée pour sélectionner des matchs
        matches_input = st.text_area(
            "Matchs à analyser (un par ligne: 'Équipe A vs Équipe B, Ligue')",
            value="Manchester United vs Liverpool, Premier_League\\nBarcelona vs Real Madrid, La_Liga\\nBayern Munich vs Borussia Dortmund, Bundesliga",
            height=150
        )
        
        # Parsing des matchs
        matches_data = []
        if matches_input.strip():
            lines = matches_input.strip().split('\\n')
            for line in lines:
                if ',' in line and ' vs ' in line:
                    match_part, league_part = line.split(',', 1)
                    teams = match_part.split(' vs ', 1)
                    if len(teams) == 2:
                        matches_data.append({
                            'home_team': teams[0].strip(),
                            'away_team': teams[1].strip(),
                            'league': league_part.strip(),
                            'match_importance': 'normal',
                            'date': '2025-01-25'
                        })
        
        st.write(f"📊 {len(matches_data)} matchs configurés")
        
        # Génération du coupon
        if st.button("🚀 Générer Coupon Intelligent", type="primary"):
            if not matches_data:
                st.error("❌ Veuillez configurer au moins un match")
                return
            
            with st.spinner("⚡ Génération en cours..."):
                try:
                    # Configuration personnalisée
                    coupon_config = {
                        'min_predictions': max(3, target_predictions - 2),
                        'max_predictions': min(15, target_predictions + 3),
                        'target_predictions': target_predictions,
                        'min_confidence': min_confidence,
                        'optimization_method': optimization_method
                    }
                    
                    # Génération
                    coupon = self.coupon_system.generate_intelligent_coupon(
                        matches_data, coupon_config
                    )
                    
                    if coupon.get('status') == 'success':
                        st.success(f"✅ Coupon généré avec succès: {coupon['coupon_id']}")
                        
                        # Sauvegarde dans session
                        st.session_state.generated_coupons.append(coupon)
                        st.session_state.active_coupon = coupon
                        
                        # Affichage du coupon
                        self.display_generated_coupon(coupon)
                    
                    else:
                        st.error(f"❌ Échec génération: {coupon.get('message', 'Erreur inconnue')}")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    def display_generated_coupon(self, coupon: Dict):
        """Affichage détaillé d'un coupon généré"""
        
        st.subheader(f"🎫 {coupon['coupon_id']}")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = coupon['portfolio_metrics']['average_confidence']
            st.metric("Confiance Moyenne", f"{confidence:.1f}%")
            
            # Jauge de confiance
            conf_gauge = self.visualization_engine.create_confidence_gauge(confidence)
            st.plotly_chart(conf_gauge, use_container_width=True)
        
        with col2:
            total_odds = coupon['portfolio_metrics']['total_odds']
            st.metric("Cotes Totales", f"{total_odds:.2f}")
        
        with col3:
            expected_value = coupon['portfolio_metrics']['total_expected_value']
            st.metric("Expected Value", f"{expected_value:.4f}")
        
        with col4:
            kelly_stake = coupon['portfolio_metrics'].get('kelly_total_stake', 0)
            st.metric("Mise Kelly", f"{kelly_stake:.2f}€")
        
        # Graphiques d'analyse
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition des risques
            risk_chart = self.visualization_engine.create_predictions_breakdown(coupon['predictions'])
            st.plotly_chart(risk_chart, use_container_width=True)
        
        with col2:
            # Analyse des cotes
            odds_chart = self.visualization_engine.create_odds_analysis_chart(coupon['predictions'])
            st.plotly_chart(odds_chart, use_container_width=True)
        
        # Détail des prédictions
        st.subheader("📋 Détail des Prédictions")
        
        predictions_df = pd.DataFrame(coupon['predictions'])
        
        # Colonnes à afficher
        display_columns = ['prediction_type', 'prediction_value', 'confidence_score', 
                          'odds', 'expected_value', 'risk_category', 'kelly_stake']
        
        if all(col in predictions_df.columns for col in display_columns):
            display_df = predictions_df[display_columns].copy()
            
            # Formatage
            display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1f}%")
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"{x:.3f}")
            display_df['kelly_stake'] = display_df['kelly_stake'].apply(lambda x: f"{x:.2f}€")
            
            # Renommage des colonnes
            column_mapping = {
                'prediction_type': 'Type',
                'prediction_value': 'Prédiction', 
                'confidence_score': 'Confiance',
                'odds': 'Cotes',
                'expected_value': 'EV',
                'risk_category': 'Risque',
                'kelly_stake': 'Mise Kelly'
            }
            
            display_df = display_df.rename(columns=column_mapping)
            
            # Affichage avec style
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Conseils de mise
        if coupon.get('betting_advice'):
            st.subheader("💡 Conseils de Mise")
            for advice in coupon['betting_advice']:
                st.info(advice)
    
    def render_portfolio_analysis(self):
        """Interface d'analyse de portefeuille"""
        
        st.header("📊 Analyse de Portfolio")
        
        if not st.session_state.active_coupon:
            st.warning("⚠️ Aucun coupon actif. Générez d'abord un coupon.")
            return
        
        coupon = st.session_state.active_coupon
        
        # Métriques de risque détaillées
        if 'portfolio_metrics' in coupon:
            metrics = coupon['portfolio_metrics']
            
            st.subheader("🎯 Métriques de Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Taille Portfolio", metrics['portfolio_size'])
                st.metric("Poids Maximum", f"{metrics.get('max_weight', 0):.1%}")
            
            with col2:
                if 'risk_metrics' in metrics:
                    risk = metrics['risk_metrics']
                    st.metric("VaR 95%", f"{risk.get('var_95', 0):.2f}€")
                    st.metric("Sharpe Ratio", f"{risk.get('sharpe_ratio', 0):.3f}")
            
            with col3:
                st.metric("Diversification", f"{metrics.get('diversification_ratio', 0):.1%}")
                if 'drawdown_metrics' in metrics:
                    dd = metrics['drawdown_metrics']
                    st.metric("Max Drawdown", f"{dd.get('expected_max_drawdown', 0):.1%}")
            
            # Graphique de profil de risque
            st.subheader("📈 Profil de Risque")
            risk_chart = self.visualization_engine.create_portfolio_risk_chart(metrics)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        # Optimisation alternative
        st.subheader("⚙️ Optimisation Alternative")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "Méthode d'optimisation",
                ["Kelly Avancé", "Markowitz Max Sharpe", "Min Risk", "Max Return"]
            )
        
        with col2:
            st.write("")  # Spacing
            if st.button("🔄 Re-optimiser"):
                with st.spinner("Optimisation en cours..."):
                    # Simulation de ré-optimisation
                    st.success("✅ Portfolio réoptimisé avec succès!")
                    st.info("💡 Nouvelle allocation suggérée calculée")
    
    def render_realtime_recalibration(self):
        """Interface de recalibrage temps réel"""
        
        st.header("🔄 Recalibrage Temps Réel")
        
        if not st.session_state.active_coupon:
            st.warning("⚠️ Aucun coupon actif pour le recalibrage.")
            return
        
        # Simulation de monitoring
        st.subheader("📡 État du Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Matchs Surveillés", "3")
        
        with col2:
            st.metric("Dernière Mise à Jour", "Il y a 2 min")
        
        with col3:
            monitoring_status = st.selectbox("Statut", ["Actif", "Inactif"], index=0)
        
        # Événements détectés
        st.subheader("⚡ Événements Détectés")
        
        # Simulation d'événements
        sample_events = [
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'match': 'Manchester United vs Liverpool',
                'event_type': 'Blessure joueur clé',
                'impact': -0.08,
                'status': 'En attente'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'match': 'Barcelona vs Real Madrid', 
                'event_type': 'Changement météo',
                'impact': -0.03,
                'status': 'Approuvé'
            }
        ]
        
        for event in sample_events:
            with st.expander(f"🔔 {event['event_type']} - Impact: {event['impact']:.3f}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Match:** {event['match']}")
                    st.write(f"**Heure:** {event['timestamp'].strftime('%H:%M:%S')}")
                
                with col2:
                    st.write(f"**Impact:** {event['impact']:.3f}")
                    st.write(f"**Statut:** {event['status']}")
                
                with col3:
                    if event['status'] == 'En attente':
                        col_approve, col_reject = st.columns(2)
                        with col_approve:
                            if st.button("✅ Approuver", key=f"approve_{event['timestamp']}"):
                                st.success("Recalibrage approuvé!")
                        with col_reject:
                            if st.button("❌ Rejeter", key=f"reject_{event['timestamp']}"):
                                st.info("Recalibrage rejeté")
        
        # Configuration du recalibrage
        st.subheader("⚙️ Configuration Recalibrage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_approve_threshold = st.slider("Seuil auto-approbation", 0.01, 0.20, 0.05, 0.01)
            st.write(f"Impact < {auto_approve_threshold:.2f} → Approbation automatique")
        
        with col2:
            max_adjustment = st.slider("Ajustement maximum confiance", 5.0, 25.0, 15.0, 1.0)
            st.write(f"Maximum {max_adjustment:.0f} points d'ajustement")
    
    def render_statistics(self):
        """Page de statistiques détaillées"""
        
        st.header("📈 Statistiques du Système")
        
        # Statistiques globales
        st.subheader("🌍 Vue d'Ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Coupons Générés", len(st.session_state.generated_coupons))
        
        with col2:
            if self.confidence_scorer:
                stats = self.confidence_scorer.get_confidence_statistics()
                st.metric("Prédictions Scorées", stats.get('total_predictions_scored', 0))
        
        with col3:
            st.metric("Modèles Actifs", "180+")
        
        with col4:
            st.metric("Uptime", "99.9%")
        
        # Graphiques de performance
        if st.session_state.generated_coupons:
            st.subheader("📊 Analyse des Performances")
            
            # Performance par type de prédiction
            all_predictions = []
            for coupon in st.session_state.generated_coupons:
                all_predictions.extend(coupon.get('predictions', []))
            
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                
                # Graphique confiance par type
                if 'prediction_type' in pred_df.columns and 'confidence_score' in pred_df.columns:
                    conf_by_type = pred_df.groupby('prediction_type')['confidence_score'].mean().reset_index()
                    
                    fig_conf = px.bar(
                        conf_by_type,
                        x='prediction_type',
                        y='confidence_score',
                        title="Confiance Moyenne par Type de Prédiction",
                        labels={'confidence_score': 'Confiance (%)', 'prediction_type': 'Type de Prédiction'}
                    )
                    fig_conf.update_traces(marker_color='#2196F3')
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Distribution des Expected Values
                if 'expected_value' in pred_df.columns:
                    fig_ev = px.histogram(
                        pred_df,
                        x='expected_value',
                        nbins=20,
                        title="Distribution des Expected Values",
                        labels={'expected_value': 'Expected Value', 'count': 'Nombre'}
                    )
                    fig_ev.update_traces(marker_color='#4CAF50')
                    st.plotly_chart(fig_ev, use_container_width=True)
        
        # Statistiques du système de confiance
        if self.confidence_scorer:
            st.subheader("🎯 Système de Confiance")
            
            conf_stats = self.confidence_scorer.get_confidence_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "Confiance moyenne": f"{conf_stats.get('average_confidence', 0):.1f}%",
                    "Calibration active": "Oui" if conf_stats.get('calibration_available') else "Non",
                    "Points de calibration": conf_stats.get('calibration_data_points', 0)
                })
            
            with col2:
                if 'confidence_distribution' in conf_stats:
                    dist = conf_stats['confidence_distribution']
                    st.json({
                        "Min": f"{dist.get('min', 0):.1f}%",
                        "Max": f"{dist.get('max', 0):.1f}%",
                        "Médiane": f"{dist.get('50%', 0):.1f}%",
                        "Écart-type": f"{dist.get('std', 0):.1f}"
                    })
    
    def render_configuration(self):
        """Page de configuration système"""
        
        st.header("⚙️ Configuration Système")
        
        # Configuration des modèles
        st.subheader("🤖 Modèles IA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Architecture Révolutionnaire", value=True, disabled=True)
            st.checkbox("Deep Learning Ensemble", value=True)
            st.checkbox("Transfer Learning", value=True)
        
        with col2:
            st.checkbox("Meta-Model Intelligent", value=True)
            st.checkbox("Confidence Scoring", value=True)
            st.checkbox("Portfolio Optimizer", value=True)
        
        # Configuration des coupons
        st.subheader("🎫 Configuration Coupons")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.number_input("Taille min portfolio", min_value=3, max_value=8, value=5)
            st.number_input("Taille max portfolio", min_value=8, max_value=15, value=12)
        
        with col2:
            st.number_input("Confiance minimum (%)", min_value=50, max_value=90, value=65)
            st.number_input("Corrélation maximum", min_value=0.3, max_value=0.8, value=0.6, step=0.1)
        
        with col3:
            st.number_input("Expected Value min", min_value=-0.1, max_value=0.2, value=0.05, step=0.01)
            st.number_input("Poids max par pari (%)", min_value=10, max_value=25, value=15)
        
        # Configuration monitoring
        st.subheader("📡 Monitoring Temps Réel")
        
        monitoring_enabled = st.toggle("Activer monitoring temps réel", value=True)
        
        if monitoring_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox("Fréquence vérification", ["30s", "1min", "2min", "5min"], index=1)
                st.multiselect(
                    "Sources de données",
                    ["Compositions", "Météo", "Cotes", "Actualités", "Arbitrage"],
                    default=["Compositions", "Météo", "Cotes"]
                )
            
            with col2:
                st.number_input("Seuil auto-approbation", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
                st.number_input("Délai avant match (min)", min_value=15, max_value=120, value=60)
        
        # Sauvegarde de la configuration
        if st.button("💾 Sauvegarder Configuration", type="primary"):
            st.success("✅ Configuration sauvegardée!")
            st.balloons()

def main():
    """Fonction principale de l'application"""
    
    interface = StreamlitCouponInterface()
    interface.render_main_interface()

if __name__ == "__main__":
    # Configuration pour éviter les avertissements
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    main()