import shap
import numpy as np
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

def test_shap_pdf_report(model_dir="./modeloteste", test_csv="LBPHaralickTests/features_EarlyFusion_P12_R3.csv"):
    """
    SHAP analysis with COMPLETE FEATURE RANKING GRAPHICS - All 20 Haralick + LBP features
    """
    
    model = joblib.load(os.path.join(model_dir, "stacking_model.pkl"))
    
    # Load ALL features explicitly - drop label
    X_test = pd.read_csv(test_csv, on_bad_lines='skip')
    print(f"Original columns: {list(X_test.columns)}")
    
    # Drop label - use ALL Haralick + LBP features
    feature_columns = [col for col in X_test.columns if col != 'label']
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    X_test = X_test[feature_columns].head(20).values
    feature_names = feature_columns  # Already a list
    
    # SHAP computation
    def model_predict(data):
        return model.predict(data).reshape(-1, 1)
    
    background = shap.kmeans(X_test, 5)
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_test)
    
    # **FEATURE RANKING BY IMPACT (Mean Absolute SHAP)**
    feature_importance = np.abs(shap_values).mean(axis=0)
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_|SHAP|': [float(x) for x in feature_importance],
        'Rank': range(1, len(feature_names) + 1)
    }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)
    
    print("\nFEATURE RANKING BY DECISION IMPACT:")
    print(ranking_df[['Rank', 'Feature', 'Mean_|SHAP|']].to_string(index=False))
    
    print("SHAP analysis complete. Creating PDF report...")
    
    # **7-PAGE PDF WITH COMPLETE RANKING GRAPHICS**
    with PdfPages('SHAP_FEATURE_RANKING_COMPLETE.pdf') as pdf:
        
        # PAGE 1: TITLE + TOP 5 RANKING
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis('off')
        predictions = model.predict(X_test)
        haralick_count = sum(1 for f in feature_names if f.startswith('haralick_'))
        lbp_count = sum(1 for f in feature_names if f.startswith('lbp_'))
        
        text_lines = [
            "SHAP FEATURE IMPACT RANKING",
            "StackingClassifier - Haralick + LBP Analysis", "",
            f"Total Features Ranked: {len(feature_names)}",
            f"Haralick: {haralick_count} | LBP: {lbp_count}", "",
            f"Base Value: {float(explainer.expected_value[0]):.4f}",
            f"Mean Prediction: {float(np.mean(predictions)):.4f}", "",
            "TOP 5 FEATURES BY DECISION IMPACT:"
        ]
        for i in range(5):
            feat = ranking_df.iloc[i]['Feature']
            imp = ranking_df.iloc[i]['Mean_|SHAP|']
            text_lines.append(f"{i+1}. {str(feat):<35} | {imp:.4f}")
        
        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, fontsize=12, va='top', fontfamily='monospace',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue"))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: SHAP SUMMARY PLOT (ALL FEATURES)
        plt.figure(figsize=(11.7, 8.3))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.suptitle('1. SHAP Summary - Feature Impact Distribution', fontsize=16, y=0.98)
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()
        
        # PAGE 3: SHAP BAR PLOT (GLOBAL RANKING)
        plt.figure(figsize=(11.7, 8.3))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.suptitle('2. Global Feature Ranking by SHAP Impact', fontsize=16, y=0.98)
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()
        
        # PAGE 4: COMPLETE HORIZONTAL BAR CHART (TOP 20 RANKED)
        plt.figure(figsize=(11.7, 8.3))
        top20_importance = ranking_df['Mean_|SHAP|'].values
        top20_names = [str(f)[:25] + '...' if len(str(f)) > 25 else str(f) for f in ranking_df['Feature']]
        
        y_pos = np.arange(len(top20_names))
        bars = plt.barh(y_pos, top20_importance, color='skyblue', alpha=0.8, edgecolor='navy')
        plt.yticks(y_pos, top20_names)
        plt.xlabel('Mean Absolute SHAP Value (Decision Impact)')
        plt.title('3. COMPLETE FEATURE RANKING - All 20 Features by Decision Impact', fontsize=14, pad=20)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()
        
        # PAGE 5: TOP 10 vs BOTTOM 10 BAR CHART
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.7, 8.3))
        
        # Top 10
        top10_imp = ranking_df.head(10)['Mean_|SHAP|'].values
        top10_names = [str(f)[:20] for f in ranking_df.head(10)['Feature']]
        y1 = np.arange(10)
        ax1.barh(y1, top10_imp, color='lightgreen', alpha=0.8)
        ax1.set_yticks(y1)
        ax1.set_yticklabels(top10_names, fontsize=9)
        ax1.set_xlabel('SHAP Impact')
        ax1.set_title('TOP 10 Features')
        ax1.grid(axis='x', alpha=0.3)
        
        # Bottom 10  
        bottom10_imp = ranking_df.tail(10)['Mean_|SHAP|'].values
        bottom10_names = [str(f)[:20] for f in ranking_df.tail(10)['Feature']]
        y2 = np.arange(10)
        ax2.barh(y2, bottom10_imp, color='lightcoral', alpha=0.8)
        ax2.set_yticks(y2)
        ax2.set_yticklabels(bottom10_names, fontsize=9)
        ax2.set_xlabel('SHAP Impact')
        ax2.set_title('BOTTOM 10 Features')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('4. Top 10 vs Bottom 10 - Decision Impact Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 6: Haralick vs LBP RANKING
        haralick_features = ranking_df[ranking_df['Feature'].str.startswith('haralick_', na=False)]
        lbp_features = ranking_df[ranking_df['Feature'].str.startswith('lbp_', na=False)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.7, 8.3))
        
        # Haralick ranking
        if len(haralick_features) > 0:
            h_imp = haralick_features['Mean_|SHAP|'].values
            h_names = [str(f) for f in haralick_features['Feature']]
            y_h = np.arange(len(h_names))
            ax1.barh(y_h, h_imp, color='lightblue')
            ax1.set_yticks(y_h)
            ax1.set_yticklabels(h_names, fontsize=10)
            ax1.set_title(f'Haralick Ranking\n({len(haralick_features)} features)')
        
        # LBP ranking  
        if len(lbp_features) > 0:
            l_imp = lbp_features['Mean_|SHAP|'].values
            l_names = [str(f)[:20] + '...' if len(str(f)) > 20 else str(f) for f in lbp_features['Feature']]
            y_l = np.arange(len(l_names))
            ax2.barh(y_l, l_imp, color='lightcoral')
            ax2.set_yticks(y_l)
            ax2.set_yticklabels(l_names, fontsize=10)
            ax2.set_title(f'LBP Ranking\n({len(lbp_features)} features)')
        
        plt.suptitle('5. Haralick vs LBP - Separate Impact Rankings', fontsize=16, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 7: FEATURE RANKING TABLE + STATS
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis('off')
        stats_text = f"""
COMPLETE FEATURE RANKING SUMMARY

Total Features: {len(ranking_df)}
Top Feature: {ranking_df.iloc[0]['Feature']}
Top Haralick: {haralick_features.iloc[0]['Feature'] if len(haralick_features)>0 else 'N/A'}
Top LBP: {lbp_features.iloc[0]['Feature'] if len(lbp_features)>0 else 'N/A'}

RANKING METHODOLOGY:
Mean Absolute SHAP Value across {len(X_test)} samples
Higher = Greater impact on model decisions

SHAP VALUES RANGE:
Max Impact: {ranking_df['Mean_|SHAP|'].max():.4f}
Min Impact: {ranking_df['Mean_|SHAP|'].min():.4f}
Mean Impact: {ranking_df['Mean_|SHAP|'].mean():.4f}
        """
        ax.text(0.05, 0.95, stats_text, fontsize=12, va='top', fontfamily='monospace',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen"))
        plt.title('6. Feature Ranking Methodology & Statistics', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print("SHAP PDF SAVED: SHAP_FEATURE_RANKING_COMPLETE.pdf")
    print("\n🎉 7-PAGE COMPLETE FEATURE RANKING REPORT:")
    print("   1. Title + Top 5")
    print("   2. SHAP Summary Plot") 
    print("   3. Global Bar Ranking")
    print("   4. COMPLETE Horizontal Bar Chart (All 20)")
    print("   5. Top 10 vs Bottom 10")
    print("   6. Haralick vs LBP Rankings")
    print("   7. Ranking Statistics")
    
    return shap_values, explainer, ranking_df

# RUN IT
if __name__ == "__main__":
    result = test_shap_pdf_report()
    shap_values, explainer, ranking_df = result
    
    print("\n📊 TOP 5 FEATURES BY DECISION IMPACT:")
    print(ranking_df.head()[['Rank', 'Feature', 'Mean_|SHAP|']].to_string(index=False))
