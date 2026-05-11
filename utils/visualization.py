import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, save_path='static/feature_importance.png'):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Feature importance plot saved to {save_path}")