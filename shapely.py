import shap


def shap_plots(model,x_test):
    explainer_tree = shap.TreeExplainer(model)
    shap_values = explainer_tree.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)

