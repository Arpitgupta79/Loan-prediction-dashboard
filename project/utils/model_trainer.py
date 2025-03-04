from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        self.best_model = None
        self.feature_importance = None
        
    def train_models(self, X_train, y_train):
        """Train multiple models and return their cross-validation scores"""
        results = {}
        
        for name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            results[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
            
        return results
    
    def train_best_model(self, X_train, y_train, model_name):
        """Train the selected model on the full training data"""
        self.best_model = self.models[model_name]
        self.best_model.fit(X_train, y_train)
        
        # Get feature importance for random forest and gradient boosting
        if isinstance(self.best_model, (RandomForestClassifier, GradientBoostingClassifier)):
            self.feature_importance = dict(zip(X_train.columns, self.best_model.feature_importances_))
            
        return self.best_model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model using multiple metrics"""
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.best_model.predict(X)
