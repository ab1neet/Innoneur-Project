import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import svds
import joblib
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

filepath = r'\2017_Financial_Data.csv'

class InvestmentRecommendationSystem:
    def __init__(self, filepath):
        self.data = self.load_dataset(filepath)
        self.user_data = None  # This would be loaded from a separate user database
        self.model = None
        self.pca = None
        self.scaler = None
        self.kmeans = None

    def load_dataset(self, filepath):
        df = pd.read_csv(filepath)
        df.set_index(df.columns[0], inplace=True)  # Set first column as index (company ticker)
        return df

    def feature_engineering(self):
        # Select relevant features
        features = [
            'Revenue', 'Revenue Growth', 'Gross Profit', 'Operating Income','Operating Expenses','Cash and short-term investments','Receivables','Inventories','Total current assets',
            '"Property, Plant & Equipment Net"','Goodwill and Intangible Assets','Long-term investments','Tax assets','Total non-current assets','Total assets','Payables','Short-term debt',
            'Total current liabilities','Long-term debt,Total debt','Deferred revenue','Tax Liabilities','Deposit Liabilities','Total non-current liabilities','Total liabilities',
            'Other comprehensive income','Retained earnings (deficit)','Total shareholders equity','Investments','Net Debt','Other Assets','Other Liabilities','Depreciation & Amortization',
            'Stock-based compensation','Operating Cash Flow','Capital Expenditure','Acquisitions and disposals','Investment purchases and sales','Investing Cash flow','Issuance (repayment) of debt',
            'Issuance (buybacks) of shares','Dividend payments','Financing Cash Flow','Effect of forex changes on cash','Net cash flow / Change in cash','Free Cash Flow','NetCash/Marketcap',
            'priceBookValueRatio','priceToBookRatio','priceToSalesRatio','priceEarningsRatio','priceToFreeCashFlowsRatio','priceToOperatingCashFlowsRatio','priceCashFlowRatio','priceEarningsToGrowthRatio',
            'priceSalesRatio','dividendYield','enterpriseValueMultiple','priceFairValue','ebitperRevenue','ebtperEBIT','niperEBT','grossProfitMargin','operatingProfitMargin','pretaxProfitMargin','netProfitMargin',
            'effectiveTaxRate','returnOnAssets','returnOnEquity','returnOnCapitalEmployed','nIperEBT','eBTperEBIT','eBITperRevenue','payablesTurnover','inventoryTurnover','fixedAssetTurnover','assetTurnover',
            'currentRatio,quickRatio','cashRatio','daysOfSalesOutstanding','daysOfInventoryOutstanding','operatingCycle','daysOfPayablesOutstanding','cashConversionCycle','debtRatio','debtEquityRatio',
            'longtermDebtToCapitalization','totalDebtToCapitalization','interestCoverage','cashFlowToDebtRatio','companyEquityMultiplier','operatingCashFlowPerShare','freeCashFlowPerShare',
            'cashPerShare','payoutRatio','operatingCashFlowSalesRatio','freeCashFlowOperatingCashFlowRatio', 'Net Income', 'EPS', 'Free Cash Flow margin', 'EBITDA', 'Return on Equity', 
            'Debt to Equity', 'Current ratio', 'priceEarningsRatio', 'priceToSalesRatio', 'priceCashFlowRatio','dividendYield', 'payoutRatio', 'operatingCashFlowPerShare',
            'freeCashFlowPerShare', 'cashPerShare', 'bookValuePerShare', 'enterpriseValue', 'enterpriseValueMultiple','ROIC', 'revenuePerShare', 'fcfPerShare','R&D Expenses',
            'Net Income','Dividend per Share','Total current liabilities','Total shareholders equity','payoutRatio'
        ]
        
        # Ensure all selected features are present in the dataset
        available_features = [f for f in features if f in self.data.columns]
        
        # Handle missing values
        feature_data = self.data[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return feature_data

    def build_model(self, n_components=7, n_clusters=10):
        features = self.feature_engineering()
        
        # Normalize features
        self.scaler = StandardScaler()
        normalized_features = self.scaler.fit_transform(features)

        # Dimensionality reduction
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(normalized_features)
      
        # Clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = self.kmeans.fit_predict(reduced_features)
        self.model = Ridge(alpha=1.0)
        self.model.fit(reduced_features, self.data['Cluster'])


        # Collaborative Filtering (Matrix Factorization)
        user_item_matrix = pd.pivot_table(self.data, values='priceEarningsRatio', 
                                          index='Cluster', columns=self.data.index)
        user_item_matrix = user_item_matrix.fillna(user_item_matrix.mean().mean())
        n_components = min(user_item_matrix.shape) - 1
        U, sigma, Vt = svds(user_item_matrix.values, k=n_components)
        sigma = np.diag(sigma)
        self.model = {'U': U, 'sigma': sigma, 'Vt': Vt, 
                      'user_index': user_item_matrix.index, 
                      'item_index': user_item_matrix.columns}

    def get_recommendations(self, user_preferences, top_n=10):
        # Map user preferences to a cluster
        user_features = np.array([
            user_preferences['risk_tolerance'],
            user_preferences['growth_preference'],
            user_preferences['value_preference'],
            user_preferences['dividend_preference'],
            user_preferences['size_preference'],
            user_preferences['dividend_yield_preference'],
            user_preferences['debt_to_equity_preference']
        ]).reshape(1, -1)
        user_cluster = self.kmeans.predict(user_features)[0]

        # Get collaborative filtering recommendations
        cf_recs = self.collaborative_filtering_recommendations(user_cluster)

        # Get content-based recommendations
        cb_recs = self.content_based_recommendations(user_preferences)

        # Combine and rank recommendations
        combined_recs = set(cf_recs) | set(cb_recs)
        final_recs = sorted(combined_recs, 
                            key=lambda x: (x in cf_recs) + (x in cb_recs) + 
                                          self.adjust_score(x, user_preferences),
                            reverse=True)[:top_n]
        
        return final_recs

    def collaborative_filtering_recommendations(self, user_cluster):
        user_index = np.where(self.model['user_index'] == user_cluster)[0][0]
        user_ratings = self.model['U'][user_index, :].dot(self.model['sigma']).dot(self.model['Vt'])
        top_items = self.model['item_index'][np.argsort(-user_ratings)]
        return top_items[:25].tolist()  # Return top 25 for diversity

    def content_based_recommendations(self, user_preferences):
        features = self.feature_engineering()
        normalized_features = self.scaler.transform(features)
        reduced_features = self.pca.transform(normalized_features)
        
        user_profile = np.array([
            user_preferences['risk_tolerance'],
            user_preferences['growth_preference'],
            user_preferences['value_preference'],
            user_preferences['dividend_preference'],
            user_preferences['size_preference'],
            user_preferences['dividend_yield_preference'],
            user_preferences['debt_to_equity_preference']
        ])
        
        similarities = cosine_similarity(reduced_features, user_profile.reshape(1, -1))
        top_items = self.data.index[np.argsort(-similarities.flatten())]
        return top_items[:4000].tolist()  

    def adjust_score(self, company_index, user_preferences):
        company = self.data.loc[company_index]
        risk_score = 1 - (company['Debt to Equity'] / self.data['Debt to Equity'].max())
        growth_score = company['Revenue Growth'] / self.data['Revenue Growth'].max()
        value_score = 1 - (company['priceEarningsRatio'] / self.data['priceEarningsRatio'].max())
        dividend_score = company['dividendYield'] / self.data['dividendYield'].max()
        size_score = company['NetCash/Marketcap'] / self.data['NetCash/Marketcap'].max()
        dividend_yield_score = company['dividendYield'] / self.data['dividendYield'].max()
        debt_to_equity_score = 1 - (company['Debt to Equity'] / self.data['Debt to Equity'].max()) 

        return (
            (1 - abs(user_preferences['risk_tolerance'] - risk_score)) +
            (1 - abs(user_preferences['growth_preference'] - growth_score)) +
            (1 - abs(user_preferences['value_preference'] - value_score)) +
            (1 - abs(user_preferences['dividend_preference'] - dividend_score)) +
            (1 - abs(user_preferences['size_preference'] - size_score))+
            (1 - abs(user_preferences['dividend_yield_preference'] - dividend_yield_score)) +
            (1 - abs(user_preferences['debt_to_equity_preference'] - debt_to_equity_score))
        ) / 7


    def evaluate_model(self):
        features = self.feature_engineering()
        reduced_features = self.pca.transform(self.scaler.transform(features))
        silhouette_avg = silhouette_score(reduced_features, self.data['Cluster'])
        print(f"Silhouette Score: {silhouette_avg}")

    def evaluate_personalization(self, user_preferences_list):
        all_recommendations = []
        
        for user_preference in user_preferences_list:
            recommendations = self.get_recommendations(user_preference, top_n=10)
            all_recommendations.append(recommendations)
        
        # Flatten the list of recommendations
        all_recommendations_flat = [item for sublist in all_recommendations for item in sublist]
        
        # Calculate the number of unique recommendations
        unique_recommendations = len(set(all_recommendations_flat))
        
        # Calculate personalization score
        personalization_score = unique_recommendations / len(all_recommendations_flat)
        
        print(f"Personalization Score: {personalization_score}")
    

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        return joblib.load(filepath)

rec_system = InvestmentRecommendationSystem(filepath)
rec_system.build_model()
rec_system.evaluate_model()

# Get recommendations for a user
def get_investor_preferences():
    preferences = {
        'risk_tolerance': float(input("Enter risk tolerance (0-1, where 0 is low risk and 1 is high risk): ")),
        'growth_preference': float(input("Enter growth preference (0-1, where 0 is low growth and 1 is high growth): ")),
        'value_preference': float(input("Enter value preference (0-1, where 0 is growth-focused and 1 is value-focused): ")),
        'size_preference': float(input("Enter size preference (0-1, where 0 is small cap and 1 is large cap): ")),
        'dividend_preference': float(input("Enter dividend preference (0-1, where 0 is no dividends and 1 is high dividends): ")),
        'dividend_yield_preference': float(input("Enter dividend yield preference (0-1, where 0 is low and 1 is high): ")),
        'debt_to_equity_preference': float(input("Enter debt to equity preference (0-1, where 0 is low and 1 is high): "))
    }
    return preferences

user_preferences = get_investor_preferences()
user_preferences_list = [user_preferences]
personalization_score = rec_system.evaluate_personalization(user_preferences_list)
recommendations= rec_system.get_recommendations(user_preferences, top_n = 10)

print(f"Top Recommendations:\n{recommendations}")

# Save the model
rec_system.save_model('investment_recommendation_model.joblib')
