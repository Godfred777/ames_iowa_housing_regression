from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression, Lasso # Import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures

class LinearRegressionPipelineWithPCA:
    """
    A pipeline with a standard scaler, PCA for dimensionality reduction, and a linear regression model. The number of PCA components can be specified when initializing the pipeline.
    parameters:
- n_components: The number of principal components to keep. If None, all components are kept.
returns:
- A pipeline object.
throws:
- None.
    """
    def __init__(self, n_components=None):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)
    

class LinearRegressionPipelineWithKernelPCA:
    def __init__(self, n_components=None, kernel='rbf'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_pca', KernelPCA(n_components=n_components, kernel=kernel)),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)
    

class LinearRegressionPipelineWithLDA:
    def __init__(self, n_components=None):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(n_components=n_components)),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)
    

class LinearRegressionPipelineWithPCAAndLDA:
    def __init__(self, n_components_pca=None, n_components_lda=None):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components_pca)),
            ('lda', LinearDiscriminantAnalysis(n_components=n_components_lda)),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)


class LinearRegressionPipelineWithL1Regulation:
    def __init__(self, alpha=1.0, max_iter=1000):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=alpha, max_iter=max_iter)) # Use Lasso for L1 regularization
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)


class LinearRegressionPipelineWithPolynomialFeatures:
    def __init__(self, degree=2):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('polynomial_features', PolynomialFeatures(degree=degree)),
            ('regressor', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)