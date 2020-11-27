import os
import sys
import streamlit as st

st.header('Short "PYOD" for Anomaly Detection')
st.info("With [**Streamlit**](https://www.streamlit.io/)")
st.info("Credits: **Yue Zhao** <zhaoy@cmu.edu> / https://github.com/yzhao062/pyod")

# License: BSD 2 clause
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
#sys.path.append(
#    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from pyod.models.cof import COF
from pyod.models.sod import SOD

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define the number of inliers and outliers
n_samples = st.slider('Number of samples', 200, 1000, step=200)
outliers_fraction = st.slider('Outliers percent', 0.05, 0.25, step=0.05)
clusters_separation = [0]

# Compare given detectors under given settings
# Initialize the data
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)

ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

# Show the statics of the data
st.write(f'Number of inliers: {n_inliers} , Number of outliers: {n_outliers}')



random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
    '(ABOD) Angle-based Outlier Detector':
        ABOD(contamination=outliers_fraction),
    '(CBLOF) Cluster-based Local Outlier Factor ':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
    '(HBOS) Histogram-base Outlier Detection': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    '(KNN) K Nearest Neighbors ': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    # 'Median KNN': KNN(method='median',
    #                   contamination=outliers_fraction),
    '(LOF) Local Outlier Factor ':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    # 'Local Correlation Integral (LOCI)':
    #     LOCI(contamination=outliers_fraction),
    '(MCD) Minimum Covariance Determinant ': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    '(PCA) Principal Component Analysis ': PCA(
        contamination=outliers_fraction, random_state=random_state),
    # 'Stochastic Outlier Selection (SOS)': SOS(
    #     contamination=outliers_fraction),
    '(LSCP) Locally Selective Combination ': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state),
    # 'Connectivity-Based Outlier Factor (COF)':
    #     COF(n_neighbors=35, contamination=outliers_fraction),
    # 'Subspace Outlier Detection (SOD)':
    #     SOD(contamination=outliers_fraction),
}
st.subheader('SELECT AN ALGORITHM:')

classifier_name = st.selectbox('THE ALGORITHM',[*classifiers])

# Show all detectors
st.subheader(f'Model is: {classifier_name}')
st.write(f'Parameters are: {classifiers[classifier_name]}')

# Fit the models with the generated data and
# compare model performances

# Fit the model
fig = plt.figure(figsize=(12.5, 10))
    

def predict_for_classifier(classifier_name):
  for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # Data generation
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # Add outliers
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]
  
  clf_name = classifier_name
  clf = classifiers[classifier_name]

  # fit the data and tag outliers
  clf.fit(X)
  scores_pred = clf.decision_function(X) * -1
  y_pred = clf.predict(X)
  threshold = percentile(scores_pred, 100 * outliers_fraction)
  n_errors = (y_pred != ground_truth).sum()
  # plot the levels lines and the points

  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
  Z = Z.reshape(xx.shape)
  subplot = plt.subplot(3, 4, i + 1)
  subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
  a = subplot.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
  subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
  b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
  c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
  subplot.axis('tight')
  subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
  subplot.set_xlabel("%s (errors: %d)" % (clf_name, n_errors))
  subplot.set_xlim((-7, 7))
  subplot.set_ylim((-7, 7))
  st.subheader("Outlier detection")
  st.pyplot(fig)

if classifier_name:
  predict_for_classifier(classifier_name)