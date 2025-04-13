import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    st.title("About the Data 💾")
  
    st.write(
        """ 
            Our dataset is a collection of patient observations documenting risk factors associated with Alzheimer's disease, as well as their specific diagnosis. 
            The data has a global perspective, coming from 20 different countries across the world, including the United States, United Kingdom, China, India, Brazil and many more, 
            with an even spread of roughly 3700 records per country. 
            The data was gathered and made public by a user on Kaggle. 
            Many of the features in our dataset are frequently cited in popular scientific literature as potential links to development and progression of various dementia symptoms and Alzheimer's disease variants, 
            making it prime for investigating relevant claims about Alzheimer's risk factors.
        """
    )

    DATA_URL_1 = ('data/alzheimers.csv')
    DATA_URL_2 = ('data/alzheimers_encoded.csv')
    alzheimers = pd.read_csv(DATA_URL_1)
    alzheimers_encoded = pd.read_csv(DATA_URL_2)

    tab1, tab2, tab3, tab4 = st.tabs(["Original Data Set", "Preprocessing", "Analysis", "Feature Selection"])

    with tab1:
        st.header("Original Data Set")
      
        st.write(
            """
                Our datasets comes from the following two links:
            """
        )

        st.write(
            """
              https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global
            """
        )

        alzheimers

        st.write(
            """
              The data is tabular, with a mixture of 24 different qualitative and quantitative features:
            """
        )

    with tab2:
        st.header("Preprocessing")
      
        st.markdown(
          """
              ### Preprocessing Steps
              
              1. **Standardization of numerical features**  
                 - All continuous variables will be standardized (zero mean, unit variance).
              
              2. **Feature-appropriate encoding methods**  
                 - **One-hot encoding** for **nominal variables** (no natural order).  
                 - **Ordinal encoding** for **ordered categorical variables** (with natural order).  
                 - **Label encoding** for **binary categorical features** (e.g., Yes/No for the target).
              
              3. **Column name normalization**  
                 - Convert column names to lowercase, replace spaces with underscores, and removed apostraphes.
          """
        )

        st.write(
            """
                For specific operations, please refer to the file preprocessing.py from the source library.
            """
        )

        st.write(
            """
                New data set after processing: 
            """
        )

        alzheimers_encoded
    
    with tab3:
        st.header("Analysis")

        st.write("# Summary Statistics:")
        st.write("We first want to understand the structure of the dataset, including the size of the dataset, qualitative and quantitative features.")

        alzheimers.info()
        alzheimers.describe().style.set_caption("Numerical Columns")
        alzheimers.select_dtypes(include="object").describe().style.set_caption("Categorical Columns")

        st.write(
            """
                Quantitative (4): Age, BMI, Cognitive Test Score, Education Level
            """
        )

        st.write(
            """
                Qualitative (20): Physical Activity Level, Alcohol Consumption, Stress Levels, Country, Diabetes, Smoking Status, ...
            """
        )

        st.write(
            """
                This dataset appears to be very clean. 
                No missing values, the row counts for each attribute remain consistent for all. 
                Data types appear as expected.
                Frequency counts for categorical variables show a good distribution for each.
            """
        )
      
        st.write("# Linear correlation analysis:")
      
        sns.heatmap(alzheimers.select_dtypes(include='number').corr().iloc[::-1], vmin=-1, vmax=1, cmap='coolwarm', annot=True, square=True)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.title("Linear correlations of numerical features")
        plt.tight_layout()
        plt.show()

        st.write(
            """
              The heatmap shows that most features have no or only limited direct linear correlation with diagnosis.
            """
        )

        st.write(
            """
              We would like to refer to categorical features to explore more influencing factors of Alzheimer's disease. 
              We will find the most valuable features and perform feature selection in subsequent analysis.
            """
        )
      
  with tab4:
        st.header("Feature Selection")

        st.write("Analysis based on the encoded dataset:")

        alzheimers_encoded.head()
      
        st.write(
            """
              Only some models have feature_importances_ or coef_, the most straightforward metrics for feature importance. 
              We will only be doing feature importance analysis on those models that have such attributes:
            """
        )

        def get_feature_importances(model, feature_names):
            # Tree based models usually have a feature_importances_ attribute
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return list(zip(feature_names, importances))
        
            # Linear models usually have a coef_ attribute
            elif hasattr(model, "coef_"):
                coef = model.coef_
                importances = np.abs(coef[0])
        
                return list(zip(feature_names, importances))
            else:
                return None
        
        def run_feature_importance_analysis(num_importances=5):
            fitted_models = {}
            for name, model in estimators.items():
                print(f"Fitting {name} ...")
                model.fit(numerical_data, target)
                fitted_models[name] = model
        
            for name, model in fitted_models.items():
                importances = get_feature_importances(model, feature_names)
                if importances is not None:
                    print(f"\n{name} feature importances:")
                    for feat, val in sorted(importances, key=lambda x: x[1], reverse=True)[:5]:
                        print(f"\t{feat}: {val:.4f}")
                else:
                    print(f"\n{name} does not provide a direct feature importance measure.")
        
        run_feature_importance_analysis()

        st.write(
            """
              Find the top 5 most important features through visualization：
            """
        )

      def plot_feature_importance(model_name, num_importances = 5):
          if model_name in estimators.keys():
              print(f"Fitting {model_name} ...")
              model = estimators[model_name]        
              model.fit(numerical_data, target)
              print("Finished fitting")
              importances = get_feature_importances(model, numerical_data.columns)
              sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
              top_importances = sorted_importances[:num_importances]
              # Separate names and values for plotting
              labels = [t[0] for t in top_importances]
              values = [t[1] for t in top_importances]
              
              # Create a bar plot
              plt.figure(figsize=(8, 5))
              plt.barh(range(len(values)), values)
              plt.yticks(range(len(values)), labels)
              plt.title(f"Feature Importances ({model_name})")
              plt.xlabel("importance")
              plt.ylabel("Feature")
              plt.show()
      
      plot_feature_importance("RandomForest")
        
if __name__ == "__main__":
    main()
