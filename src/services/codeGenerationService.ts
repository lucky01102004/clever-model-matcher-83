
// This service handles code generation using the Gemini API

import { toast } from "@/components/ui/use-toast";

interface CodeGenerationRequest {
  dataDescription: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  };
  task: string;
}

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent";

// Get API key from localStorage or return empty string
export const getApiKey = (): string => {
  return localStorage.getItem("gemini-api-key") || "";
};

// Save API key to localStorage
export const saveApiKey = (apiKey: string): void => {
  localStorage.setItem("gemini-api-key", apiKey);
};

// Check if API key is set
export const isApiKeySet = (): boolean => {
  return !!getApiKey();
};

export const suggestAlgorithms = async (request: CodeGenerationRequest): Promise<string> => {
  const { dataDescription, task } = request;
  const apiKey = getApiKey();
  
  // Generate example suggestions if no API key is provided
  if (!apiKey) {
    return generateExampleSuggestions(dataDescription, task);
  }
  
  try {
    // Prepare the prompt for algorithm suggestions
    const prompt = `
I have a CSV dataset with the following structure:
- ${dataDescription.rows} rows and ${dataDescription.columns} columns
- Column names: ${dataDescription.columnNames.join(', ')}
- Here's a sample of the data:
${JSON.stringify(dataDescription.dataSample, null, 2)}

Task: ${task}

Based on this dataset and task, please suggest the most suitable algorithm(s) or approach(es). 
For each suggestion, explain:
1. Why this algorithm is appropriate for this data and task
2. What data preprocessing might be required
3. Any potential limitations or considerations

IMPORTANT: If the dataset includes classification tasks, identify the class labels in the dataset. If the class labels are strings, explain the importance of converting them into a one-hot encoded vector to ensure compatibility with machine learning algorithms.

Provide your response in a clear, concise format that can be easily displayed to users.
`;

    // Call the Gemini API
    const response = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1000,
        }
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API error:", errorData);
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the generated text from the response
    let suggestedAlgorithms = "";
    
    if (data.candidates && data.candidates.length > 0 && 
        data.candidates[0].content && 
        data.candidates[0].content.parts && 
        data.candidates[0].content.parts.length > 0) {
      suggestedAlgorithms = data.candidates[0].content.parts[0].text;
    } else {
      throw new Error("Unexpected API response format");
    }

    return suggestedAlgorithms;
  } catch (error) {
    console.error('Error suggesting algorithms:', error);
    toast({
      title: "Error",
      description: "Failed to suggest algorithms. Using example suggestions instead.",
      variant: "destructive",
    });
    return generateExampleSuggestions(dataDescription, task);
  }
};

export const generateCode = async (request: CodeGenerationRequest): Promise<string> => {
  const { dataDescription, task } = request;
  const apiKey = getApiKey();
  
  // Generate example code if no API key is provided
  if (!apiKey) {
    return generateExampleCode(dataDescription, task);
  }
  
  try {
    // Prepare the prompt for the AI
    const prompt = `
I have a CSV dataset with the following structure:
- ${dataDescription.rows} rows and ${dataDescription.columns} columns
- Column names: ${dataDescription.columnNames.join(', ')}
- Here's a sample of the data:
${JSON.stringify(dataDescription.dataSample, null, 2)}

Task: ${task}

IMPORTANT REQUIREMENT: Identify the class labels in the dataset. If the class labels are strings, convert them into a one-hot encoded vector to ensure compatibility with machine learning algorithms. The output should be structured and ready for model training.

Please generate Python code to accomplish this task. Use pandas for data manipulation, matplotlib or seaborn for any visualizations, scikit-learn for any machine learning tasks, and ensure proper handling of categorical variables through one-hot encoding when needed.
`;

    // Call the Gemini API
    const response = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1000,
        }
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API error:", errorData);
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the generated text from the response
    let generatedText = "";
    
    if (data.candidates && data.candidates.length > 0 && 
        data.candidates[0].content && 
        data.candidates[0].content.parts && 
        data.candidates[0].content.parts.length > 0) {
      generatedText = data.candidates[0].content.parts[0].text;
    } else {
      throw new Error("Unexpected API response format");
    }

    return generatedText;
  } catch (error) {
    console.error('Error generating code:', error);
    toast({
      title: "Error",
      description: "Failed to generate code. Using example code instead.",
      variant: "destructive",
    });
    return generateExampleCode(dataDescription, task);
  }
};

// Helper function to generate example code when no API key is available
const generateExampleCode = (dataDescription: CodeGenerationRequest['dataDescription'], task: string): string => {
  const columnNames = dataDescription.columnNames;
  const isMachineLearning = task.toLowerCase().includes('predict') || 
                            task.toLowerCase().includes('classify') ||
                            task.toLowerCase().includes('regression') ||
                            task.toLowerCase().includes('cluster');
  
  const hasNumericColumns = dataDescription.dataSample.some(row => {
    return Object.values(row).some(val => !isNaN(Number(val)));
  });
  
  if (isMachineLearning) {
    // If it's a machine learning task
    const targetColumn = columnNames[columnNames.length - 1];
    const featureColumns = columnNames.filter(col => col !== targetColumn);
    
    return `# Example code for a machine learning task on your dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset 
# (In your real application, replace this with the actual file path)
df = pd.read_csv('your_dataset.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\\nDataset Preview:")
print(df.head())

# Data preprocessing
print("\\nChecking for missing values:")
print(df.isnull().sum())

# Handle missing values (if any)
df = df.fillna(df.mean())

# Identify data types for each column
print("\\nColumn Data Types:")
print(df.dtypes)

# Select features and target
X = df[${JSON.stringify(featureColumns)}]
y = df['${targetColumn}']

# Check if target column has string/categorical values
if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
    print("\\nTarget column '${targetColumn}' contains categorical data.")
    print("\\nUnique classes in target:")
    print(y.value_counts())
    
    # For classification tasks, encode the target variable if it's categorical
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print("\\nEncoded target classes mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name} -> {i}")

# Identify categorical and numerical columns in features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
# Extract feature names after one-hot encoding
if hasattr(model['preprocessor'], 'transformers_'):
    # Get column names after transformation
    ohe = model['preprocessor'].named_transformers_['cat']
    if categorical_features:
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    else:
        cat_feature_names = []
    feature_names = numerical_features + cat_feature_names
else:
    feature_names = X.columns.tolist()

# Get feature importances from the model
importances = model['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
`;
  } else if (hasNumericColumns) {
    // If it's an exploratory data analysis task with numeric data
    return `# Example code for exploratory data analysis on your dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# (In your real application, replace this with the actual file path)
df = pd.read_csv('your_dataset.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\\nDataset Preview:")
print(df.head())

# Statistical summary
print("\\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# Identify data types for each column
print("\\nColumn Data Types:")
print(df.dtypes)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("\\nCategorical Columns:", list(categorical_columns))

# For categorical columns, check unique values
for col in categorical_columns:
    print(f"\\nUnique values in {col}:")
    print(df[col].value_counts())
    
    # If this is a potential class label column with a reasonable number of unique values,
    # demonstrate one-hot encoding
    if df[col].nunique() < 10:
        print(f"\\nOne-hot encoding example for {col}:")
        one_hot = pd.get_dummies(df[col], prefix=col)
        print(one_hot.head())

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Distribution of numerical features
numeric_columns = df.select_dtypes(include=['number']).columns
n_cols = 2
n_rows = (len(numeric_columns) + 1) // 2
plt.figure(figsize=(14, n_rows * 4))

for i, column in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()

plt.show()

# Boxplots for numerical features
plt.figure(figsize=(14, n_rows * 4))
for i, column in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()

plt.show()

# Pairplot of the dataset (limited to first 5 numeric columns for clarity)
sns.pairplot(df[numeric_columns[:5]])
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# If categorical columns exist, show their relationship with numeric data
if len(categorical_columns) > 0 and len(numeric_columns) > 0:
    # For each categorical column with a reasonable number of categories
    for cat_col in categorical_columns:
        if df[cat_col].nunique() < 8:  # Only for columns with a small number of categories
            # Create box plots of numeric features grouped by the categorical feature
            for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=cat_col, y=num_col, data=df)
                plt.title(f'{num_col} by {cat_col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
`;
  } else {
    // For categorical data or other types
    return `# Example code for analyzing categorical data in your dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
# (In your real application, replace this with the actual file path)
df = pd.read_csv('your_dataset.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\\nDataset Preview:")
print(df.head())

# Get column types
print("\\nColumn Data Types:")
print(df.dtypes)

# Count of unique values in each column
print("\\nUnique Values Count:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# Checking for missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("\\nCategorical columns:", list(categorical_columns))

# Analyzing categorical columns
n_cols = 1
n_rows = len(categorical_columns)
plt.figure(figsize=(12, n_rows * 5))

for i, column in enumerate(categorical_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    value_counts = df[column].value_counts().sort_values(ascending=False).head(10)
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Top 10 values in {column}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.tight_layout()

plt.show()

# Convert categorical variables to one-hot encoded vectors
print("\\nOne-Hot Encoding Categorical Variables:")
# Create a one-hot encoder instance
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Select categorical columns with low cardinality (fewer than 10 unique values)
low_cardinality_cols = [col for col in categorical_columns if df[col].nunique() < 10]

if low_cardinality_cols:
    # Fit and transform the selected categorical columns
    encoded_data = encoder.fit_transform(df[low_cardinality_cols])
    
    # Create a dataframe with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(low_cardinality_cols)
    )
    
    # Display the first few rows of the encoded dataframe
    print("\\nFirst 5 rows of one-hot encoded data:")
    print(encoded_df.head())
    
    # Concatenate encoded columns with the original numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        final_df = pd.concat([numeric_df.reset_index(drop=True), encoded_df], axis=1)
        print("\\nFinal dataframe with numeric and encoded categorical data:")
        print(final_df.head())
else:
    print("No suitable categorical columns found for one-hot encoding demonstration.")

# Frequency tables for categorical variables
for column in categorical_columns:
    print(f"\\nFrequency table for {column}:")
    print(df[column].value_counts(normalize=True).head(10))

# Relationships between categorical variables (for the first 2 categorical columns)
if len(categorical_columns) >= 2:
    plt.figure(figsize=(12, 8))
    pd.crosstab(df[categorical_columns[0]], df[categorical_columns[1]]).plot(kind='heatmap', cmap='Blues', annot=True, fmt='d')
    plt.title(f'Relationship between {categorical_columns[0]} and {categorical_columns[1]}')
    plt.tight_layout()
    plt.show()
`;
  }
};

// Helper function to generate example algorithm suggestions when no API key is available
const generateExampleSuggestions = (dataDescription: CodeGenerationRequest['dataDescription'], task: string): string => {
  const taskLower = task.toLowerCase();
  
  // Determine the likely task type
  const isClassification = taskLower.includes('classify') || 
                          taskLower.includes('class') || 
                          taskLower.includes('predict') && taskLower.includes('category');
                          
  const isRegression = taskLower.includes('predict') || 
                       taskLower.includes('forecast') || 
                       taskLower.includes('regression') ||
                       taskLower.includes('value');
                       
  const isClustering = taskLower.includes('cluster') || 
                       taskLower.includes('segment') || 
                       taskLower.includes('group');
                       
  const isDimensionalityReduction = taskLower.includes('dimension') || 
                                   taskLower.includes('reduce') || 
                                   taskLower.includes('compression');
  
  if (isClassification) {
    return `# Suggested Classification Algorithms

Based on your dataset with ${dataDescription.rows} rows and ${dataDescription.columns} columns, here are recommended classification algorithms:

## Data Preprocessing - Handling Class Labels

**IMPORTANT**: When working with class labels, always check if they are strings or categorical. If so, you must convert them into a one-hot encoded vector or numeric labels:

- **One-hot encoding**: Converts categorical variables into a binary matrix where each category gets a binary column.
- **Label encoding**: Maps each class to a unique integer.

This preprocessing is essential for algorithm compatibility and optimal performance.

## 1. Random Forest Classifier
- **Why it's appropriate**: Random Forest works well with mixed data types and handles non-linear relationships. It's robust against overfitting.
- **Preprocessing needed**: 
  - Handle missing values
  - Encode categorical variables
  - Feature scaling is not required
- **Limitations**: 
  - Can be computationally expensive for very large datasets
  - Less interpretable than simpler models

## 2. Gradient Boosting Classifier (XGBoost)
- **Why it's appropriate**: Typically achieves high accuracy and handles complex relationships in data.
- **Preprocessing needed**:
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - More prone to overfitting on noisy data
  - Requires careful parameter tuning

## 3. Logistic Regression
- **Why it's appropriate**: Simple, interpretable baseline model. Works well when the relationship is approximately linear.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - May underperform on complex non-linear relationships
  - Assumes independence between features

## 4. Support Vector Machine (SVM)
- **Why it's appropriate**: Effective in high-dimensional spaces and with clear margins of separation.
- **Preprocessing needed**:
  - Feature scaling is crucial
  - Handle missing values
- **Limitations**:
  - Slow training time for large datasets
  - Sensitive to parameter tuning

## 5. Neural Network
- **Why it's appropriate**: Can capture complex non-linear patterns in the data.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - Requires more data to generalize well
  - Computationally intensive
  - Risk of overfitting with small datasets
`;
  } else if (isRegression) {
    return `# Suggested Regression Algorithms

Based on your dataset with ${dataDescription.rows} rows and ${dataDescription.columns} columns, here are recommended regression algorithms:

## Data Preprocessing - Handling Categorical Variables

**IMPORTANT**: If your dataset contains categorical features, you should convert them into one-hot encoded vectors to ensure compatibility with the regression algorithms:

- **One-hot encoding**: Transforms categorical variables into binary vectors where each category gets its own binary column.
- This preprocessing step is essential for incorporating categorical data in regression models.

## 1. Gradient Boosting Regressor
- **Why it's appropriate**: Typically achieves high accuracy and can model complex non-linear relationships.
- **Preprocessing needed**:
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - Risk of overfitting with noisy data
  - Requires parameter tuning

## 2. Random Forest Regressor
- **Why it's appropriate**: Handles non-linear relationships well and is resistant to overfitting.
- **Preprocessing needed**:
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - Less interpretable than linear models
  - Prediction variance can be high with limited data

## 3. Linear Regression
- **Why it's appropriate**: Simple, interpretable baseline model. Works well for linear relationships.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - Cannot capture non-linear relationships
  - Sensitive to outliers

## 4. Support Vector Regression (SVR)
- **Why it's appropriate**: Good for moderate-sized datasets with complex relationships.
- **Preprocessing needed**:
  - Feature scaling is crucial
  - Handle missing values
- **Limitations**:
  - Slower for large datasets
  - Requires careful parameter tuning

## 5. Neural Network Regressor
- **Why it's appropriate**: Can model complex non-linear relationships.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - Encode categorical variables
- **Limitations**:
  - Requires more data
  - Risk of overfitting with small datasets
  - Training can be computationally expensive
`;
  } else if (isClustering) {
    return `# Suggested Clustering Algorithms

Based on your dataset with ${dataDescription.rows} rows and ${dataDescription.columns} columns, here are recommended clustering algorithms:

## Data Preprocessing - Handling Categorical Data

**IMPORTANT**: When using clustering algorithms with categorical data, you must convert string categories into numeric representations:

- **One-hot encoding**: Transforms categorical variables into binary vectors, creating separate dimensions for each category.
- This preprocessing ensures that clustering algorithms can properly measure distances between points with categorical attributes.

## 1. K-Means Clustering
- **Why it's appropriate**: Simple, fast, and works well when clusters are spherical and similar in size.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
  - Consider dimensionality reduction for high-dimensional data
- **Limitations**:
  - Requires specifying the number of clusters in advance
  - Struggles with clusters of varying shapes and densities

## 2. DBSCAN (Density-Based Spatial Clustering)
- **Why it's appropriate**: Can find arbitrarily shaped clusters and automatically identifies noise points.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Sensitive to parameter choices
  - May struggle with varying density clusters

## 3. Hierarchical Clustering
- **Why it's appropriate**: Creates a hierarchy of clusters, useful for exploratory analysis.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Computationally expensive for large datasets
  - Results can be difficult to interpret with many data points

## 4. Gaussian Mixture Models
- **Why it's appropriate**: Flexible model that can handle overlapping clusters and provides probability of membership.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Requires specifying number of components
  - Sensitive to initialization

## 5. Agglomerative Clustering
- **Why it's appropriate**: Bottom-up hierarchical approach that's intuitive to understand.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Computationally intensive for large datasets
  - Cannot revise previous steps
`;
  } else if (isDimensionalityReduction) {
    return `# Suggested Dimensionality Reduction Techniques

Based on your dataset with ${dataDescription.rows} rows and ${dataDescription.columns} columns, here are recommended dimensionality reduction techniques:

## Data Preprocessing - Handling Categorical Variables

**IMPORTANT**: When applying dimensionality reduction to datasets with categorical features:

- **One-hot encoding**: Convert categorical variables into binary vectors before applying dimensionality reduction techniques.
- This ensures that categorical data is properly represented in the reduced dimensional space.

## 1. Principal Component Analysis (PCA)
- **Why it's appropriate**: Fast, well-established method that preserves maximum variance.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Only captures linear relationships
  - Resulting components may be difficult to interpret

## 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Why it's appropriate**: Excellent for visualization and preserving local structure.
- **Preprocessing needed**:
  - Feature scaling
  - Consider using PCA first for very high-dimensional data
  - One-hot encode categorical variables
- **Limitations**:
  - Computationally intensive for large datasets
  - Results can vary with different parameter settings
  - Better for visualization than as input to other algorithms

## 3. UMAP (Uniform Manifold Approximation and Projection)
- **Why it's appropriate**: Preserves both local and global structure, faster than t-SNE.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Complex algorithm with many parameters
  - Relatively new technique with less documentation

## 4. Linear Discriminant Analysis (LDA)
- **Why it's appropriate**: Supervised technique that maximizes class separability.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - Encode class labels for the target variable
- **Limitations**:
  - Requires class labels (supervised)
  - Assumes normally distributed data with equal covariance matrices

## 5. Autoencoders
- **Why it's appropriate**: Neural network approach that can capture complex non-linear patterns.
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables
- **Limitations**:
  - Requires more data and computational resources
  - Complex to implement and tune
`;
  } else {
    // General exploratory data analysis
    return `# Suggested Approaches for Exploratory Data Analysis

Based on your dataset with ${dataDescription.rows} rows and ${dataDescription.columns} columns, here are recommended analytical approaches:

## Data Preprocessing - Class Label Identification

**IMPORTANT**: When analyzing datasets with categorical variables or potential class labels:

- Identify which columns contain class labels or categories
- For machine learning preparation, convert string class labels into one-hot encoded vectors
- This ensures compatibility with visualization tools and machine learning algorithms

## 1. Statistical Analysis
- **Why it's appropriate**: Provides baseline understanding of data distributions and relationships.
- **Techniques to include**:
  - Descriptive statistics (mean, median, std dev)
  - Correlation analysis
  - Hypothesis testing where appropriate
- **Preprocessing needed**:
  - Handle missing values
  - Check for and handle outliers

## 2. Data Visualization
- **Why it's appropriate**: Helps identify patterns and relationships visually.
- **Techniques to include**:
  - Distribution plots (histograms, box plots)
  - Relationship plots (scatter plots, pair plots)
  - Categorical analysis (bar charts, count plots)
- **Preprocessing needed**:
  - Handle missing values
  - Consider data transformations for skewed variables

## 3. Feature Importance Analysis
- **Why it's appropriate**: Identifies which variables have the strongest relationships with target variables.
- **Techniques to include**:
  - Correlation matrices
  - Feature importance from tree-based models
  - ANOVA or Chi-square tests
- **Preprocessing needed**:
  - Handle missing values
  - Encode categorical variables

## 4. Dimensionality Reduction for Visualization
- **Why it's appropriate**: Helps visualize high-dimensional data.
- **Techniques to include**:
  - PCA
  - t-SNE
- **Preprocessing needed**:
  - Feature scaling
  - Handle missing values
  - One-hot encode categorical variables

## 5. Time Series Analysis (if applicable)
- **Why it's appropriate**: Reveals patterns over time if data has temporal components.
- **Techniques to include**:
  - Trend analysis
  - Seasonality detection
  - Autocorrelation analysis
- **Preprocessing needed**:
  - Ensure data is in chronological order
  - Handle missing values
`;
  }
};

