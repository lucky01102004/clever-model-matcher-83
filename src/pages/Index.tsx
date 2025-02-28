
import { useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronRight, Upload, Code, Database, Settings } from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";

const Index = () => {
  const [step, setStep] = useState(1);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [datasetStats, setDatasetStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  } | null>(null);

  const steps = [
    {
      title: "Upload Dataset",
      description: "Upload your CSV or Excel file to begin",
      icon: Upload,
    },
    {
      title: "Data Analysis",
      description: "Review dataset statistics and insights",
      icon: Database,
    },
    {
      title: "Algorithm Selection",
      description: "Get ML algorithm recommendations",
      icon: Settings,
    },
    {
      title: "Code Generation",
      description: "Download ready-to-use Python code",
      icon: Code,
    },
  ];

  const handleBeginAnalysis = () => {
    if (step === 1 && !selectedFile) {
      toast.error("Please upload a dataset before proceeding");
      return;
    }

    if (step === 3 && !selectedAlgorithm) {
      toast.error("Please select an algorithm before proceeding");
      return;
    }

    if (step < steps.length) {
      setStep(step + 1);
      toast.success(`Moving to ${steps[step].title}`);
    } else {
      toast.error("You are already at the final step");
    }
  };

  const handleFileSelect = (
    file: File | null,
    stats?: {
      rows: number;
      columns: number;
      columnNames: string[];
      dataSample: Record<string, string>[];
    }
  ) => {
    setSelectedFile(file);
    if (file && stats) {
      setDatasetStats(stats);
      toast.success("File analyzed successfully");
    } else {
      setDatasetStats(null);
    }
  };

  const handleAlgorithmSelect = (algorithmName: string) => {
    setSelectedAlgorithm(algorithmName);
    toast.success(`Selected ${algorithmName} algorithm`);
  };

  const calculateDataTypes = (data: Record<string, string>[]) => {
    const columnTypes: Record<string, Set<string>> = {};
    
    Object.keys(data[0] || {}).forEach(column => {
      columnTypes[column] = new Set();
    });

    data.forEach(row => {
      Object.entries(row).forEach(([column, value]) => {
        if (value === null || value === undefined || value === "") {
          columnTypes[column].add("missing");
        } else if (!isNaN(Number(value))) {
          columnTypes[column].add("numerical");
        } else if (value.toLowerCase() === "true" || value.toLowerCase() === "false") {
          columnTypes[column].add("boolean");
        } else if (!isNaN(Date.parse(value))) {
          columnTypes[column].add("datetime");
        } else {
          columnTypes[column].add("categorical");
        }
      });
    });

    const typeCounts = {
      numerical: 0,
      categorical: 0,
      datetime: 0,
      boolean: 0
    };

    Object.values(columnTypes).forEach(typeSet => {
      const types = Array.from(typeSet);
      if (types.includes("numerical")) typeCounts.numerical++;
      else if (types.includes("datetime")) typeCounts.datetime++;
      else if (types.includes("boolean")) typeCounts.boolean++;
      else typeCounts.categorical++;
    });

    return typeCounts;
  };

  const calculateMissingValues = (data: Record<string, string>[]) => {
    let totalCells = 0;
    let missingCells = 0;

    data.forEach(row => {
      Object.values(row).forEach(value => {
        totalCells++;
        if (value === null || value === undefined || value === "") {
          missingCells++;
        }
      });
    });

    return ((missingCells / totalCells) * 100).toFixed(1);
  };

  const getRecommendedAlgorithms = (
    dataTypes: {
      numerical: number;
      categorical: number;
      datetime: number;
      boolean: number;
    },
    totalRows: number,
    missingValuesPercentage: number
  ) => {
    const algorithms = [
      {
        name: "Random Forest",
        score: 0,
        description: "Best for handling both numerical and categorical data. Excellent for avoiding overfitting.",
        useCases: "Classification, Regression",
      },
      {
        name: "XGBoost",
        score: 0,
        description: "Powerful gradient boosting algorithm. Handles missing values well.",
        useCases: "Classification, Regression, Ranking",
      },
      {
        name: "Neural Network",
        score: 0,
        description: "Deep learning model for complex patterns. Good with large datasets.",
        useCases: "Classification, Regression, Pattern Recognition",
      },
      {
        name: "LightGBM",
        score: 0,
        description: "Fast gradient boosting framework. Efficient with large datasets.",
        useCases: "Classification, Regression",
      },
      {
        name: "CatBoost",
        score: 0,
        description: "Handles categorical features automatically. Fast training.",
        useCases: "Classification, Regression",
      },
      {
        name: "SVM",
        score: 0,
        description: "Effective for high-dimensional spaces. Good with clear margins.",
        useCases: "Classification, Regression",
      },
      {
        name: "K-Nearest Neighbors",
        score: 0,
        description: "Simple and interpretable. Good for small to medium datasets.",
        useCases: "Classification, Regression",
      },
      {
        name: "Logistic Regression",
        score: 0,
        description: "Simple and interpretable. Good baseline model.",
        useCases: "Binary Classification",
      },
      {
        name: "Decision Tree",
        score: 0,
        description: "Highly interpretable. Good for feature importance.",
        useCases: "Classification, Regression",
      },
      {
        name: "AdaBoost",
        score: 0,
        description: "Combines weak learners into strong ones. Good with weak patterns.",
        useCases: "Classification, Regression",
      }
    ];

    return algorithms.map(algo => {
      let score = 0;
      
      if (missingValuesPercentage > 5) {
        if (["XGBoost", "CatBoost", "Random Forest"].includes(algo.name)) {
          score += 20;
        }
      }

      if (dataTypes.categorical > 0) {
        if (["CatBoost", "Random Forest", "LightGBM"].includes(algo.name)) {
          score += dataTypes.categorical * 5;
        }
      }

      if (totalRows > 10000) {
        if (["LightGBM", "XGBoost", "Neural Network"].includes(algo.name)) {
          score += 15;
        } else if (["SVM", "K-Nearest Neighbors"].includes(algo.name)) {
          score -= 10;
        }
      } else if (totalRows < 1000) {
        if (["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"].includes(algo.name)) {
          score += 15;
        }
      }

      if (dataTypes.numerical > 0) {
        if (["Neural Network", "XGBoost", "Random Forest"].includes(algo.name)) {
          score += dataTypes.numerical * 3;
        }
      }

      return {
        ...algo,
        score: Math.min(100, Math.max(0, score)),
      };
    })
    .sort((a, b) => b.score - a.score);
  };

  const generateAlgorithmCode = (algorithm: string, fileName: string) => {
    const imports = {
      "Random Forest": "from sklearn.ensemble import RandomForestClassifier",
      "XGBoost": "import xgboost as xgb\nfrom xgboost import XGBClassifier",
      "Neural Network": "from sklearn.neural_network import MLPClassifier",
      "LightGBM": "import lightgbm as lgb\nfrom lightgbm import LGBMClassifier",
      "CatBoost": "from catboost import CatBoostClassifier",
      "SVM": "from sklearn.svm import SVC",
      "K-Nearest Neighbors": "from sklearn.neighbors import KNeighborsClassifier",
      "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
      "Decision Tree": "from sklearn.tree import DecisionTreeClassifier",
      "AdaBoost": "from sklearn.ensemble import AdaBoostClassifier"
    };

    const modelInit = {
      "Random Forest": `model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)`,
      "XGBoost": `model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)`,
      "Neural Network": `model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=1000,
    random_state=42
)`,
      "LightGBM": `model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)`,
      "CatBoost": `model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=False
)`,
      "SVM": `model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)`,
      "K-Nearest Neighbors": `model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto'
)`,
      "Logistic Regression": `model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)`,
      "Decision Tree": `model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=2,
    random_state=42
)`,
      "AdaBoost": `model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)`
    };

    return `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
${imports[algorithm as keyof typeof imports]}

# Load the dataset
df = pd.read_csv('${fileName || "dataset.csv"}')

# Display basic information
print("Dataset Shape:", df.shape)
print("\\nFirst 5 rows:")
print(df.head())
print("\\nData Types:")
print(df.dtypes)
print("\\nSummary Statistics:")
print(df.describe())

# Handle missing values
df = df.fillna(df.mean() if df.select_dtypes(include=[np.number]).columns.any() else df.mode().iloc[0])

# Identify categorical columns and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Assume the target column is the last one - adjust as needed
target_column = df.columns[-1]
feature_columns = [col for col in df.columns if col != target_column]

# Prepare features and target
X = df[feature_columns]
y = df[target_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Initialize and train the model
${modelInit[algorithm as keyof typeof modelInit]}

# Create and fit pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluation metrics
print("\\nModel Evaluation:")
print("--------------------")

try:
    # Classification metrics (will work if target is categorical)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
except:
    # Regression metrics (will work if target is numerical)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Feature importance (for models that support it)
try:
    if hasattr(pipeline['model'], 'feature_importances_'):
        importances = pipeline['model'].feature_importances_
        feature_names = X.columns
        
        # Get indices of features sorted by importance
        indices = np.argsort(importances)[::-1]
        
        print("\\nFeature Importance:")
        for i, idx in enumerate(indices):
            if i < 10:  # Print top 10 features
                print(f"{feature_names[idx]}: {importances[idx]:.4f}")
except:
    pass

print("\\nPredictions (First 5):")
print(y_pred[:5])

# Save the model
import joblib
joblib.dump(pipeline, '${algorithm.toLowerCase().replace(/\s+/g, '_')}_model.pkl')
print("\\nModel saved as '${algorithm.toLowerCase().replace(/\s+/g, '_')}_model.pkl'")
`;
  };

  const renderStepContent = () => {
    switch (step) {
      case 1:
        return (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                Start Your ML Journey
              </h2>
              <p className="text-primary-600">
                Upload your dataset to receive personalized ML recommendations
              </p>
            </div>
            <FileUpload onFileSelect={handleFileSelect} />
            {datasetStats && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 space-y-6"
              >
                <div className="bg-primary-50 p-6 rounded-lg">
                  <h3 className="text-lg font-semibold text-primary-900 mb-4">
                    Dataset Analysis
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white p-4 rounded-lg shadow-sm">
                      <p className="text-sm text-primary-600">Total Rows</p>
                      <p className="text-2xl font-semibold text-primary-900">
                        {datasetStats.rows.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow-sm">
                      <p className="text-sm text-primary-600">Total Columns</p>
                      <p className="text-2xl font-semibold text-primary-900">
                        {datasetStats.columns.toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-primary-50 p-6 rounded-lg">
                  <h3 className="text-lg font-semibold text-primary-900 mb-4">
                    Column Names
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {datasetStats.columnNames.map((column, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-white rounded-full text-sm text-primary-600"
                      >
                        {column}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="bg-primary-50 p-6 rounded-lg overflow-x-auto">
                  <h3 className="text-lg font-semibold text-primary-900 mb-4">
                    Data Preview (First 5 rows)
                  </h3>
                  <table className="min-w-full bg-white rounded-lg overflow-hidden">
                    <thead className="bg-primary-50">
                      <tr>
                        {datasetStats.columnNames.map((column, index) => (
                          <th
                            key={index}
                            className="px-4 py-2 text-left text-sm font-medium text-primary-900"
                          >
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {datasetStats.dataSample.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-t border-primary-100">
                          {datasetStats.columnNames.map((column, colIndex) => (
                            <td
                              key={colIndex}
                              className="px-4 py-2 text-sm text-primary-600"
                            >
                              {String(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </div>
        );
      case 2:
        if (!datasetStats) {
          return (
            <div className="text-center p-8">
              <p className="text-primary-600">
                Please upload a dataset first to view analysis
              </p>
            </div>
          );
        }

        {
          const analysisDataTypes = calculateDataTypes(datasetStats.dataSample);
          const analysisMissingValues = Number(calculateMissingValues(datasetStats.dataSample));

          return (
            <div>
              <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                  Data Analysis
                </h2>
                <p className="text-primary-600">
                  Analyzing your dataset to understand its characteristics
                </p>
              </div>
              <div className="space-y-4">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-medium">Dataset Overview</h3>
                        <p className="text-sm text-primary-600">
                          {selectedFile?.name}
                        </p>
                      </div>
                      <Database className="h-8 w-8 text-accent" />
                    </div>
                  </CardContent>
                </Card>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardContent className="p-6">
                      <h3 className="text-lg font-medium mb-2">Basic Statistics</h3>
                      <ul className="space-y-2 text-sm text-primary-600">
                        <li>Rows: {datasetStats.rows.toLocaleString()}</li>
                        <li>Columns: {datasetStats.columns.toLocaleString()}</li>
                        <li>Missing Values: {analysisMissingValues}%</li>
                      </ul>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-6">
                      <h3 className="text-lg font-medium mb-2">Data Types</h3>
                      <ul className="space-y-2 text-sm text-primary-600">
                        <li>Numerical: {analysisDataTypes.numerical} columns</li>
                        <li>Categorical: {analysisDataTypes.categorical} columns</li>
                        <li>Datetime: {analysisDataTypes.datetime} columns</li>
                        <li>Boolean: {analysisDataTypes.boolean} columns</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          );
        }

      case 3:
        if (!datasetStats) {
          return (
            <div className="text-center p-8">
              <p className="text-primary-600">
                Please upload a dataset first to view algorithm recommendations
              </p>
            </div>
          );
        }

        {
          const algorithmDataTypes = calculateDataTypes(datasetStats.dataSample);
          const algorithmMissingValues = Number(calculateMissingValues(datasetStats.dataSample));
          const recommendedAlgorithms = getRecommendedAlgorithms(
            algorithmDataTypes,
            datasetStats.rows,
            algorithmMissingValues
          );

          const bestAlgorithms = recommendedAlgorithms.slice(0, 5);
          const alternativeAlgorithms = recommendedAlgorithms.slice(5);

          return (
            <div>
              <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                  Algorithm Selection
                </h2>
                <p className="text-primary-600">
                  Choose any algorithm that best suits your needs. Match percentages are recommendations based on your data.
                </p>
              </div>

              <div className="space-y-6">
                <div>
                  <h3 className="text-xl font-semibold text-primary-900 mb-4">
                    Recommended Algorithms
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {bestAlgorithms.map((algo, i) => (
                      <Card 
                        key={i} 
                        className={`border-2 cursor-pointer transition-all duration-200 ${
                          selectedAlgorithm === algo.name
                            ? "border-accent bg-accent/5"
                            : "border-primary-200 hover:border-accent/50"
                        }`}
                        onClick={() => handleAlgorithmSelect(algo.name)}
                      >
                        <CardContent className="p-6">
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium">{algo.name}</h3>
                            <span className="text-accent font-semibold">
                              {algo.score}% Match
                            </span>
                          </div>
                          <p className="text-sm text-primary-600 mb-2">
                            {algo.description}
                          </p>
                          <p className="text-xs text-primary-500">
                            <span className="font-semibold">Use Cases:</span>{" "}
                            {algo.useCases}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-primary-900 mb-4">
                    Other Algorithms
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {alternativeAlgorithms.map((algo, i) => (
                      <Card 
                        key={i}
                        className={`cursor-pointer transition-all duration-200 ${
                          selectedAlgorithm === algo.name
                            ? "border-2 border-accent bg-accent/5"
                            : "hover:border-accent/50"
                        }`}
                        onClick={() => handleAlgorithmSelect(algo.name)}
                      >
                        <CardContent className="p-6">
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium">{algo.name}</h3>
                            <span className="text-primary-600 font-semibold">
                              {algo.score}% Match
                            </span>
                          </div>
                          <p className="text-sm text-primary-600 mb-2">
                            {algo.description}
                          </p>
                          <p className="text-xs text-primary-500">
                            <span className="font-semibold">Use Cases:</span>{" "}
                            {algo.useCases}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        }

      case 4:
        return (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                Generated Code
              </h2>
              <p className="text-primary-600">
                Ready-to-use Python code with evaluation metrics
              </p>
            </div>

            {!selectedAlgorithm ? (
              <div className="text-center p-6 bg-yellow-50 rounded-lg mb-6">
                <p className="text-yellow-700">
                  Please go back and select an algorithm to generate code
                </p>
              </div>
            ) : (
              <>
                <Card className="mb-6">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-medium">Selected Algorithm</h3>
                        <p className="text-accent font-semibold">{selectedAlgorithm}</p>
                      </div>
                      <div>
                        <h3 className="text-lg font-medium">Dataset</h3>
                        <p className="text-primary-600">{selectedFile?.name || "dataset.csv"}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-6">
                    <h3 className="text-lg font-medium mb-4">Python Code with Evaluation Metrics</h3>
                    <pre className="bg-primary-50 p-4 rounded-lg overflow-x-auto text-left">
                      <code className="text-sm text-primary-900 whitespace-pre-wrap">
                        {generateAlgorithmCode(selectedAlgorithm, selectedFile?.name || "dataset.csv")}
                      </code>
                    </pre>
                    <div className="mt-6">
                      <Button 
                        className="bg-accent hover:bg-accent/90 text-white"
                        onClick={() => {
                          const code = generateAlgorithmCode(selectedAlgorithm, selectedFile?.name || "dataset.csv");
                          const blob = new Blob([code], { type: 'text/plain' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `${selectedAlgorithm.toLowerCase().replace(/\s+/g, '_')}_script.py`;
                          a.click();
                          URL.revokeObjectURL(url);
                          toast.success("Code downloaded successfully");
                        }}
                      >
                        Download Python Script
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card className="mt-6">
                  <CardContent className="p-6">
                    <h3 className="text-lg font-medium mb-4">Expected Output Preview</h3>
                    <div className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto font-mono text-sm">
                      <p>Dataset Shape: (1000, 10)</p>
                      <p>&nbsp;</p>
                      <p>First 5 rows:</p>
                      <p>[sample data preview would appear here]</p>
                      <p>&nbsp;</p>
                      <p>Model Evaluation:</p>
                      <p>--------------------</p>
                      <p>Accuracy: 0.8756</p>
                      <p>Precision: 0.8821</p>
                      <p>Recall: 0.8756</p>
                      <p>F1 Score: 0.8783</p>
                      <p>&nbsp;</p>
                      <p>Confusion Matrix:</p>
                      <p>[confusion matrix would appear here]</p>
                      <p>&nbsp;</p>
                      <p>Classification Report:</p>
                      <p>[detailed classification metrics would appear here]</p>
                      <p>&nbsp;</p>
                      <p>Feature Importance:</p>
                      {datasetStats && datasetStats.columnNames.slice(0, 3).map((col, i) => (
                        <p key={i}>{col}: {(Math.random() * 0.3 + 0.1).toFixed(4)}</p>
                      ))}
                      <p>&nbsp;</p>
                      <p>Model saved as '{selectedAlgorithm.toLowerCase().replace(/\s+/g, '_')}_model.pkl'</p>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-secondary to-background">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-primary-900 mb-4">
            ML Algorithm Generator
          </h1>
          <p className="text-lg text-primary-600 max-w-2xl mx-auto">
            Upload your dataset and receive intelligent recommendations for the best
            machine learning algorithms and hyperparameters.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
          {steps.map((s, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <Card
                className={`p-6 ${
                  step === i + 1
                    ? "border-accent border-2"
                    : "border-primary-200"
                }`}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`p-3 rounded-full ${
                      step === i + 1
                        ? "bg-accent text-white"
                        : "bg-primary-100 text-primary-600"
                    }`}
                  >
                    <s.icon size={24} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-primary-900">{s.title}</h3>
                    <p className="text-sm text-primary-600">{s.description}</p>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="max-w-3xl mx-auto"
        >
          <Card className="p-8">
            {renderStepContent()}
            <div className="mt-8 text-center">
              <Button
                size="lg"
                className="bg-accent hover:bg-accent/90 text-white"
                onClick={handleBeginAnalysis}
              >
                {step < steps.length ? (
                  <>
                    Next Step
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </>
                ) : (
                  "Download Code"
                )}
              </Button>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Index;
