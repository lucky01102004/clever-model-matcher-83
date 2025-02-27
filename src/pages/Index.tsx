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

        const dataTypes = calculateDataTypes(datasetStats.dataSample);
        const missingValuesPercentage = Number(calculateMissingValues(datasetStats.dataSample));
        const algorithms = getRecommendedAlgorithms(
          dataTypes,
          datasetStats.rows,
          missingValuesPercentage
        );

        const topAlgorithms = algorithms.slice(0, 5);
        const otherAlgorithms = algorithms.slice(5);

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
                      <li>Missing Values: {missingValuesPercentage}%</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-6">
                    <h3 className="text-lg font-medium mb-2">Data Types</h3>
                    <ul className="space-y-2 text-sm text-primary-600">
                      <li>Numerical: {dataTypes.numerical} columns</li>
                      <li>Categorical: {dataTypes.categorical} columns</li>
                      <li>Datetime: {dataTypes.datetime} columns</li>
                      <li>Boolean: {dataTypes.boolean} columns</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        );
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

        const dataTypes = calculateDataTypes(datasetStats.dataSample);
        const missingValuesPercentage = Number(calculateMissingValues(datasetStats.dataSample));
        const algorithms = getRecommendedAlgorithms(
          dataTypes,
          datasetStats.rows,
          missingValuesPercentage
        );

        const topAlgorithms = algorithms.slice(0, 5);
        const otherAlgorithms = algorithms.slice(5);

        return (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                Recommended Algorithms
              </h2>
              <p className="text-primary-600">
                Based on your dataset characteristics
              </p>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-primary-900 mb-4">
                  Top Recommendations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {topAlgorithms.map((algo, i) => (
                    <Card key={i} className="border-2 border-accent">
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
                  Other Available Algorithms
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {otherAlgorithms.map((algo, i) => (
                    <Card key={i}>
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
      case 4:
        return (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-primary-900 mb-2">
                Generated Code
              </h2>
              <p className="text-primary-600">
                Ready-to-use Python code for your selected algorithm
              </p>
            </div>
            <Card>
              <CardContent className="p-6">
                <pre className="bg-primary-50 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-primary-900">
                    {`import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('${selectedFile?.name || "dataset.csv"}')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
`}
                  </code>
                </pre>
              </CardContent>
            </Card>
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
