
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

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    if (file) {
      toast.success("File uploaded successfully");
    }
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
          </div>
        );
      case 2:
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
                      <li>Rows: 1,000</li>
                      <li>Columns: 15</li>
                      <li>Missing Values: 2%</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-6">
                    <h3 className="text-lg font-medium mb-2">Data Types</h3>
                    <ul className="space-y-2 text-sm text-primary-600">
                      <li>Numerical: 8 columns</li>
                      <li>Categorical: 5 columns</li>
                      <li>Datetime: 2 columns</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        );
      case 3:
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  name: "Random Forest",
                  accuracy: "92%",
                  description:
                    "Best for handling both numerical and categorical data",
                },
                {
                  name: "XGBoost",
                  accuracy: "89%",
                  description: "Excellent for complex non-linear relationships",
                },
                {
                  name: "Neural Network",
                  accuracy: "87%",
                  description: "Good for large datasets with deep patterns",
                },
                {
                  name: "SVM",
                  accuracy: "85%",
                  description: "Effective for high-dimensional data",
                },
              ].map((algo, i) => (
                <Card key={i}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-medium">{algo.name}</h3>
                      <span className="text-accent font-semibold">
                        {algo.accuracy}
                      </span>
                    </div>
                    <p className="text-sm text-primary-600">
                      {algo.description}
                    </p>
                  </CardContent>
                </Card>
              ))}
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
