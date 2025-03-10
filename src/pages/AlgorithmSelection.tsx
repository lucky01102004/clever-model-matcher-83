
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Database, CheckCircle, FileWarning, Info } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { toast } from "sonner";

// Define algorithm types
type AlgorithmType = 'classification' | 'regression' | 'clustering' | 'dimensionality_reduction';

// Define algorithm interface
interface Algorithm {
  id: string;
  name: string;
  description: string;
  type: AlgorithmType;
  score: number; // Compatibility score (0-100)
  libraries: string[];
  pros: string[];
  cons: string[];
}

// Pre-defined algorithms by type
const algorithmsByType: Record<AlgorithmType, Algorithm[]> = {
  classification: [
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'A statistical model that uses a logistic function to model a binary dependent variable.',
      type: 'classification',
      score: 85,
      libraries: ['sklearn', 'statsmodels'],
      pros: ['Simple', 'Interpretable', 'Fast training'],
      cons: ['May underfit complex relationships', 'Assumes linear decision boundary']
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'An ensemble learning method that constructs multiple decision trees during training.',
      type: 'classification',
      score: 92,
      libraries: ['sklearn'],
      pros: ['Handles non-linear relationships', 'Feature importance', 'Reduces overfitting'],
      cons: ['Less interpretable', 'Computationally intensive for large datasets']
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'Finds a hyperplane that best divides a dataset into classes.',
      type: 'classification',
      score: 88,
      libraries: ['sklearn'],
      pros: ['Effective in high-dimensional spaces', 'Works well with clear margin of separation'],
      cons: ['Sensitive to feature scaling', 'Slower training with large datasets']
    },
    {
      id: 'knn',
      name: 'K-Nearest Neighbors',
      description: 'Instance-based learning where new instances are classified based on closest training examples.',
      type: 'classification',
      score: 80,
      libraries: ['sklearn'],
      pros: ['Simple to understand', 'No training phase', 'Naturally handles multi-class cases'],
      cons: ['Computationally expensive for large datasets', 'Sensitive to irrelevant features']
    },
    {
      id: 'neural_network',
      name: 'Neural Network',
      description: 'A series of algorithms that attempt to recognize underlying relationships in a set of data.',
      type: 'classification',
      score: 90,
      libraries: ['tensorflow', 'keras', 'pytorch'],
      pros: ['Handles complex non-linear relationships', 'Flexible architecture for different data types'],
      cons: ['Needs large amounts of data', 'Black box (limited interpretability)', 'Computationally intensive']
    }
  ],
  regression: [
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      description: 'Models the relationship between a dependent variable and one or more independent variables.',
      type: 'regression',
      score: 88,
      libraries: ['sklearn', 'statsmodels'],
      pros: ['Simple and interpretable', 'Fast training', 'Easy to understand coefficients'],
      cons: ['Assumes linear relationship', 'Sensitive to outliers']
    },
    {
      id: 'random_forest_regressor',
      name: 'Random Forest Regressor',
      description: 'An ensemble of decision trees for regression tasks.',
      type: 'regression',
      score: 92,
      libraries: ['sklearn'],
      pros: ['Handles non-linearity well', 'Robust to outliers', 'Feature importance'],
      cons: ['Less interpretable', 'Memory intensive for large datasets']
    },
    {
      id: 'svr',
      name: 'Support Vector Regression',
      description: 'Applies SVM principles to regression tasks.',
      type: 'regression',
      score: 85,
      libraries: ['sklearn'],
      pros: ['Effective with high-dimensional data', 'Works well with small datasets'],
      cons: ['Sensitive to parameter tuning', 'Slower for large datasets']
    },
    {
      id: 'gradient_boosting',
      name: 'Gradient Boosting Regressor',
      description: 'Builds trees sequentially where each tree corrects errors from previous trees.',
      type: 'regression',
      score: 94,
      libraries: ['sklearn', 'xgboost', 'lightgbm'],
      pros: ['High accuracy', 'Handles mixed data types well', 'Implicit feature selection'],
      cons: ['Can overfit noisy data', 'Computationally intensive', 'Requires parameter tuning']
    },
    {
      id: 'nn_regressor',
      name: 'Neural Network Regressor',
      description: 'Uses neural networks for regression tasks.',
      type: 'regression',
      score: 89,
      libraries: ['tensorflow', 'keras', 'pytorch'],
      pros: ['Handles complex relationships', 'Can model almost any function'],
      cons: ['Needs more data', 'Difficult to interpret', 'Computationally intensive']
    }
  ],
  clustering: [
    {
      id: 'kmeans',
      name: 'K-Means Clustering',
      description: 'Partitions observations into k clusters, each with the nearest mean.',
      type: 'clustering',
      score: 90,
      libraries: ['sklearn'],
      pros: ['Simple to understand', 'Scalable to large datasets', 'Fast for low-dimensional data'],
      cons: ['Requires specifying number of clusters', 'Sensitive to initial conditions', 'Assumes spherical clusters']
    },
    {
      id: 'hierarchical',
      name: 'Hierarchical Clustering',
      description: 'Builds nested clusters by merging or splitting them successively.',
      type: 'clustering',
      score: 85,
      libraries: ['sklearn', 'scipy'],
      pros: ['No need to specify number of clusters beforehand', 'Produces dendrogram visualization'],
      cons: ['Computationally intensive for large datasets', 'Can be sensitive to noise']
    },
    {
      id: 'dbscan',
      name: 'DBSCAN',
      description: 'Density-based spatial clustering that groups together points that are closely packed.',
      type: 'clustering',
      score: 88,
      libraries: ['sklearn'],
      pros: ['Can find arbitrarily shaped clusters', 'Doesn\'t require number of clusters', 'Robust to outliers'],
      cons: ['Sensitive to parameters', 'Struggles with varying density clusters']
    },
    {
      id: 'gaussian_mixture',
      name: 'Gaussian Mixture Models',
      description: 'Assumes data points are generated from a mixture of several Gaussian distributions.',
      type: 'clustering',
      score: 82,
      libraries: ['sklearn'],
      pros: ['Soft clustering (probability of membership)', 'Flexible cluster shapes'],
      cons: ['Sensitive to initialization', 'May converge to local optima']
    },
    {
      id: 'agglomerative',
      name: 'Agglomerative Clustering',
      description: 'A hierarchical clustering approach that builds nested clusters bottom-up.',
      type: 'clustering',
      score: 80,
      libraries: ['sklearn'],
      pros: ['Creates hierarchy of clusters', 'No need to specify number of clusters in advance'],
      cons: ['Computationally intensive', 'Difficult to scale to large datasets']
    }
  ],
  dimensionality_reduction: [
    {
      id: 'pca',
      name: 'Principal Component Analysis',
      description: 'Reduces the dimensionality by finding the directions of maximum variance.',
      type: 'dimensionality_reduction',
      score: 95,
      libraries: ['sklearn'],
      pros: ['Simple to understand', 'Fast computation', 'Preserves most variance'],
      cons: ['Only captures linear relationships', 'Sensitive to feature scaling']
    },
    {
      id: 'tsne',
      name: 't-SNE',
      description: 'Visualizes high-dimensional data by modeling similar points close together.',
      type: 'dimensionality_reduction',
      score: 88,
      libraries: ['sklearn'],
      pros: ['Excellent for visualization', 'Preserves local structure'],
      cons: ['Computationally intensive', 'Stochastic results', 'Not suitable for very large datasets']
    },
    {
      id: 'umap',
      name: 'UMAP',
      description: 'Uniform Manifold Approximation and Projection for dimension reduction.',
      type: 'dimensionality_reduction',
      score: 92,
      libraries: ['umap-learn'],
      pros: ['Preserves global and local structure', 'Faster than t-SNE', 'Scales better to large datasets'],
      cons: ['More recent algorithm with less documentation', 'Sensitive to parameters']
    },
    {
      id: 'lda',
      name: 'Linear Discriminant Analysis',
      description: 'Finds a linear combination of features that separates classes.',
      type: 'dimensionality_reduction',
      score: 84,
      libraries: ['sklearn'],
      pros: ['Supervised (uses class labels)', 'Good for pre-classification dimensionality reduction'],
      cons: ['Assumes normal distribution', 'Limited to linear boundaries']
    },
    {
      id: 'autoencoder',
      name: 'Autoencoder',
      description: 'Uses neural networks to learn efficient encodings of data.',
      type: 'dimensionality_reduction',
      score: 90,
      libraries: ['tensorflow', 'keras', 'pytorch'],
      pros: ['Can learn non-linear representations', 'Highly flexible architecture'],
      cons: ['Requires more data', 'Computationally intensive', 'More complex to implement and tune']
    }
  ]
};

const AlgorithmSelection = () => {
  const navigate = useNavigate();
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
    statistics?: any;
  } | null>(null);
  
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [datasetType, setDatasetType] = useState<AlgorithmType | null>(null);
  const [recommendedAlgorithms, setRecommendedAlgorithms] = useState<Algorithm[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);
  const [classLabels, setClassLabels] = useState<string[]>([]);
  
  useEffect(() => {
    // Retrieve data from localStorage when component mounts
    const savedData = localStorage.getItem('uploadedDataset');
    if (savedData) {
      try {
        const parsedData = JSON.parse(savedData);
        setFileStats(parsedData);
        if (parsedData.suggestedTarget) {
          setSelectedTarget(parsedData.suggestedTarget);
          
          // Try to determine dataset type
          guessDatasetType(parsedData);
          
          // Identify potential class labels in the target column
          identifyClassLabels(parsedData);
        } else {
          setSelectedTarget(null);
          setDatasetType(null);
        }
      } catch (error) {
        console.error("Error parsing saved data:", error);
        toast.error("Failed to load dataset. Please return to Upload Dataset page.");
      }
    }
  }, []);
  
  const identifyClassLabels = (stats: any) => {
    if (!stats.suggestedTarget) return;
    
    // Get values from suggested target column
    const targetValues = stats.dataSample.map((row: any) => String(row[stats.suggestedTarget || ""]));
    const uniqueValues = [...new Set(targetValues)];
    
    // If we have a reasonable number of unique values (not too many), treat them as class labels
    if (uniqueValues.length > 1 && uniqueValues.length <= 15) {
      setClassLabels(uniqueValues);
    } else {
      setClassLabels([]);
    }
  };
  
  const guessDatasetType = (stats: any) => {
    if (!stats.suggestedTarget) return;
    
    const targetValues = stats.dataSample.map((row: any) => String(row[stats.suggestedTarget || ""]));
    const uniqueValues = new Set(targetValues);
    
    // Check statistics
    if (stats.statistics?.classDistribution) {
      const uniqueClasses = Object.keys(stats.statistics.classDistribution).length;
      
      if (uniqueClasses <= 10) {
        // Classification with few classes
        setDatasetType('classification');
        return;
      }
    }
    
    // Check if target is likely numeric, suggesting regression
    const numericTarget = !isNaN(Number(targetValues[0]));
    if (numericTarget) {
      setDatasetType('regression');
    } else if (uniqueValues.size <= 15) {
      // Classification with moderate number of classes
      setDatasetType('classification');
    } else {
      // Default to clustering if unclear
      setDatasetType('clustering');
    }
  };
  
  useEffect(() => {
    if (datasetType) {
      // Get algorithms for the dataset type
      const algorithms = algorithmsByType[datasetType] || [];
      
      // Sort by score descending
      const sorted = [...algorithms].sort((a, b) => b.score - a.score);
      
      // Take top 5
      setRecommendedAlgorithms(sorted.slice(0, 5));
      
      // Select top algorithm by default
      if (sorted.length > 0 && !selectedAlgorithm) {
        setSelectedAlgorithm(sorted[0]);
      }
    } else {
      setRecommendedAlgorithms([]);
      setSelectedAlgorithm(null);
    }
  }, [datasetType]);
  
  const handleSelectAlgorithm = (algorithm: Algorithm) => {
    setSelectedAlgorithm(algorithm);
    toast(`${algorithm.name} selected`);
  };
  
  const handleContinue = () => {
    if (!selectedAlgorithm) {
      toast.error("Please select an algorithm first");
      return;
    }
    
    // Save selected algorithm and data in localStorage for code generation
    const codeGenerationData = {
      algorithmId: selectedAlgorithm.id,
      algorithmName: selectedAlgorithm.name,
      algorithmType: selectedAlgorithm.type,
      dataDescription: fileStats,
      targetColumn: selectedTarget,
      classLabels: classLabels.length > 0 ? classLabels : null
    };
    
    localStorage.setItem('codeGenerationData', JSON.stringify(codeGenerationData));
    toast.success("Ready for code generation");
  };

  const handleGoToUpload = () => {
    navigate('/upload-dataset');
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <Link to="/" className="flex items-center text-primary hover:underline">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Home
        </Link>
      </div>
      
      <h1 className="text-3xl font-bold mb-2">Algorithm Selection</h1>
      <p className="text-gray-600 mb-8">Select the most suitable algorithm for your dataset</p>
      
      <div className="grid grid-cols-1 gap-8">
        {fileStats ? (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Dataset Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-gray-50 p-3 rounded">
                    <p className="text-sm text-gray-500">Rows</p>
                    <p className="text-lg font-medium">{fileStats.rows}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded">
                    <p className="text-sm text-gray-500">Columns</p>
                    <p className="text-lg font-medium">{fileStats.columns}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded">
                    <p className="text-sm text-gray-500">Dataset Type</p>
                    <p className="text-lg font-medium capitalize">
                      {datasetType ? datasetType.replace('_', ' ') : 'Unknown'}
                    </p>
                  </div>
                </div>
                
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Target Column
                  </label>
                  <Select 
                    value={selectedTarget || ""} 
                    onValueChange={(value) => {
                      setSelectedTarget(value);
                      
                      // Re-analyze target column
                      const newStats = {...fileStats, suggestedTarget: value};
                      guessDatasetType(newStats);
                      identifyClassLabels(newStats);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {fileStats.columnNames.map(column => (
                        <SelectItem 
                          key={column} 
                          value={column}
                          className={column === fileStats.suggestedTarget ? "font-bold bg-primary/10" : ""}
                        >
                          {column} {column === fileStats.suggestedTarget ? "(Suggested)" : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                {/* Class Labels Alert */}
                {classLabels.length > 0 && (
                  <Alert className="mb-6">
                    <Info className="h-4 w-4" />
                    <AlertTitle>Class Labels Identified</AlertTitle>
                    <AlertDescription>
                      <p className="mb-2">
                        We've identified {classLabels.length} class labels in the target column. 
                        These will be automatically one-hot encoded during processing.
                      </p>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {classLabels.map((label, index) => (
                          <Badge key={index} variant="outline">
                            {label}
                          </Badge>
                        ))}
                      </div>
                    </AlertDescription>
                  </Alert>
                )}
                
                {selectedTarget && (
                  <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Dataset Type
                    </label>
                    <Select 
                      value={datasetType || ""} 
                      onValueChange={(value) => setDatasetType(value as AlgorithmType)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select dataset type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="classification">Classification</SelectItem>
                        <SelectItem value="regression">Regression</SelectItem>
                        <SelectItem value="clustering">Clustering</SelectItem>
                        <SelectItem value="dimensionality_reduction">Dimensionality Reduction</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </CardContent>
            </Card>
            
            {datasetType && recommendedAlgorithms.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recommended Algorithms</CardTitle>
                  <CardDescription>
                    Based on your dataset, we recommend the following algorithms:
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 gap-4">
                    {recommendedAlgorithms.map(algorithm => (
                      <div 
                        key={algorithm.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          selectedAlgorithm?.id === algorithm.id 
                            ? 'border-primary bg-primary/5' 
                            : 'border-gray-200 hover:border-primary/50'
                        }`}
                        onClick={() => handleSelectAlgorithm(algorithm)}
                      >
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2">
                              <h3 className="text-lg font-medium">{algorithm.name}</h3>
                              <Badge variant="outline" className="text-xs">
                                {algorithm.score}% match
                              </Badge>
                            </div>
                            <p className="text-sm text-gray-600 mt-1">{algorithm.description}</p>
                          </div>
                          {selectedAlgorithm?.id === algorithm.id && (
                            <CheckCircle className="h-5 w-5 text-primary" />
                          )}
                        </div>
                        
                        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <h4 className="text-sm font-medium text-gray-700">Pros</h4>
                            <ul className="mt-1 text-sm text-gray-600">
                              {algorithm.pros.map((pro, i) => (
                                <li key={i} className="flex items-start">
                                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 mr-2"></span>
                                  {pro}
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <h4 className="text-sm font-medium text-gray-700">Cons</h4>
                            <ul className="mt-1 text-sm text-gray-600">
                              {algorithm.cons.map((con, i) => (
                                <li key={i} className="flex items-start">
                                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500 mt-1.5 mr-2"></span>
                                  {con}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                        
                        <div className="mt-3">
                          <h4 className="text-sm font-medium text-gray-700">Libraries</h4>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {algorithm.libraries.map(lib => (
                              <Badge key={lib} variant="secondary" className="text-xs">
                                {lib}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Show all algorithms option */}
                  <div className="mt-4">
                    <Select
                      onValueChange={(value) => {
                        const algorithm = Object.values(algorithmsByType)
                          .flat()
                          .find(alg => alg.id === value);
                          
                        if (algorithm) {
                          handleSelectAlgorithm(algorithm);
                        }
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select from all algorithms" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="browse_header" disabled>Browse all algorithms</SelectItem>
                        {Object.entries(algorithmsByType).map(([type, algorithms]) => (
                          <React.Fragment key={type}>
                            <SelectItem value={`header_${type}`} disabled className="font-bold uppercase">
                              {type.replace('_', ' ')}
                            </SelectItem>
                            {algorithms.map(alg => (
                              <SelectItem key={alg.id} value={alg.id}>
                                {alg.name} ({alg.score}% match)
                              </SelectItem>
                            ))}
                          </React.Fragment>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
                
                <CardFooter className="flex flex-col pb-3 pt-6">
                  <div className="w-full mb-4 bg-blue-50 p-4 rounded-lg text-sm text-blue-800">
                    <p className="font-medium mb-1">Preprocessing note:</p>
                    <p>
                      {classLabels.length > 0 ? 
                        `Your dataset contains ${classLabels.length} string class labels that will be automatically converted to one-hot encoded vectors during preprocessing.` :
                        "Any string or categorical data in your dataset will be automatically converted to a numeric format compatible with machine learning algorithms."}
                    </p>
                  </div>
                </CardFooter>
              </Card>
            )}
          
            <div className="flex justify-center">
              <Button 
                disabled={!selectedAlgorithm} 
                onClick={handleContinue}
                asChild
              >
                <Link to="/code-generator">
                  Continue to Code Generation
                </Link>
              </Button>
            </div>
          </>
        ) : (
          <Card>
            <CardContent className="py-10">
              <div className="text-center">
                <FileWarning className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                <h2 className="text-xl font-medium mb-2">No dataset found</h2>
                <p className="text-gray-500 mb-6">
                  You need to upload a dataset before selecting an algorithm
                </p>
                <Button onClick={handleGoToUpload}>
                  Go to Upload Dataset
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AlgorithmSelection;

