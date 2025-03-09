
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUpload } from "@/components/FileUpload";
import { toast } from "@/hooks/use-toast";
import ApiKeyManager from "@/components/ApiKeyManager";
import { isApiKeySet, suggestAlgorithms } from "@/services/codeGenerationService";
import { Loader2, Lightbulb } from "lucide-react";

const AlgorithmSelection = () => {
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    targetColumn?: string;
  } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [task, setTask] = useState<string>("");
  const [algorithmSuggestions, setAlgorithmSuggestions] = useState<string>("");
  const [isSuggesting, setIsSuggesting] = useState<boolean>(false);
  const [hasApiKey, setHasApiKey] = useState<boolean>(false);

  useEffect(() => {
    setHasApiKey(isApiKeySet());
    
    // Check for API key changes
    const checkApiKey = () => {
      setHasApiKey(isApiKeySet());
    };
    
    window.addEventListener('storage', checkApiKey);
    return () => window.removeEventListener('storage', checkApiKey);
  }, []);

  const handleFileSelect = (file: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  }) => {
    setSelectedFile(file);
    if (stats) {
      // Detect target column - typically the last column or one with classification-like values
      const possibleTargetColumns = detectTargetColumn(stats);
      setFileStats({
        ...stats,
        targetColumn: possibleTargetColumns
      });
    } else {
      setFileStats(null);
    }
  };

  // Simple heuristic to detect target column
  const detectTargetColumn = (stats: {
    columnNames: string[];
    dataSample: Record<string, string>[];
  }): string => {
    const { columnNames, dataSample } = stats;
    
    // Common target column names
    const targetKeywords = ['target', 'class', 'label', 'outcome', 'result', 'category', 'type'];
    
    // Check if any column name contains target keywords
    for (const column of columnNames) {
      const lowerColumn = column.toLowerCase();
      if (targetKeywords.some(keyword => lowerColumn.includes(keyword))) {
        return column;
      }
    }
    
    // If no obvious target column, return the last column as a fallback
    return columnNames[columnNames.length - 1];
  };

  const handleSuggestAlgorithms = async () => {
    if (!fileStats) return;
    
    setIsSuggesting(true);
    try {
      // Use the detected target column in the task if no specific task is provided
      const taskDescription = task.trim() || 
        `Suggest the best algorithms for predicting the ${fileStats.targetColumn} column based on other features`;
      
      const suggestions = await suggestAlgorithms({
        dataDescription: fileStats,
        task: taskDescription
      });
      setAlgorithmSuggestions(suggestions);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate algorithm suggestions. Please check your API key.",
        variant: "destructive",
      });
    } finally {
      setIsSuggesting(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      {/* API Key Management */}
      <ApiKeyManager />
      
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Algorithm Selection</CardTitle>
          <CardDescription>
            Upload your dataset to get algorithm recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">1. Upload Your Dataset</h3>
            <FileUpload onFileSelect={handleFileSelect} />
          </div>

          {/* Target Column Display */}
          {fileStats?.targetColumn && (
            <div className="p-4 bg-blue-50 rounded-md">
              <p className="font-medium">Detected Target Column: <span className="text-blue-600">{fileStats.targetColumn}</span></p>
              <p className="text-sm text-gray-600 mt-1">
                This is our best guess for what you're trying to predict. You can specify a different task below.
              </p>
            </div>
          )}

          {/* Optional Task Description */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">2. Specify Your Task (Optional)</h3>
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder={fileStats?.targetColumn 
                ? `Suggest algorithms to predict ${fileStats.targetColumn}` 
                : "E.g., Classify customers into segments based on purchasing behavior"}
              className="w-full p-2 border rounded-md min-h-[80px]"
            />
          </div>

          {/* Action Button */}
          <Button
            onClick={handleSuggestAlgorithms}
            disabled={!fileStats || isSuggesting || !hasApiKey}
            className="w-full"
          >
            {isSuggesting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Lightbulb className="h-4 w-4 mr-2" />
                Suggest Best Algorithms
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Algorithm Suggestions */}
      {algorithmSuggestions && (
        <Card>
          <CardHeader>
            <CardTitle>Recommended Algorithms</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 p-4 rounded-md prose max-w-none">
              <pre className="whitespace-pre-wrap" style={{ fontFamily: 'inherit' }}>
                {algorithmSuggestions}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AlgorithmSelection;
