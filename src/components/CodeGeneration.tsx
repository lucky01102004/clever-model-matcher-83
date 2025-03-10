
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { FileUpload } from "./FileUpload";
import { generateCode, suggestAlgorithms } from "@/services/codeGenerationService";
import { Loader2, Code, Lightbulb, ArrowLeft } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Link } from "react-router-dom";

const CodeGeneration = () => {
  const [task, setTask] = useState<string>("");
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [algorithmSuggestions, setAlgorithmSuggestions] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isSuggesting, setIsSuggesting] = useState<boolean>(false);
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
  } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<string>("generate");
  const [savedAlgorithmData, setSavedAlgorithmData] = useState<any>(null);

  useEffect(() => {
    // Load algorithm data from localStorage if available
    const storedData = localStorage.getItem('codeGenerationData');
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData);
        setSavedAlgorithmData(parsedData);
        
        // Pre-populate data
        if (parsedData.dataDescription) {
          setFileStats(parsedData.dataDescription);
        }
        
        // Create task description based on algorithm
        if (parsedData.algorithmName && parsedData.targetColumn) {
          let taskDescription = '';
          
          switch (parsedData.algorithmType) {
            case 'classification':
              taskDescription = `Build a ${parsedData.algorithmName} model to classify ${parsedData.targetColumn}`;
              break;
            case 'regression':
              taskDescription = `Create a ${parsedData.algorithmName} model to predict ${parsedData.targetColumn}`;
              break;
            case 'clustering':
              taskDescription = `Perform ${parsedData.algorithmName} clustering on the dataset`;
              break;
            case 'dimensionality_reduction':
              taskDescription = `Apply ${parsedData.algorithmName} for dimensionality reduction`;
              break;
            default:
              taskDescription = `Use ${parsedData.algorithmName} on the dataset`;
          }
          
          setTask(taskDescription);
        }
      } catch (error) {
        console.error("Error parsing stored algorithm data:", error);
      }
    }
  }, []);

  const handleFileSelect = (file: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
  }) => {
    setSelectedFile(file);
    if (stats) {
      setFileStats(stats);
    } else {
      setFileStats(null);
    }
  };

  const handleGenerate = async () => {
    if (!fileStats || !task.trim()) {
      toast.error("Please upload a dataset and describe your task");
      return;
    }
    
    setIsGenerating(true);
    try {
      const code = await generateCode({
        dataDescription: fileStats,
        task: task.trim()
      });
      setGeneratedCode(code);
      setActiveTab("generate");
    } catch (error) {
      console.error("Error generating code:", error);
      toast.error("Failed to generate code. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSuggestAlgorithms = async () => {
    if (!fileStats || !task.trim()) {
      toast.error("Please upload a dataset and describe your task");
      return;
    }
    
    setIsSuggesting(true);
    try {
      const suggestions = await suggestAlgorithms({
        dataDescription: fileStats,
        task: task.trim()
      });
      setAlgorithmSuggestions(suggestions);
      setActiveTab("suggest");
    } catch (error) {
      console.error("Error suggesting algorithms:", error);
      toast.error("Failed to suggest algorithms. Please try again.");
    } finally {
      setIsSuggesting(false);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <Link to="/" className="flex items-center text-primary hover:underline">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Home
        </Link>
      </div>
      
      <h1 className="text-3xl font-bold mb-2">Code Generation</h1>
      <p className="text-gray-600 mb-8">Generate Python code for your data analysis and machine learning tasks</p>
      
      <div className="w-full max-w-4xl mx-auto space-y-8">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">Python Code Generator</CardTitle>
            <CardDescription>
              {savedAlgorithmData ? 
                `Generate code for ${savedAlgorithmData.algorithmName} on your dataset` :
                "Upload a dataset and describe what you want to analyze"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Upload - Show only if no algorithm data */}
            {!savedAlgorithmData && (
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">1. Upload Your Dataset</h3>
                <FileUpload onFileSelect={handleFileSelect} />
              </div>
            )}
            
            {/* Dataset info if using saved algorithm */}
            {savedAlgorithmData && fileStats && (
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Selected Dataset & Algorithm</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
                  <div>
                    <p className="text-sm text-gray-500">Dataset Size</p>
                    <p className="font-medium">{fileStats.rows} rows × {fileStats.columns} columns</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Algorithm</p>
                    <p className="font-medium">{savedAlgorithmData.algorithmName}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Target Column</p>
                    <p className="font-medium">{savedAlgorithmData.targetColumn || "None"}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Task Description */}
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">
                {savedAlgorithmData ? "2. Review Task Description" : "2. Describe Your Task"}
              </h3>
              <Label htmlFor="task">What do you want to do with this data?</Label>
              <Textarea
                id="task"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="E.g., Create a visualization showing the relationship between column A and column B, or build a prediction model for column C based on other features."
                className="min-h-[100px]"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col space-y-3 sm:flex-row sm:space-y-0 sm:space-x-3">
              <Button
                onClick={handleSuggestAlgorithms}
                disabled={!fileStats || !task.trim() || isSuggesting || isGenerating}
                className="flex-1"
                variant="outline"
              >
                {isSuggesting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Lightbulb className="h-4 w-4 mr-2" />
                    Suggest Algorithms
                  </>
                )}
              </Button>
              
              <Button
                onClick={handleGenerate}
                disabled={!fileStats || !task.trim() || isGenerating || isSuggesting}
                className="flex-1"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Code className="h-4 w-4 mr-2" />
                    Generate Python Code
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Output Tabs */}
        {(generatedCode || algorithmSuggestions) && (
          <Card>
            <CardHeader>
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="suggest" disabled={!algorithmSuggestions}>
                    Algorithm Suggestions
                  </TabsTrigger>
                  <TabsTrigger value="generate" disabled={!generatedCode}>
                    Generated Code
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsContent value="suggest" className="mt-0">
                  {algorithmSuggestions && (
                    <div className="bg-gray-50 p-4 rounded-md prose max-w-none">
                      <pre className="whitespace-pre-wrap" style={{ fontFamily: 'inherit' }}>
                        {algorithmSuggestions}
                      </pre>
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="generate" className="mt-0">
                  {generatedCode && (
                    <pre className="bg-gray-100 p-4 rounded-md overflow-auto max-h-[500px] text-sm">
                      <code>{generatedCode}</code>
                    </pre>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
            <CardFooter>
              <Button
                variant="outline"
                onClick={() => {
                  const textToCopy = activeTab === "generate" ? generatedCode : algorithmSuggestions;
                  navigator.clipboard.writeText(textToCopy);
                  toast.success(activeTab === "generate" ? "Code copied to clipboard" : "Algorithm suggestions copied to clipboard");
                }}
              >
                Copy to Clipboard
              </Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </div>
  );
};

export default CodeGeneration;
