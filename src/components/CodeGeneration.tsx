
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { FileUpload } from "./FileUpload";
import { generateCode, suggestAlgorithms } from "@/services/codeGenerationService";
import { Loader2, Code, Lightbulb } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
  } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<string>("generate");

  const handleFileSelect = (file: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  }) => {
    setSelectedFile(file);
    if (stats) {
      setFileStats(stats);
    } else {
      setFileStats(null);
    }
  };

  const handleGenerate = async () => {
    if (!fileStats || !task.trim()) return;
    
    setIsGenerating(true);
    try {
      const code = await generateCode({
        dataDescription: fileStats,
        task: task.trim()
      });
      setGeneratedCode(code);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSuggestAlgorithms = async () => {
    if (!fileStats || !task.trim()) return;
    
    setIsSuggesting(true);
    try {
      const suggestions = await suggestAlgorithms({
        dataDescription: fileStats,
        task: task.trim()
      });
      setAlgorithmSuggestions(suggestions);
      setActiveTab("suggest");
    } finally {
      setIsSuggesting(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Python Code Generator</CardTitle>
          <CardDescription>
            Upload a CSV file and describe the analysis you want to perform
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">1. Upload Your Dataset</h3>
            <FileUpload onFileSelect={handleFileSelect} />
          </div>

          {/* Task Description */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">2. Describe Your Task</h3>
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
          </CardContent>
          <CardFooter>
            <Button
              variant="outline"
              onClick={() => {
                const textToCopy = activeTab === "generate" ? generatedCode : algorithmSuggestions;
                navigator.clipboard.writeText(textToCopy);
                toast({
                  title: "Copied to clipboard",
                  description: activeTab === "generate" ? "Code copied to clipboard" : "Algorithm suggestions copied to clipboard",
                });
              }}
            >
              Copy to Clipboard
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
};

export default CodeGeneration;
