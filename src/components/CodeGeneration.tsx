
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { FileUpload } from "./FileUpload";
import { generateCode } from "@/services/codeGenerationService";
import { Loader2, Code } from "lucide-react";

const CodeGeneration = () => {
  const [task, setTask] = useState<string>("");
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={!fileStats || !task.trim() || isGenerating}
            className="w-full"
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
        </CardContent>
      </Card>

      {/* Generated Code Output */}
      {generatedCode && (
        <Card>
          <CardHeader>
            <CardTitle>Generated Python Code</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-gray-100 p-4 rounded-md overflow-auto max-h-[500px] text-sm">
              <code>{generatedCode}</code>
            </pre>
          </CardContent>
          <CardFooter>
            <Button
              variant="outline"
              onClick={() => {
                navigator.clipboard.writeText(generatedCode);
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
