
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUpload } from "@/components/FileUpload";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronLeft, Database } from "lucide-react";
import { Link } from "react-router-dom";
import { toast } from "@/hooks/use-toast";

const AlgorithmSelection = () => {
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    targetColumn?: string;
  } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);

  const handleFileSelect = (file: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    targetColumn?: string;
  }) => {
    setSelectedFile(file);
    if (stats) {
      setFileStats(stats);
      if (stats.targetColumn) {
        setSelectedTarget(stats.targetColumn);
        toast({
          title: "Target column detected",
          description: `We detected "${stats.targetColumn}" as the likely target column.`,
        });
      } else {
        setSelectedTarget(null);
        toast({
          title: "No target column detected",
          description: "Please select your target column manually.",
        });
      }
    } else {
      setFileStats(null);
      setSelectedTarget(null);
    }
  };

  const handleTargetChange = (value: string) => {
    setSelectedTarget(value);
  };

  const handleContinue = () => {
    if (selectedTarget && fileStats) {
      toast({
        title: "Ready to proceed",
        description: `Using "${selectedTarget}" as the target column for your analysis.`,
      });
      // Here you would navigate to the next step or process the data
    }
  };

  return (
    <div className="container mx-auto py-8 max-w-4xl">
      <div className="flex items-center mb-6">
        <Link to="/" className="mr-4">
          <Button variant="outline" size="icon">
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="text-3xl font-bold">Algorithm Selection</h1>
      </div>

      <div className="space-y-8">
        <Card>
          <CardHeader>
            <CardTitle>Upload Your Dataset</CardTitle>
            <CardDescription>
              Upload a CSV or Excel file to get started with algorithm selection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} />
          </CardContent>
        </Card>

        {fileStats && (
          <Card>
            <CardHeader>
              <CardTitle>Dataset Information</CardTitle>
              <CardDescription>
                We found {fileStats.rows} rows and {fileStats.columns} columns in your dataset
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="target-column">Select Target Column</Label>
                  <Select value={selectedTarget || ''} onValueChange={handleTargetChange}>
                    <SelectTrigger id="target-column" className="w-full">
                      <SelectValue placeholder="Select the target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {fileStats.columnNames.map((column) => (
                        <SelectItem 
                          key={column} 
                          value={column}
                          className={column === fileStats.targetColumn ? "bg-primary-50" : ""}
                        >
                          {column} {column === fileStats.targetColumn ? "(Detected)" : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-2">Data Preview</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          {fileStats.columnNames.map((column) => (
                            <th
                              key={column}
                              className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                                column === selectedTarget ? "bg-primary-100" : ""
                              }`}
                            >
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {fileStats.dataSample.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            {fileStats.columnNames.map((column) => (
                              <td
                                key={`${rowIndex}-${column}`}
                                className={`px-6 py-4 whitespace-nowrap text-sm text-gray-500 ${
                                  column === selectedTarget ? "bg-primary-50" : ""
                                }`}
                              >
                                {row[column] !== undefined ? String(row[column]) : ""}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <Button 
                className="w-full" 
                onClick={handleContinue}
                disabled={!selectedTarget}
              >
                <Database className="mr-2 h-4 w-4" />
                Continue with Analysis
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AlgorithmSelection;
