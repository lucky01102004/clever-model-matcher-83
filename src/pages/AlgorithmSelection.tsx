
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";

interface FileStats {
  rows: number;
  columns: number;
  columnNames: string[];
  dataSample: Record<string, string>[];
  targetColumn?: string;
}

const AlgorithmSelection = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileStats, setFileStats] = useState<FileStats | null>(null);

  const handleFileSelect = (selectedFile: File | null, stats?: FileStats) => {
    setFile(selectedFile);
    if (stats) {
      setFileStats(stats);
    } else {
      setFileStats(null);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <Link to="/">
        <Button variant="outline" className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Home
        </Button>
      </Link>
      
      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Algorithm Selection</CardTitle>
            <CardDescription>
              Upload your dataset and choose the algorithm for analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} />
          </CardContent>
        </Card>

        {fileStats && (
          <Card>
            <CardHeader>
              <CardTitle>Dataset Analysis</CardTitle>
              <CardDescription>
                Summary of the uploaded dataset
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium">Dataset Information</h3>
                  <ul className="mt-2 space-y-1">
                    <li>Rows: {fileStats.rows}</li>
                    <li>Columns: {fileStats.columns}</li>
                    {fileStats.targetColumn && (
                      <li className="font-medium text-primary-600">
                        Target Column: {fileStats.targetColumn}
                      </li>
                    )}
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium">Available Features</h3>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {fileStats.columnNames.map((column) => (
                      <span 
                        key={column} 
                        className={`px-3 py-1 rounded-full text-sm ${
                          column === fileStats.targetColumn 
                            ? "bg-primary-100 text-primary-800 font-medium" 
                            : "bg-gray-100"
                        }`}
                      >
                        {column}
                      </span>
                    ))}
                  </div>
                </div>

                {fileStats.dataSample.length > 0 && (
                  <div>
                    <h3 className="text-lg font-medium">Data Preview</h3>
                    <div className="mt-2 overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            {fileStats.columnNames.map((column) => (
                              <th 
                                key={column}
                                className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                                  column === fileStats.targetColumn ? "bg-primary-50" : ""
                                }`}
                              >
                                {column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {fileStats.dataSample.map((row, idx) => (
                            <tr key={idx}>
                              {fileStats.columnNames.map((column) => (
                                <td 
                                  key={`${idx}-${column}`}
                                  className={`px-6 py-4 whitespace-nowrap text-sm text-gray-500 ${
                                    column === fileStats.targetColumn ? "bg-primary-50" : ""
                                  }`}
                                >
                                  {row[column]?.toString() || ""}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AlgorithmSelection;
