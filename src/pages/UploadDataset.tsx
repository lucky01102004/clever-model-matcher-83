
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Database } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { FileUpload } from '@/components/FileUpload';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from 'sonner';

const UploadDataset = () => {
  const navigate = useNavigate();
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
    statistics?: {
      mean: Record<string, number>;
      median: Record<string, number>;
      mode: Record<string, any>;
      stdDev: Record<string, number>;
      nullCount: Record<string, number>;
      correlationMatrix?: Record<string, Record<string, number>>;
      classDistribution?: Record<string, number>;
    }
  } | null>(null);
  
  useEffect(() => {
    // Check if we already have data in localStorage
    const savedData = localStorage.getItem('uploadedDataset');
    if (savedData) {
      try {
        const parsedData = JSON.parse(savedData);
        setFileStats(parsedData);
      } catch (error) {
        console.error("Error parsing saved data:", error);
      }
    }
  }, []);
  
  const handleFileSelect = (_: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
    statistics?: {
      mean: Record<string, number>;
      median: Record<string, number>;
      mode: Record<string, any>;
      stdDev: Record<string, number>;
      nullCount: Record<string, number>;
      correlationMatrix?: Record<string, Record<string, number>>;
      classDistribution?: Record<string, number>;
    }
  }) => {
    if (stats) {
      setFileStats(stats);
      // Save the data to localStorage for other modules to access
      localStorage.setItem('uploadedDataset', JSON.stringify(stats));
      toast.success("Dataset uploaded successfully");
    } else {
      setFileStats(null);
      localStorage.removeItem('uploadedDataset');
    }
  };

  const handleContinueToAnalysis = () => {
    if (fileStats) {
      navigate('/data-analysis');
    } else {
      toast.error("Please upload a dataset first");
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
      
      <h1 className="text-3xl font-bold mb-2">Upload Dataset</h1>
      <p className="text-gray-600 mb-8">Upload your CSV or Excel files for analysis</p>
      
      <div className="grid grid-cols-1 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Upload File</CardTitle>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} />
          </CardContent>
        </Card>
        
        {fileStats && (
          <Card>
            <CardHeader>
              <CardTitle>Dataset Preview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gray-50 p-3 rounded">
                    <p className="text-sm text-gray-500">Rows</p>
                    <p className="text-lg font-medium">{fileStats.rows}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded">
                    <p className="text-sm text-gray-500">Columns</p>
                    <p className="text-lg font-medium">{fileStats.columns}</p>
                  </div>
                  {fileStats.suggestedTarget && (
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-500">Suggested Target</p>
                      <p className="text-lg font-medium">{fileStats.suggestedTarget}</p>
                    </div>
                  )}
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Data Sample</h3>
                  <div className="bg-gray-50 p-3 rounded overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-100">
                        <tr>
                          {fileStats.columnNames.map(column => (
                            <th 
                              key={column}
                              className={`px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                                column === fileStats.suggestedTarget ? "bg-primary/20" : ""
                              }`}
                            >
                              {column} {column === fileStats.suggestedTarget ? "(Target)" : ""}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {fileStats.dataSample.map((row, i) => (
                          <tr key={i}>
                            {fileStats.columnNames.map(column => (
                              <td 
                                key={`${i}-${column}`}
                                className={`px-3 py-2 whitespace-nowrap text-sm text-gray-500 ${
                                  column === fileStats.suggestedTarget ? "bg-primary/10" : ""
                                }`}
                              >
                                {String(row[column] || '')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <Button className="flex-1" onClick={handleContinueToAnalysis}>
                    Proceed to Analysis
                  </Button>
                  <Button className="flex-1" variant="outline" asChild>
                    <Link to="/algorithm-selection">
                      Skip to Algorithm Selection
                    </Link>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        
        {!fileStats && (
          <Card>
            <CardContent className="py-8">
              <div className="flex flex-col items-center justify-center text-center text-gray-500">
                <Database className="h-12 w-12 mb-4 opacity-40" />
                <p>Upload a dataset to view the preview</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default UploadDataset;
