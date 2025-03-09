
import React, { useState } from 'react';
import { ArrowLeft, Database } from 'lucide-react';
import { Link } from 'react-router-dom';
import { FileUpload } from '@/components/FileUpload';
import { Button } from '@/components/ui/button';

const UploadDataset = () => {
  const [fileStats, setFileStats] = useState<{
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  } | null>(null);
  
  const handleFileSelect = (_: File | null, stats?: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  }) => {
    if (stats) {
      setFileStats(stats);
    } else {
      setFileStats(null);
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
      
      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Upload File</h2>
          <FileUpload onFileSelect={handleFileSelect} />
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Dataset Preview</h2>
          
          {fileStats ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-sm text-gray-500">Rows</p>
                  <p className="text-lg font-medium">{fileStats.rows}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-sm text-gray-500">Columns</p>
                  <p className="text-lg font-medium">{fileStats.columns}</p>
                </div>
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
                            className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            {column}
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
                              className="px-3 py-2 whitespace-nowrap text-sm text-gray-500"
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
                <Button asChild className="flex-1">
                  <Link to="/data-analysis">
                    Proceed to Analysis
                  </Link>
                </Button>
                <Button asChild className="flex-1" variant="outline">
                  <Link to="/algorithm-selection">
                    Skip to Algorithm Selection
                  </Link>
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center text-gray-500">
              <Database className="h-12 w-12 mb-4 opacity-40" />
              <p>Upload a dataset to view the preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadDataset;
