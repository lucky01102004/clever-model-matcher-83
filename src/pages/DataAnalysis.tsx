import React, { useState, useEffect } from 'react';
import { ArrowLeft, Database, Upload, BarChart } from 'lucide-react';
import { Link } from 'react-router-dom';
import { FileUpload } from '@/components/FileUpload';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const DataAnalysis = () => {
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
    } else {
      setFileStats(null);
    }
  };

  // Generate distribution chart data
  const getDistributionData = () => {
    if (!fileStats?.statistics?.classDistribution) return [];
    
    return Object.entries(fileStats.statistics.classDistribution).map(([label, count]) => ({
      name: label,
      value: count
    }));
  };
  
  // Generate null counts chart data 
  const getNullCountsData = () => {
    if (!fileStats?.statistics?.nullCount) return [];
    
    return Object.entries(fileStats.statistics.nullCount)
      .filter(([_, count]) => count > 0)
      .map(([column, count]) => ({
        name: column,
        value: count
      }));
  };
  
  // Format correlation matrix for heatmap
  const getCorrelationData = () => {
    if (!fileStats?.statistics?.correlationMatrix) return [];
    
    const correlations = fileStats.statistics.correlationMatrix;
    const columns = Object.keys(correlations);
    
    return columns.flatMap(col1 => 
      columns.map(col2 => ({
        x: col1,
        y: col2,
        value: correlations[col1][col2] || 0
      }))
    );
  };
  
  // Generate colors for pie chart
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FF6B6B', '#6B88FF'];

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <Link to="/" className="flex items-center text-primary hover:underline">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Home
        </Link>
      </div>
      
      <h1 className="text-3xl font-bold mb-2">Data Analysis</h1>
      <p className="text-gray-600 mb-8">Analyze your dataset and get insights</p>
      
      <div className="grid grid-cols-1 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Upload a Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} />
          </CardContent>
        </Card>
        
        {fileStats ? (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Dataset Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                      <p className="text-sm text-gray-500">Target Column</p>
                      <p className="text-lg font-medium">{fileStats.suggestedTarget}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            <Tabs defaultValue="statistics" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="statistics">Statistics</TabsTrigger>
                <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
                <TabsTrigger value="correlations">Correlations</TabsTrigger>
              </TabsList>
              
              <TabsContent value="statistics" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Statistical Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-100">
                          <tr>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Column</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mean</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Median</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Std. Dev</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mode</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Null Count</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {fileStats.columnNames.map(column => (
                            <tr key={column} className={column === fileStats.suggestedTarget ? "bg-primary/5" : ""}>
                              <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900">
                                {column} {column === fileStats.suggestedTarget ? "(Target)" : ""}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                {fileStats.statistics?.mean[column] !== undefined 
                                  ? fileStats.statistics.mean[column].toFixed(2) 
                                  : "N/A"}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                {fileStats.statistics?.median[column] !== undefined 
                                  ? fileStats.statistics.median[column].toFixed(2) 
                                  : "N/A"}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                {fileStats.statistics?.stdDev[column] !== undefined 
                                  ? fileStats.statistics.stdDev[column].toFixed(2) 
                                  : "N/A"}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                {fileStats.statistics?.mode[column] 
                                  ? fileStats.statistics.mode[column].slice(0, 3).join(", ") 
                                  : "N/A"}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                {fileStats.statistics?.nullCount[column] || 0}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="visualizations" className="mt-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {fileStats.statistics?.classDistribution && Object.keys(fileStats.statistics.classDistribution).length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Class Distribution</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-72">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={getDistributionData()}
                                cx="50%"
                                cy="50%"
                                labelLine={true}
                                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {getDistributionData().map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                              </Pie>
                              <Tooltip />
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                  
                  {getNullCountsData().length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Null Values by Column</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-72">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsBarChart
                              data={getNullCountsData()}
                              layout="vertical"
                              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis type="number" />
                              <YAxis type="category" dataKey="name" width={100} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="value" fill="#8884d8" name="Null Count" />
                            </RechartsBarChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </TabsContent>
              
              <TabsContent value="correlations" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Correlation Matrix</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {fileStats.statistics?.correlationMatrix && Object.keys(fileStats.statistics.correlationMatrix).length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="min-w-full border border-gray-200">
                          <thead>
                            <tr>
                              <th className="px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Column
                              </th>
                              {Object.keys(fileStats.statistics.correlationMatrix).map(column => (
                                <th key={column} className="px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(fileStats.statistics.correlationMatrix).map(([row, columns]) => (
                              <tr key={row}>
                                <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 border-b border-gray-200 bg-gray-50">
                                  {row}
                                </td>
                                {Object.entries(fileStats.statistics.correlationMatrix).map(([column]) => (
                                  <td 
                                    key={`${row}-${column}`} 
                                    className="px-3 py-2 whitespace-nowrap text-sm text-center border-b border-gray-200"
                                    style={{
                                      backgroundColor: columns[column] >= 0.8 ? 'rgba(52, 211, 153, 0.2)' : 
                                                       columns[column] >= 0.5 ? 'rgba(96, 165, 250, 0.2)' :
                                                       columns[column] <= -0.5 ? 'rgba(248, 113, 113, 0.2)' : 
                                                       'transparent'
                                    }}
                                  >
                                    {columns[column] !== undefined ? columns[column].toFixed(2) : "N/A"}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500">
                        No correlation data available. The dataset may not have enough numeric columns.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
            
            <div className="flex justify-center space-x-4">
              <Button asChild>
                <Link to="/algorithm-selection">
                  Continue to Algorithm Selection
                </Link>
              </Button>
            </div>
          </>
        ) : (
          <Card>
            <CardContent className="py-10">
              <div className="text-center">
                <BarChart className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                <h2 className="text-xl font-medium mb-2">Upload a dataset first</h2>
                <p className="text-gray-500 mb-6">
                  Upload a CSV or Excel file to start analyzing your data
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default DataAnalysis;
