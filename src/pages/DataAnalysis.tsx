
import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const DataAnalysis = () => {
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
      
      <div className="bg-white p-6 rounded-lg shadow-sm">
        <div className="text-center py-10">
          <h2 className="text-xl font-medium mb-2">Upload a dataset first</h2>
          <p className="text-gray-500">
            Go to the Upload Dataset module to get started with data analysis
          </p>
          <Link
            to="/upload-dataset"
            className="inline-block mt-4 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90"
          >
            Go to Upload Dataset
          </Link>
        </div>
      </div>
    </div>
  );
};

export default DataAnalysis;
