
import React from 'react';
import ModuleCard from '@/components/ModuleCard';
import { Upload, BarChart3, BrainCircuit, Code } from 'lucide-react';

const Index = () => {
  return (
    <div className="container mx-auto py-12 px-4">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Data Analysis & Code Generation Platform</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Upload your datasets, perform analysis, select algorithms, and generate code with our integrated platform
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ModuleCard
          title="Upload Dataset"
          description="Upload CSV or Excel files to start your data analysis journey"
          icon={<Upload className="h-6 w-6" />}
          to="/upload-dataset"
        />
        
        <ModuleCard
          title="Data Analysis"
          description="Explore and visualize your data to gain insights"
          icon={<BarChart3 className="h-6 w-6" />}
          to="/data-analysis"
        />
        
        <ModuleCard
          title="Algorithm Selection"
          description="Choose and configure algorithms for your data"
          icon={<BrainCircuit className="h-6 w-6" />}
          to="/algorithm-selection"
        />
        
        <ModuleCard
          title="Code Generation"
          description="Generate ready-to-use code for your selected algorithms"
          icon={<Code className="h-6 w-6" />}
          to="/code-generator"
        />
      </div>
    </div>
  );
};

export default Index;
