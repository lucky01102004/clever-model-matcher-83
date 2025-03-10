
import React from 'react';
import ModuleCard from '@/components/ModuleCard';
import { Upload, BarChart3, BrainCircuit, Code, ArrowRight } from 'lucide-react';
import { Card } from '@/components/ui/card';

const Index = () => {
  return (
    <div className="container mx-auto py-12 px-4">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Data Analysis & Code Generation Platform</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Upload your datasets, perform analysis, select algorithms, and generate code with our integrated platform
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
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
      
      {/* How it Works Section */}
      <div className="mt-16 mb-12">
        <h2 className="text-3xl font-bold text-center mb-8">How It Works</h2>
        <Card className="p-6 shadow-md bg-white/50">
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row gap-4 items-start">
              <div className="flex items-center justify-center bg-primary/10 text-primary rounded-full w-10 h-10 mt-1 shrink-0">1</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Upload Your Dataset</h3>
                <p className="text-gray-700">Start by uploading your CSV or Excel file in the Upload Dataset module. This will be the foundation for all subsequent analysis and code generation. Your data is stored locally in your browser for privacy.</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-4 items-start">
              <div className="flex items-center justify-center bg-primary/10 text-primary rounded-full w-10 h-10 mt-1 shrink-0">2</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Analyze Your Data</h3>
                <p className="text-gray-700">Once uploaded, navigate to the Data Analysis module to explore your dataset. View statistical summaries, correlations, and visualizations to understand patterns and relationships in your data.</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-4 items-start">
              <div className="flex items-center justify-center bg-primary/10 text-primary rounded-full w-10 h-10 mt-1 shrink-0">3</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Select Appropriate Algorithms</h3>
                <p className="text-gray-700">Based on your data characteristics and analysis, the Algorithm Selection module will suggest suitable machine learning algorithms. You can customize parameters based on your specific requirements.</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-4 items-start">
              <div className="flex items-center justify-center bg-primary/10 text-primary rounded-full w-10 h-10 mt-1 shrink-0">4</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Generate Executable Code</h3>
                <p className="text-gray-700">Finally, use the Code Generation module to create ready-to-use Python code implementing your selected algorithms on your dataset. Copy the code to use in your own environment or projects.</p>
              </div>
            </div>
          </div>
          
          <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-100">
            <h4 className="text-lg font-semibold text-blue-700 mb-2">Pro Tips:</h4>
            <ul className="list-disc pl-5 space-y-2 text-gray-700">
              <li>Ensure your dataset is clean and properly formatted for best results</li>
              <li>Take time to explore the data analysis visualizations to better understand your data</li>
              <li>Compare different algorithms to find the most suitable for your specific use case</li>
              <li>For complex datasets, consider pre-processing your data before uploading</li>
            </ul>
          </div>
          
          <div className="mt-8 text-center">
            <a href="/upload-dataset" className="inline-flex items-center text-primary hover:text-primary/80 font-medium">
              Get Started by Uploading Your Dataset
              <ArrowRight className="ml-2 h-4 w-4" />
            </a>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Index;
