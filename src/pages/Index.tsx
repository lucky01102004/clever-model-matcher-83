
import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Code, LineChart, BrainCircuit, FileInput } from "lucide-react";
import { ModuleBanner } from "@/components/ModuleBanner";

const Index = () => {
  const [greeting, setGreeting] = useState<string>("");

  useEffect(() => {
    const hours = new Date().getHours();
    let greetingText = "";

    if (hours < 12) {
      greetingText = "Good morning";
    } else if (hours < 18) {
      greetingText = "Good afternoon";
    } else {
      greetingText = "Good evening";
    }

    setGreeting(greetingText);
  }, []);

  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-3xl">{greeting}, welcome to AI Model Hub</CardTitle>
          <CardDescription>
            Explore our AI tools and services to help you with your data science projects
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">
            Select one of the modules below to get started, or explore our documentation 
            to learn more about our platform capabilities.
          </p>
          <div className="mt-4 flex space-x-4">
            <Button variant="default">View Documentation</Button>
            <Button variant="outline">Explore Examples</Button>
          </div>
        </CardContent>
      </Card>

      <h2 className="text-2xl font-bold mb-4">Available Modules</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <ModuleBanner
          title="Code Generation"
          description="Generate code snippets based on natural language prompts"
          icon={<Code size={24} />}
          to="/code-generator"
        />
        
        <ModuleBanner
          title="Algorithm Selection"
          description="Choose the right algorithm for your machine learning task"
          icon={<BrainCircuit size={24} />}
          to="/algorithm-selection"
        />
        
        <ModuleBanner
          title="Data Analysis"
          description="Get insights from your data with automated analysis"
          icon={<LineChart size={24} />}
          to="/data-analysis"
        />
        
        <ModuleBanner
          title="Dataset Management"
          description="Upload and manage your datasets for AI model training"
          icon={<FileInput size={24} />}
          to="/dataset-management"
        />
      </div>

      <Separator className="my-8" />

      <div className="text-center text-gray-500 text-sm">
        <p>© {new Date().getFullYear()} AI Model Hub. All rights reserved.</p>
      </div>
    </div>
  );
};

export default Index;
