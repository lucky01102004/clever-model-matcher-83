
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Code, Lightbulb, LineChart } from "lucide-react";
import { Link } from "react-router-dom";

export default function Index() {
  return (
    <div className="container mx-auto py-10 px-4">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">ML Studio</h1>
        <p className="text-xl text-muted-foreground">
          Your all-in-one platform for machine learning experimentation
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Algorithm Selection Module */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="bg-blue-50">
            <Lightbulb className="h-10 w-10 text-blue-500 mb-2" />
            <CardTitle>Algorithm Selection</CardTitle>
            <CardDescription>
              Get AI-powered recommendations for the best algorithms for your dataset
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <p className="text-sm">
              Upload your dataset and get intelligent suggestions on which machine learning 
              algorithms will work best for your specific use case.
            </p>
          </CardContent>
          <CardFooter>
            <Link to="/algorithm-selection" className="w-full">
              <Button className="w-full">
                <span>Get Started</span>
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardFooter>
        </Card>

        {/* Code Generation Module */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="bg-green-50">
            <Code className="h-10 w-10 text-green-500 mb-2" />
            <CardTitle>Code Generation</CardTitle>
            <CardDescription>
              Generate Python code for your machine learning projects
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <p className="text-sm">
              Describe what you want to do with your data, and our AI will generate 
              the Python code needed to implement your analysis or model.
            </p>
          </CardContent>
          <CardFooter>
            <Link to="/code-generator" className="w-full">
              <Button className="w-full" variant="outline">
                <span>Generate Code</span>
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardFooter>
        </Card>

        {/* Visualization Module - Placeholder for future implementation */}
        <Card className="hover:shadow-lg transition-shadow opacity-75">
          <CardHeader className="bg-purple-50">
            <LineChart className="h-10 w-10 text-purple-500 mb-2" />
            <CardTitle>Data Visualization</CardTitle>
            <CardDescription>
              Explore and visualize your data (Coming Soon)
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <p className="text-sm">
              Create interactive visualizations to better understand your data patterns
              and relationships between variables.
            </p>
          </CardContent>
          <CardFooter>
            <Button className="w-full" disabled>
              <span>Coming Soon</span>
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}
