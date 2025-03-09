
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Heading } from "@/components/ui/heading";
import { Link } from "react-router-dom";
import { Code, ChevronRight, BarChart } from "lucide-react";

const Index = () => {
  const modules = [
    {
      id: "code-generator",
      title: "Code Generation",
      description: "Generate Python code for your data analysis tasks",
      icon: Code,
      path: "/code-generator"
    },
    {
      id: "algorithm-selection",
      title: "Algorithm Selection",
      description: "Find the best machine learning algorithm for your dataset",
      icon: BarChart,
      path: "/algorithm-selection"
    }
  ];

  return (
    <div className="container mx-auto py-10">
      <div className="max-w-4xl mx-auto">
        <Heading
          title="AI Data Science Assistant"
          description="Your intelligent companion for data analysis and machine learning tasks"
          className="mb-8"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {modules.map((module) => (
            <Link to={module.path} key={module.id} className="no-underline">
              <Card className="h-full hover:shadow-md transition-shadow cursor-pointer">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <div className="space-y-1">
                    <CardTitle className="text-xl">{module.title}</CardTitle>
                    <CardDescription className="text-sm text-muted-foreground">
                      {module.description}
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-2">
                    <module.icon className="h-6 w-6 text-primary" />
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>

        <div className="mt-10">
          <Heading
            title="Get Started"
            description="Choose a module above to begin working with your data"
            size="sm"
            className="mb-4"
          />
          <div className="flex flex-col sm:flex-row gap-4">
            <Button asChild>
              <Link to="/algorithm-selection">Try Algorithm Selection</Link>
            </Button>
            <Button variant="outline" asChild>
              <Link to="/code-generator">Go to Code Generator</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
