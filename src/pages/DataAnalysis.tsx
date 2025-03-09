
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";

const DataAnalysis = () => {
  return (
    <div className="container mx-auto py-8 px-4">
      <Link to="/">
        <Button variant="outline" className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Home
        </Button>
      </Link>
      
      <Card>
        <CardHeader>
          <CardTitle>Data Analysis</CardTitle>
          <CardDescription>
            Get insights from your data with automated analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">
            This module is under development. Check back soon for updates!
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default DataAnalysis;
