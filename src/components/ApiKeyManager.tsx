
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Check, Key, Eye, EyeOff } from "lucide-react";
import { saveApiKey, getApiKey, isApiKeySet } from "@/services/codeGenerationService";
import { toast } from "@/hooks/use-toast";

const ApiKeyManager = () => {
  const [apiKey, setApiKey] = useState("");
  const [isVisible, setIsVisible] = useState(false);
  const [keyStatus, setKeyStatus] = useState<boolean>(false);

  useEffect(() => {
    setKeyStatus(isApiKeySet());
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim()) {
      saveApiKey(apiKey.trim());
      setApiKey("");
      setKeyStatus(true);
      toast({
        title: "API Key Saved",
        description: "Your Gemini API key has been saved",
      });
    }
  };

  const handleReset = () => {
    saveApiKey("");
    setKeyStatus(false);
    toast({
      title: "API Key Removed",
      description: "Your Gemini API key has been removed",
    });
  };

  if (keyStatus) {
    return (
      <Card className="w-full border-green-100 bg-green-50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center">
            <Check className="h-4 w-4 mr-2 text-green-600" />
            Gemini API Key Set
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleReset}
            className="text-xs"
          >
            Reset API Key
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg">Gemini API Key Required</CardTitle>
        <CardDescription>
          Enter your Gemini API key to enable code generation
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="apiKey">API Key</Label>
            <div className="flex">
              <Input
                id="apiKey"
                type={isVisible ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="AIzaSy..."
                className="flex-1"
              />
              <Button 
                type="button" 
                variant="outline" 
                size="icon"
                onClick={() => setIsVisible(!isVisible)} 
                className="ml-2"
              >
                {isVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          <Button type="submit" className="w-full">
            <Key className="h-4 w-4 mr-2" />
            Save API Key
          </Button>
        </form>
      </CardContent>
      <CardFooter className="text-xs text-gray-500 pt-0">
        Your API key is stored only in your browser's local storage.
      </CardFooter>
    </Card>
  );
};

export default ApiKeyManager;
