
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Check, Key } from "lucide-react";

interface ApiKeyInputProps {
  onApiKeySave: (apiKey: string) => void;
  isApiKeySet: boolean;
}

const ApiKeyInput = ({ onApiKeySave, isApiKeySet }: ApiKeyInputProps) => {
  const [apiKey, setApiKey] = useState("");
  const [isVisible, setIsVisible] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim()) {
      onApiKeySave(apiKey.trim());
      setApiKey("");
    }
  };

  if (isApiKeySet) {
    return (
      <Card className="w-full border-green-100 bg-green-50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center">
            <Check className="h-4 w-4 mr-2 text-green-600" />
            API Key Set
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => onApiKeySave("")}
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
        <CardTitle className="text-lg">API Key Required</CardTitle>
        <CardDescription>
          Enter your OpenAI API key to enable code generation
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
                placeholder="sk-..."
                className="flex-1"
              />
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

export default ApiKeyInput;
