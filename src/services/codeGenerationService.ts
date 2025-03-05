
// This service handles code generation using the Gemini API

import { toast } from "@/components/ui/use-toast";

interface CodeGenerationRequest {
  dataDescription: {
    rows: number;
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
  };
  task: string;
}

const GEMINI_API_KEY = "AIzaSyC4dOG17Id7Gq80zUqaF-JOeq3crzqUFZw";
const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent";

export const suggestAlgorithms = async (request: CodeGenerationRequest): Promise<string> => {
  const { dataDescription, task } = request;
  
  try {
    // Prepare the prompt for algorithm suggestions
    const prompt = `
I have a CSV dataset with the following structure:
- ${dataDescription.rows} rows and ${dataDescription.columns} columns
- Column names: ${dataDescription.columnNames.join(', ')}
- Here's a sample of the data:
${JSON.stringify(dataDescription.dataSample, null, 2)}

Task: ${task}

Based on this dataset and task, please suggest the most suitable algorithm(s) or approach(es). 
For each suggestion, explain:
1. Why this algorithm is appropriate for this data and task
2. What data preprocessing might be required
3. Any potential limitations or considerations

Provide your response in a clear, concise format that can be easily displayed to users.
`;

    // Call the Gemini API
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1000,
        }
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API error:", errorData);
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the generated text from the response
    let suggestedAlgorithms = "";
    
    if (data.candidates && data.candidates.length > 0 && 
        data.candidates[0].content && 
        data.candidates[0].content.parts && 
        data.candidates[0].content.parts.length > 0) {
      suggestedAlgorithms = data.candidates[0].content.parts[0].text;
    } else {
      throw new Error("Unexpected API response format");
    }

    return suggestedAlgorithms;
  } catch (error) {
    console.error('Error suggesting algorithms:', error);
    toast({
      title: "Error",
      description: "Failed to suggest algorithms. Please try again.",
      variant: "destructive",
    });
    return "Failed to suggest algorithms. Please try again.";
  }
};

export const generateCode = async (request: CodeGenerationRequest): Promise<string> => {
  const { dataDescription, task } = request;
  
  try {
    // Prepare the prompt for the AI
    const prompt = `
I have a CSV dataset with the following structure:
- ${dataDescription.rows} rows and ${dataDescription.columns} columns
- Column names: ${dataDescription.columnNames.join(', ')}
- Here's a sample of the data:
${JSON.stringify(dataDescription.dataSample, null, 2)}

Task: ${task}

Please generate Python code to accomplish this task. Use pandas for data manipulation, matplotlib or seaborn for any visualizations, and scikit-learn for any machine learning tasks if needed.
`;

    // Call the Gemini API
    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1000,
        }
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API error:", errorData);
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the generated text from the response
    let generatedText = "";
    
    if (data.candidates && data.candidates.length > 0 && 
        data.candidates[0].content && 
        data.candidates[0].content.parts && 
        data.candidates[0].content.parts.length > 0) {
      generatedText = data.candidates[0].content.parts[0].text;
    } else {
      throw new Error("Unexpected API response format");
    }

    return generatedText;
  } catch (error) {
    console.error('Error generating code:', error);
    toast({
      title: "Error",
      description: "Failed to generate code. Please try again.",
      variant: "destructive",
    });
    return "# Error generating code. Please try again.";
  }
};
