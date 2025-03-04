
// This service handles code generation using the OpenAI API

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

    // Free API for code generation (OpenAI-like)
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Remove the Authorization header to avoid 401 errors in the demo
        // In a real application, you would include your API key here
        // 'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful data science assistant that generates Python code.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.2,
        max_tokens: 1000,
      }),
    });

    // For demo purposes, we'll simulate a response since we can't make actual API calls
    // In a real application, you would parse the actual response
    const mockResponse = `
\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Display basic information
print("Dataset Info:")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\\nSummary Statistics:")
print(df.describe())

# Visualize data
plt.figure(figsize=(12, 6))
for i, column in enumerate(df.select_dtypes(include=['number']).columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    if i >= 5:
        break
plt.tight_layout()
plt.savefig('data_distribution.png')
plt.show()

# Correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()
\`\`\`
`;

    return mockResponse;
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

