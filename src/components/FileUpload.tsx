
import { useState } from "react";
import { Upload, X, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface FileUploadProps {
  onFileSelect: (file: File | null, stats?: { 
    rows: number; 
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    targetColumn?: string;
  }) => void;
}

export const FileUpload = ({ onFileSelect }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const detectTargetColumn = (data: any[]): string | undefined => {
    if (!data || data.length === 0) return undefined;
    
    // Common target column names
    const targetKeywords = ['target', 'class', 'label', 'y', 'outcome', 'result', 'category'];
    
    const columnNames = Object.keys(data[0]);
    
    // 1. First look for exact matches in column names
    for (const keyword of targetKeywords) {
      const match = columnNames.find(col => 
        col.toLowerCase() === keyword.toLowerCase()
      );
      if (match) return match;
    }
    
    // 2. Then look for columns containing target keywords
    for (const keyword of targetKeywords) {
      const match = columnNames.find(col => 
        col.toLowerCase().includes(keyword.toLowerCase())
      );
      if (match) return match;
    }
    
    // 3. Check for binary/categorical columns with few unique values (common for target)
    const uniqueValueCounts = columnNames.map(col => {
      const values = new Set(data.map(row => row[col]));
      return { column: col, uniqueCount: values.size };
    });
    
    // Sort columns by number of unique values (ascending)
    uniqueValueCounts.sort((a, b) => a.uniqueCount - b.uniqueCount);
    
    // If we find a column with few unique values (2-10), it's likely the target
    const likelyTarget = uniqueValueCounts.find(col => 
      col.uniqueCount >= 2 && col.uniqueCount <= 10
    );
    
    return likelyTarget?.column;
  };

  const processCSV = (content: string, file: File) => {
    Papa.parse(content, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        if (results.data && results.data.length > 0) {
          const columnNames = Object.keys(results.data[0] || {});
          const targetColumn = detectTargetColumn(results.data);
          
          const stats = {
            rows: results.data.length,
            columns: columnNames.length,
            columnNames: columnNames,
            dataSample: results.data.slice(0, 5) as Record<string, string>[],
            targetColumn
          };
          onFileSelect(file, stats);
        } else {
          setError("Error: The file appears to be empty or malformatted.");
          setFile(null);
          onFileSelect(null);
        }
      },
      error: (error) => {
        console.error("Error parsing CSV:", error);
        setError("Error parsing CSV file. Please check the file format.");
        setFile(null);
        onFileSelect(null);
      }
    });
  };

  const processExcel = (arrayBuffer: ArrayBuffer, file: File) => {
    try {
      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];
      
      // Convert to JSON
      const data = XLSX.utils.sheet_to_json(worksheet);
      
      if (data && data.length > 0) {
        const columnNames = Object.keys(data[0] || {});
        const targetColumn = detectTargetColumn(data);
        
        const stats = {
          rows: data.length,
          columns: columnNames.length,
          columnNames: columnNames,
          dataSample: data.slice(0, 5) as Record<string, string>[],
          targetColumn
        };
        onFileSelect(file, stats);
      } else {
        setError("Error: The Excel file appears to be empty or malformatted.");
        setFile(null);
        onFileSelect(null);
      }
    } catch (err) {
      console.error("Error processing Excel file:", err);
      setError("Error processing Excel file. Please check the file format.");
      setFile(null);
      onFileSelect(null);
    }
  };

  const analyzeFile = (file: File) => {
    setError(null);
    const reader = new FileReader();
    
    if (file.name.endsWith('.csv')) {
      reader.onload = (event) => {
        if (event.target?.result) {
          processCSV(event.target.result as string, file);
        }
      };
      reader.readAsText(file);
    } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
      reader.onload = (event) => {
        if (event.target?.result) {
          processExcel(event.target.result as ArrayBuffer, file);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      setError("Unsupported file format. Please upload CSV or Excel files only.");
      setFile(null);
      onFileSelect(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);

    const droppedFile = e.dataTransfer.files[0];
    if (isValidFile(droppedFile)) {
      setFile(droppedFile);
      analyzeFile(droppedFile);
    } else {
      setError("Unsupported file format. Please upload CSV or Excel files only.");
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      analyzeFile(selectedFile);
    } else if (selectedFile) {
      setError("Unsupported file format. Please upload CSV or Excel files only.");
    }
  };

  const isValidFile = (file: File) => {
    // Accept both CSV and Excel files
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  const removeFile = () => {
    setFile(null);
    setError(null);
    onFileSelect(null);
  };

  return (
    <div className="w-full">
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center ${
          isDragging
            ? "border-accent bg-accent/5"
            : "border-primary-200 hover:border-accent"
        } transition-colors`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleFileInput}
        />
        <div className="space-y-4">
          <div className="flex justify-center">
            <Upload
              className={`h-12 w-12 ${
                isDragging ? "text-accent" : "text-primary-400"
              }`}
            />
          </div>
          <div>
            <p className="text-lg font-medium text-primary-900">
              Drag and drop your file here
            </p>
            <p className="text-sm text-primary-600">
              or click to browse (CSV and Excel files supported)
            </p>
          </div>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <AnimatePresence>
        {file && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 p-4 bg-primary-50 rounded-lg flex items-center justify-between"
          >
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-white rounded">
                <Upload className="h-4 w-4 text-primary-600" />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-medium text-primary-900">
                  {file.name}
                </span>
                <span className="text-xs text-primary-600">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
            </div>
            <button
              onClick={removeFile}
              className="p-1 hover:bg-primary-100 rounded-full transition-colors"
            >
              <X className="h-4 w-4 text-primary-600" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
