
import { useState } from "react";
import { Upload, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { toast } from "sonner";

interface FileUploadProps {
  onFileSelect: (file: File | null, stats?: { 
    rows: number; 
    columns: number;
    columnNames: string[];
    dataSample: Record<string, string>[];
    suggestedTarget?: string;
  }) => void;
}

export const FileUpload = ({ onFileSelect }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const detectPossibleTargetColumn = (data: any[], columnNames: string[]) => {
    // Common target column names
    const targetKeywords = [
      "target", "label", "class", "output", "result", "outcome", "dependent", 
      "prediction", "category", "classification", "y", "status"
    ];
    
    // Look for columns with target-like names
    for (const keyword of targetKeywords) {
      const matchingColumn = columnNames.find(col => 
        col.toLowerCase().includes(keyword.toLowerCase())
      );
      if (matchingColumn) return matchingColumn;
    }
    
    // If no matching name, check for binary/categorical columns with few unique values
    const columnStats = columnNames.map(col => {
      const values = data.map(row => row[col]);
      const uniqueValues = new Set(values);
      return { 
        column: col, 
        uniqueCount: uniqueValues.size,
        numeric: !isNaN(Number(values[0]))
      };
    });
    
    // Sort to prioritize categorical columns with few unique values
    const sortedColumns = columnStats
      .filter(stat => stat.uniqueCount > 1 && stat.uniqueCount <= 10)
      .sort((a, b) => a.uniqueCount - b.uniqueCount);
    
    return sortedColumns.length > 0 ? sortedColumns[0].column : columnNames[columnNames.length - 1];
  };

  const processExcelFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = e.target?.result;
        const workbook = XLSX.read(data, { type: 'binary' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet);
        
        if (jsonData.length === 0) {
          toast.error("Excel file is empty or invalid format");
          return;
        }
        
        const columnNames = Object.keys(jsonData[0] || {});
        const suggestedTarget = detectPossibleTargetColumn(jsonData, columnNames);
        
        const stats = {
          rows: jsonData.length,
          columns: columnNames.length,
          columnNames: columnNames,
          dataSample: jsonData.slice(0, 5) as Record<string, string>[],
          suggestedTarget
        };
        
        onFileSelect(file, stats);
      } catch (error) {
        console.error("Error processing Excel file:", error);
        toast.error("Failed to process Excel file");
      }
    };
    reader.onerror = () => {
      toast.error("Error reading file");
    };
    reader.readAsBinaryString(file);
  };

  const processCsvFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        Papa.parse(event.target.result as string, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            if (results.data.length === 0 || !results.data[0]) {
              toast.error("CSV file is empty or invalid format");
              return;
            }
            
            const columnNames = Object.keys(results.data[0] || {});
            const suggestedTarget = detectPossibleTargetColumn(results.data, columnNames);
            
            const stats = {
              rows: results.data.length,
              columns: columnNames.length,
              columnNames: columnNames,
              dataSample: results.data.slice(0, 5) as Record<string, string>[],
              suggestedTarget
            };
            
            onFileSelect(file, stats);
          },
          error: (error) => {
            console.error("Error parsing CSV:", error);
            toast.error("Failed to process CSV file");
          }
        });
      }
    };
    reader.onerror = () => {
      toast.error("Error reading file");
    };
    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (isValidFile(droppedFile)) {
      setFile(droppedFile);
      processFile(droppedFile);
    } else {
      toast.error("Please upload only CSV or Excel files");
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      processFile(selectedFile);
    } else if (selectedFile) {
      toast.error("Please upload only CSV or Excel files");
    }
  };

  const processFile = (file: File) => {
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    
    if (fileExtension === 'csv') {
      processCsvFile(file);
    } else if (fileExtension === 'xlsx' || fileExtension === 'xls') {
      processExcelFile(file);
    }
  };

  const isValidFile = (file: File) => {
    const validTypes = ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    return validTypes.includes(file.type) || fileExtension === 'csv' || fileExtension === 'xlsx' || fileExtension === 'xls';
  };

  const removeFile = () => {
    setFile(null);
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
              or click to browse (CSV or Excel files only)
            </p>
          </div>
        </div>
      </div>

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
