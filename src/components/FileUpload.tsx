
import { useState } from "react";
import { Upload, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Papa from "papaparse";
import * as XLSX from "xlsx";

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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const detectTargetColumn = (data: any[]) => {
    // Common names for target/class columns
    const possibleTargetNames = ['target', 'class', 'label', 'outcome', 'result', 'y'];
    
    if (!data || data.length === 0) return undefined;
    
    const headers = Object.keys(data[0]);
    
    // First try to find exact matches in lowercase
    for (const name of possibleTargetNames) {
      const match = headers.find(h => h.toLowerCase() === name);
      if (match) return match;
    }
    
    // Then try to find columns containing these words
    for (const name of possibleTargetNames) {
      const match = headers.find(h => h.toLowerCase().includes(name));
      if (match) return match;
    }
    
    // If we can't find any matches, return the last column as a fallback
    return headers[headers.length - 1];
  };

  const analyzeFile = (file: File) => {
    const fileType = file.name.split('.').pop()?.toLowerCase();
    
    if (fileType === 'csv') {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          Papa.parse(event.target.result as string, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
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
            },
            error: (error) => {
              console.error("Error parsing CSV:", error);
            }
          });
        }
      };
      reader.readAsText(file);
    } else if (fileType === 'xlsx' || fileType === 'xls') {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          const data = new Uint8Array(event.target.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
          const jsonData = XLSX.utils.sheet_to_json(firstSheet);
          
          if (jsonData.length > 0) {
            const columnNames = Object.keys(jsonData[0] || {});
            const targetColumn = detectTargetColumn(jsonData);
            const stats = {
              rows: jsonData.length,
              columns: columnNames.length,
              columnNames: columnNames,
              dataSample: jsonData.slice(0, 5) as Record<string, string>[],
              targetColumn
            };
            onFileSelect(file, stats);
          }
        }
      };
      reader.readAsArrayBuffer(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (isValidFile(droppedFile)) {
      setFile(droppedFile);
      analyzeFile(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      analyzeFile(selectedFile);
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
              or click to browse (CSV and Excel files)
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
