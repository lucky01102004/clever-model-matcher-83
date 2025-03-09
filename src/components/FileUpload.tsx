
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
    statistics?: {
      mean: Record<string, number>;
      median: Record<string, number>;
      mode: Record<string, any>;
      stdDev: Record<string, number>;
      nullCount: Record<string, number>;
      correlationMatrix?: Record<string, Record<string, number>>;
      classDistribution?: Record<string, number>;
    }
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

  const calculateStatistics = (data: any[], columnNames: string[]) => {
    const numericColumns = columnNames.filter(col => 
      !isNaN(Number(data[0][col])) && data[0][col] !== null && data[0][col] !== ""
    );
    
    const statistics = {
      mean: {} as Record<string, number>,
      median: {} as Record<string, number>,
      mode: {} as Record<string, any>,
      stdDev: {} as Record<string, number>,
      nullCount: {} as Record<string, number>,
      correlationMatrix: {} as Record<string, Record<string, number>>,
      classDistribution: {} as Record<string, number>
    };
    
    // Calculate stats for each column
    columnNames.forEach(col => {
      const values = data.map(row => row[col]);
      const numericValues = values.filter(val => 
        val !== null && val !== "" && !isNaN(Number(val))
      ).map(val => Number(val));
      
      // Count nulls
      statistics.nullCount[col] = values.filter(val => val === null || val === "").length;
      
      // For numeric columns, calculate other statistics
      if (numericValues.length > 0) {
        // Mean
        statistics.mean[col] = numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
        
        // Median
        const sorted = [...numericValues].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        statistics.median[col] = sorted.length % 2 === 0 
          ? (sorted[mid - 1] + sorted[mid]) / 2 
          : sorted[mid];
        
        // Standard Deviation
        const mean = statistics.mean[col];
        const squareDiffs = numericValues.map(val => Math.pow(val - mean, 2));
        statistics.stdDev[col] = Math.sqrt(
          squareDiffs.reduce((sum, val) => sum + val, 0) / numericValues.length
        );
      }
      
      // Mode (for all columns)
      const countMap: Record<string, number> = {};
      values.forEach(val => {
        const strVal = String(val);
        countMap[strVal] = (countMap[strVal] || 0) + 1;
      });
      
      let maxCount = 0;
      let modes: string[] = [];
      
      Object.entries(countMap).forEach(([val, count]) => {
        if (count > maxCount) {
          maxCount = count;
          modes = [val];
        } else if (count === maxCount) {
          modes.push(val);
        }
      });
      
      statistics.mode[col] = modes;
    });
    
    // Calculate correlation matrix for numeric columns
    if (numericColumns.length > 1) {
      numericColumns.forEach(col1 => {
        statistics.correlationMatrix[col1] = {};
        
        numericColumns.forEach(col2 => {
          if (col1 === col2) {
            statistics.correlationMatrix[col1][col2] = 1;
            return;
          }
          
          const values1 = data.map(row => Number(row[col1]));
          const values2 = data.map(row => Number(row[col2]));
          
          const mean1 = statistics.mean[col1];
          const mean2 = statistics.mean[col2];
          
          let numerator = 0;
          let denominator1 = 0;
          let denominator2 = 0;
          
          for (let i = 0; i < values1.length; i++) {
            const diff1 = values1[i] - mean1;
            const diff2 = values2[i] - mean2;
            
            numerator += diff1 * diff2;
            denominator1 += diff1 * diff1;
            denominator2 += diff2 * diff2;
          }
          
          const correlation = numerator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
          statistics.correlationMatrix[col1][col2] = isNaN(correlation) ? 0 : correlation;
        });
      });
    }
    
    // Calculate class distribution for potential target columns
    const suggestedTarget = detectPossibleTargetColumn(data, columnNames);
    if (suggestedTarget) {
      const targetValues = data.map(row => String(row[suggestedTarget]));
      const distribution: Record<string, number> = {};
      
      targetValues.forEach(val => {
        distribution[val] = (distribution[val] || 0) + 1;
      });
      
      statistics.classDistribution = distribution;
    }
    
    return statistics;
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
        const statistics = calculateStatistics(jsonData, columnNames);
        
        const stats = {
          rows: jsonData.length,
          columns: columnNames.length,
          columnNames: columnNames,
          dataSample: jsonData.slice(0, 5) as Record<string, string>[],
          suggestedTarget,
          statistics
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
            const statistics = calculateStatistics(results.data, columnNames);
            
            const stats = {
              rows: results.data.length,
              columns: columnNames.length,
              columnNames: columnNames,
              dataSample: results.data.slice(0, 5) as Record<string, string>[],
              suggestedTarget,
              statistics
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
