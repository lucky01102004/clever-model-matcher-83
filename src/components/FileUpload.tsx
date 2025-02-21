
import { useState } from "react";
import { Upload, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
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

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (isValidFile(droppedFile)) {
      setFile(droppedFile);
      onFileSelect(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const isValidFile = (file: File) => {
    const validTypes = ["text/csv", "application/vnd.ms-excel"];
    return validTypes.includes(file.type);
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
              or click to browse (CSV, Excel files only)
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
