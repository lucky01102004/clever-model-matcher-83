
import React from 'react';
import { Link } from 'react-router-dom';
import { cn } from "@/lib/utils";

interface ModuleCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  to: string;
  className?: string;
}

const ModuleCard = ({ title, description, icon, to, className }: ModuleCardProps) => {
  return (
    <Link 
      to={to}
      className={cn(
        "block p-6 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all bg-white",
        "transform hover:-translate-y-1 duration-200",
        "hover:border-primary/50",
        className
      )}
    >
      <div className="flex flex-col h-full">
        <div className="flex items-center mb-4">
          <div className="w-12 h-12 flex items-center justify-center rounded-full bg-primary/10 text-primary mr-4">
            {icon}
          </div>
          <h3 className="text-xl font-semibold text-gray-900">{title}</h3>
        </div>
        <p className="text-gray-600 flex-grow">{description}</p>
      </div>
    </Link>
  );
};

export default ModuleCard;
