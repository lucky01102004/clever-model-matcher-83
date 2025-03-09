
import { ReactNode } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

interface ModuleBannerProps {
  title: string;
  description: string;
  icon: ReactNode;
  to: string;
}

export const ModuleBanner = ({ title, description, icon, to }: ModuleBannerProps) => {
  return (
    <Link to={to} className="block w-full">
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className="rounded-lg bg-white p-6 shadow-md border border-gray-200 hover:shadow-lg transition-all duration-200 cursor-pointer"
      >
        <div className="flex items-start space-x-4">
          <div className="rounded-full bg-primary-100 p-3 text-primary-700">
            {icon}
          </div>
          <div>
            <h3 className="text-xl font-medium text-primary-900">{title}</h3>
            <p className="mt-1 text-gray-600">{description}</p>
          </div>
        </div>
      </motion.div>
    </Link>
  );
};
