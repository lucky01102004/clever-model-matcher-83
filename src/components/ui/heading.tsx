
import { cn } from "@/lib/utils";

interface HeadingProps {
  title: string;
  description?: string;
  className?: string;
  size?: "default" | "sm" | "lg";
}

export function Heading({
  title,
  description,
  className,
  size = "default"
}: HeadingProps) {
  const headerSize = {
    default: "text-3xl font-bold",
    sm: "text-2xl font-semibold",
    lg: "text-4xl font-extrabold"
  };

  const descriptionSize = {
    default: "text-muted-foreground",
    sm: "text-sm text-muted-foreground",
    lg: "text-lg text-muted-foreground"
  };

  return (
    <div className={cn("space-y-2", className)}>
      <h1 className={headerSize[size]}>{title}</h1>
      {description && <p className={descriptionSize[size]}>{description}</p>}
    </div>
  );
}
