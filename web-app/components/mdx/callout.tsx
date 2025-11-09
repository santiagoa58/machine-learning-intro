import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  Lightbulb,
  XCircle,
} from "lucide-react";

interface CalloutProps {
  type?: "info" | "warning" | "success" | "error" | "tip";
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export function Callout({
  type = "info",
  title,
  children,
  className,
}: CalloutProps) {
  const icons = {
    info: Info,
    warning: AlertTriangle,
    success: CheckCircle2,
    error: XCircle,
    tip: Lightbulb,
  };

  const Icon = icons[type];

  const variantMap = {
    info: "default",
    warning: "destructive",
    success: "default",
    error: "destructive",
    tip: "default",
  } as const;

  const colorMap = {
    info: "text-blue-600 dark:text-blue-400",
    warning: "text-yellow-600 dark:text-yellow-400",
    success: "text-green-600 dark:text-green-400",
    error: "text-red-600 dark:text-red-400",
    tip: "text-purple-600 dark:text-purple-400",
  };

  return (
    <Alert variant={variantMap[type]} className={cn("my-6", className)}>
      <Icon className={`h-4 w-4 ${colorMap[type]}`} />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  );
}
