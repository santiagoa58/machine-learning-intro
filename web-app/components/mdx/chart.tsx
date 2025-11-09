"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { formatNumber } from "@/lib/utils";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ChartProps {
  data: Array<Record<string, string | number>>;
  type?: "line" | "bar" | "scatter";
  xKey: string;
  yKey: string | string[];
  title?: string;
  description?: string;
  height?: number;
  colors?: string[];
}

export function Chart({
  data,
  type = "line",
  xKey,
  yKey,
  title,
  description,
  height = 300,
  colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"],
}: ChartProps) {
  const yKeys = Array.isArray(yKey) ? yKey : [yKey];
  const formatter = Intl.NumberFormat("en", { useGrouping: true });

  const renderChart = () => {
    const commonProps = {
      data,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (type) {
      case "line":
        return (
          <LineChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-800"
            />
            <XAxis
              dataKey={xKey}
              className="text-gray-600 dark:text-gray-400"
            />
            <YAxis className="text-gray-600 dark:text-gray-400" />
            <Tooltip
              formatter={(value) => formatNumber(formatter, value)}
              payloadUniqBy
              contentStyle={{
                borderRadius: "0.5rem",
              }}
            />
            <Legend />
            {yKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={colors[index % colors.length]}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        );

      case "bar":
        return (
          <BarChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-800"
            />
            <XAxis
              dataKey={xKey}
              className="text-gray-600 dark:text-gray-400"
            />
            <YAxis className="text-gray-600 dark:text-gray-400" />
            <Tooltip
              payloadUniqBy
              contentStyle={{
                borderRadius: "0.5rem",
              }}
              formatter={(value) => formatNumber(formatter, value)}
            />
            <Legend />
            {yKeys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                fill={colors[index % colors.length]}
              />
            ))}
          </BarChart>
        );

      case "scatter":
        return (
          <ScatterChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-800"
            />
            <XAxis
              dataKey={xKey}
              type="number"
              className="text-gray-600 dark:text-gray-400"
            />
            <YAxis
              dataKey={yKeys[0]}
              type="number"
              className="text-gray-600 dark:text-gray-400"
            />
            <Tooltip
              payloadUniqBy
              formatter={(value) => formatNumber(formatter, value)}
              contentStyle={{
                borderRadius: "0.5rem",
              }}
              cursor={{ strokeDasharray: "3 3" }}
            />
            <Legend />
            <Scatter name={yKeys[0]} data={data} fill={colors[0]} />
          </ScatterChart>
        );
    }
  };

  const chartContent = (
    <ResponsiveContainer width="100%" height={height}>
      {renderChart()}
    </ResponsiveContainer>
  );

  if (title || description) {
    return (
      <Card className="my-6">
        {(title || description) && (
          <CardHeader>
            {title && <CardTitle>{title}</CardTitle>}
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
        )}
        <CardContent>{chartContent}</CardContent>
      </Card>
    );
  }

  return <div className="my-6">{chartContent}</div>;
}
