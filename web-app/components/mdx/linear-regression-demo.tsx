"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { formatNumber } from "@/lib/utils";
import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface DataPoint {
  x: number;
  y: number;
}

interface LinearRegressionDemoProps {
  data: DataPoint[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
  description?: string;
  showPrediction?: boolean;
  predictionValue?: number;
  height?: number;
}

// Calculate linear regression coefficients
function calculateLinearRegression(data: DataPoint[]): {
  slope: number;
  intercept: number;
  r2: number;
  predict: (x: number) => number;
} {
  const n = data.length;
  const sumX = data.reduce((sum, point) => sum + point.x, 0);
  const sumY = data.reduce((sum, point) => sum + point.y, 0);
  const sumXY = data.reduce((sum, point) => sum + point.x * point.y, 0);
  const sumX2 = data.reduce((sum, point) => sum + point.x * point.x, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Calculate R²
  const meanY = sumY / n;
  const ssTotal = data.reduce(
    (sum, point) => sum + Math.pow(point.y - meanY, 2),
    0
  );
  const ssResidual = data.reduce(
    (sum, point) => sum + Math.pow(point.y - (slope * point.x + intercept), 2),
    0
  );
  const r2 = 1 - ssResidual / ssTotal;

  const predict = (x: number) => slope * x + intercept;

  return { slope, intercept, r2, predict };
}

export function LinearRegressionDemo({
  data,
  xLabel = "X",
  yLabel = "Y",
  title,
  description,
  showPrediction = false,
  predictionValue,
  height = 400,
}: LinearRegressionDemoProps) {
  const formatter = Intl.NumberFormat("en", { useGrouping: true });
  const { slope, intercept, r2, predict } = useMemo(
    () => calculateLinearRegression(data),
    [data]
  );

  // Create line data points
  const minX = Math.min(...data.map((p) => p.x));
  const maxX = Math.max(...data.map((p) => p.x));
  const lineData = [
    { x: minX, y: predict(minX) },
    { x: maxX, y: predict(maxX) },
  ];

  // Prediction point
  const predictionPoint =
    showPrediction && predictionValue !== undefined
      ? { x: predictionValue, y: predict(predictionValue) }
      : null;

  return (
    <Card className="my-6">
      {(title || description) && (
        <CardHeader>
          {title && <CardTitle>{title}</CardTitle>}
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent className="space-y-4">
        <ResponsiveContainer width="100%" height={height} className="mb-6">
          <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-gray-200 dark:stroke-gray-700"
            />
            <XAxis
              type="number"
              dataKey="x"
              name={xLabel}
              label={{ value: xLabel, position: "insideBottom", offset: -10 }}
              className="text-gray-600 dark:text-gray-400"
            />
            <YAxis
              type="number"
              dataKey="y"
              name={yLabel}
              label={{ value: yLabel, angle: -90, position: "insideLeft" }}
              className="text-gray-600 dark:text-gray-400"
            />
            <Tooltip
              contentStyle={{
                borderRadius: "0.5rem",
              }}
              formatter={(value) => formatNumber(formatter, value)}
              payloadUniqBy={true}
              cursor={{ strokeDasharray: "3 3" }}
            />
            <Legend
              align="left"
              verticalAlign="bottom"
              wrapperStyle={{ bottom: 0 }}
            />
            {/* Actual data points */}
            <Scatter
              name="Actual Data"
              data={data}
              fill="#3b82f6"
              shape="circle"
            />

            {/* Regression line */}
            <Scatter
              name="Fitted Line"
              data={lineData}
              fill="#ef4444"
              line
              shape={"cross"}
              lineType="joint"
            />

            {/* Prediction point */}
            {predictionPoint && (
              <Scatter
                name={`Prediction (${xLabel}=${predictionValue})`}
                data={[predictionPoint]}
                fill="#10b981"
                shape="star"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>

        {/* Model statistics */}
        <div className="font-semibold text-sm">Model Parameters:</div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm font-mono">
          <div>
            <span className="text-gray-600 dark:text-gray-400">Slope:</span>{" "}
            <span className="font-semibold">{slope.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">Intercept:</span>{" "}
            <span className="font-semibold">{intercept.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">R² Score:</span>{" "}
            <span className="font-semibold">{r2.toFixed(3)}</span>
          </div>
        </div>
        <div className="text-sm mt-2">
          <span className="text-gray-600 dark:text-gray-400">Equation:</span>{" "}
          <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-pink-600 dark:text-pink-400 not-prose">
            {yLabel} = {slope.toFixed(2)} × {xLabel} + {intercept.toFixed(2)}
          </code>
        </div>
        {predictionPoint && (
          <div className="text-sm mt-2">
            <span className="text-gray-600 dark:text-gray-400">
              Prediction:
            </span>{" "}
            <span className="font-semibold text-green-600 dark:text-green-400">
              When {xLabel} = {predictionValue}, predicted {yLabel} ={" "}
              {predictionPoint.y.toFixed(2)}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
