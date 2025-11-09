import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function parseNumber(value: unknown) {
  try {
    return Number(value as string);
  } catch (e: unknown) {
    console.warn(
      `failed to parse value of type ${typeof value} to number: `,
      value,
      e
    );
  }
  return undefined;
}

export function parseNumbers<T = unknown>(...args: T[]) {
  const values = Array.isArray(args[0]) ? args.flat() : args;
  return values.map(parseNumber).filter((x) => x != null);
}

export function formatNumber<T = unknown>(
  formatter: Intl.NumberFormat,
  args: T
) {
  if (Array.isArray(args)) {
    return parseNumbers(args).map(formatter.format);
  }
  const value = parseNumber(args);
  return value != null ? formatter.format(value) : undefined;
}
