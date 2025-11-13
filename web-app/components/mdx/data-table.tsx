import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface DataTableProps {
  data: Array<Record<string, string | number>>;
  columns?: string[];
  caption?: string;
  title?: string;
  description?: string;
}

export function DataTable({ data, columns, caption, title, description }: DataTableProps) {
  // If columns not provided, use all keys from first row
  const tableColumns = columns || (data.length > 0 ? Object.keys(data[0]) : []);

  const tableContent = (
    <Table>
      {caption && <TableCaption>{caption}</TableCaption>}
      <TableHeader>
        <TableRow>
          {tableColumns.map((column) => (
            <TableHead key={column} className="font-semibold capitalize">
              {column.replace(/([A-Z])/g, ' $1').trim()}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((row, index) => (
          <TableRow key={index}>
            {tableColumns.map((column) => (
              <TableCell key={column}>{row[column]}</TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
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
        <CardContent>{tableContent}</CardContent>
      </Card>
    );
  }

  return <div className="my-6">{tableContent}</div>;
}
