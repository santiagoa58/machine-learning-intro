'use client';

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface TabsProps {
  items: Array<{
    label: string;
    content: React.ReactNode;
  }>;
}

export function Tabs({ items }: TabsProps) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <Card className="my-6">
      <div className="border-b border-gray-200 dark:border-gray-800">
        <nav className="flex gap-4 px-6" aria-label="Tabs">
          {items.map((item, index) => (
            <button
              key={index}
              onClick={() => setActiveTab(index)}
              className={`
                py-4 px-1 border-b-2 font-medium text-sm transition-colors
                ${
                  activeTab === index
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>
      <CardContent className="pt-6">
        {items[activeTab].content}
      </CardContent>
    </Card>
  );
}
