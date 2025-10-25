import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ContentRenderer } from './content-renderer';
import type { Content } from '@/lib/content-loader';

// Mock react-syntax-highlighter to avoid module resolution issues in tests
vi.mock('react-syntax-highlighter', () => ({
  Prism: ({ children, language }: any) => (
    <pre>
      <code className={`language-${language}`}>{children}</code>
    </pre>
  ),
}));

vi.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({
  oneDark: {},
}));

describe('ContentRenderer Component', () => {
  describe('Text Sections', () => {
    it('should render a text section', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Hello, this is a test paragraph.',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      expect(screen.getByText(/Hello, this is a test paragraph/i)).toBeInTheDocument();
    });

    it('should render multiple text sections', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'First paragraph',
            },
            {
              type: 'text',
              content: 'Second paragraph',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      expect(screen.getByText(/First paragraph/i)).toBeInTheDocument();
      expect(screen.getByText(/Second paragraph/i)).toBeInTheDocument();
    });

    it('should render markdown in text sections', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: '# Heading\n\n**Bold text**',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      // Check that markdown is rendered (heading and bold)
      const heading = screen.getByRole('heading', { level: 1 });
      expect(heading).toHaveTextContent('Heading');

      expect(screen.getByText(/Bold text/i)).toBeInTheDocument();
    });
  });

  describe('Code Sections', () => {
    it('should render a code section with language', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'code',
              language: 'python',
              content: 'print("Hello, World!")',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      // Check that code content is present
      expect(screen.getByText(/print\("Hello, World!"\)/i)).toBeInTheDocument();

      // Check that language is displayed
      expect(screen.getByText(/python/i)).toBeInTheDocument();
    });

    it('should render code in a code block', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'code',
              language: 'javascript',
              content: 'const x = 5;',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      // Code should be in a <pre> or <code> element
      const codeElement = screen.getByText(/const x = 5;/i).closest('code');
      expect(codeElement).toBeInTheDocument();
    });

    it('should support multiple programming languages', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'code',
              language: 'python',
              content: 'x = 5',
            },
            {
              type: 'code',
              language: 'javascript',
              content: 'const x = 5;',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      expect(screen.getByText(/python/i)).toBeInTheDocument();
      expect(screen.getByText(/javascript/i)).toBeInTheDocument();
    });
  });

  describe('Interactive Sections', () => {
    it('should render an interactive section placeholder', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'exercise',
        content: {
          sections: [
            {
              type: 'interactive',
              exerciseType: 'python-editor',
              initialCode: 'x = 5',
              solution: 'x = 10',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      // For now, check that interactive section shows a placeholder
      // (We'll implement full Python editor later)
      expect(screen.getByText(/interactive exercise/i)).toBeInTheDocument();
    });

    it('should display initial code for interactive section', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'exercise',
        content: {
          sections: [
            {
              type: 'interactive',
              exerciseType: 'python-editor',
              initialCode: '# Write your code here\nx = 5',
              solution: 'x = 10',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      expect(screen.getByText(/Write your code here/i)).toBeInTheDocument();
    });
  });

  describe('Mixed Content', () => {
    it('should render mixed content types in order', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Introduction text',
            },
            {
              type: 'code',
              language: 'python',
              content: 'print("Hello")',
            },
            {
              type: 'text',
              content: 'Explanation text',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      expect(screen.getByText(/Introduction text/i)).toBeInTheDocument();
      expect(screen.getByText(/print\("Hello"\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Explanation text/i)).toBeInTheDocument();
    });
  });

  describe('Empty Content', () => {
    it('should handle empty sections array', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [],
        },
      };

      render(<ContentRenderer content={content} />);

      // Should not crash, might show empty state
      expect(screen.queryByText(/no content/i)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Test content',
            },
          ],
        },
      };

      const { container } = render(<ContentRenderer content={content} />);

      // Check for semantic HTML structure
      expect(container.querySelector('article')).toBeInTheDocument();
    });

    it('should have code sections with proper semantics', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'code',
              language: 'python',
              content: 'x = 5',
            },
          ],
        },
      };

      render(<ContentRenderer content={content} />);

      const codeElement = screen.getByText(/x = 5/i).closest('code');
      expect(codeElement).toBeInTheDocument();
    });
  });

  describe('Custom className', () => {
    it('should accept and apply custom className', () => {
      const content: Content = {
        id: 'test',
        title: 'Test Content',
        description: 'Test description',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Test',
            },
          ],
        },
      };

      const { container } = render(
        <ContentRenderer content={content} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
