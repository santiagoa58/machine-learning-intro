import { describe, it, expect, beforeEach, vi } from 'vitest';
import { z } from 'zod';
import { ContentLoader, ContentSchema, type FileSystem } from './content-loader';

describe('ContentLoader', () => {
  let mockFileSystem: FileSystem;

  beforeEach(() => {
    mockFileSystem = {
      readFile: vi.fn(),
    };
  });

  describe('Schema Validation', () => {
    it('should export ContentSchema', () => {
      expect(ContentSchema).toBeDefined();
      expect(ContentSchema).toBeInstanceOf(z.ZodType);
    });

    it('should validate content with all required fields', () => {
      const validContent = {
        id: 'intro-to-ml',
        title: 'Introduction to Machine Learning',
        description: 'Learn the basics of ML',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Welcome to machine learning!',
            },
          ],
        },
      };

      const result = ContentSchema.safeParse(validContent);
      expect(result.success).toBe(true);
    });

    it('should reject content missing required fields', () => {
      const invalidContent = {
        id: 'test',
        title: 'Test',
        // missing description, type, and content
      };

      const result = ContentSchema.safeParse(invalidContent);
      expect(result.success).toBe(false);
    });

    it('should support text sections', () => {
      const content = {
        id: 'test',
        title: 'Test',
        description: 'Test desc',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Some text content',
            },
          ],
        },
      };

      const result = ContentSchema.safeParse(content);
      expect(result.success).toBe(true);
    });

    it('should support code sections with language', () => {
      const content = {
        id: 'test',
        title: 'Test',
        description: 'Test desc',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'code',
              language: 'python',
              content: 'print("Hello")',
            },
          ],
        },
      };

      const result = ContentSchema.safeParse(content);
      expect(result.success).toBe(true);
    });

    it('should support interactive sections', () => {
      const content = {
        id: 'test',
        title: 'Test',
        description: 'Test desc',
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

      const result = ContentSchema.safeParse(content);
      expect(result.success).toBe(true);
    });
  });

  describe('loadContent', () => {
    it('should load and parse valid JSON content', async () => {
      const mockContent = {
        id: 'test-content',
        title: 'Test Content',
        description: 'A test',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Test text',
            },
          ],
        },
      };

      vi.mocked(mockFileSystem.readFile).mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);
      const result = await loader.loadContent('test-content');

      expect(result).toEqual(mockContent);
      expect(mockFileSystem.readFile).toHaveBeenCalledWith('/content/test-content.json', 'utf-8');
    });

    it('should throw error when file not found', async () => {
      mockFileSystem.readFile.mockRejectedValue(
        Object.assign(new Error('File not found'), { code: 'ENOENT' })
      );

      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('nonexistent')).rejects.toThrow(
        'Content not found: nonexistent'
      );
    });

    it('should throw error when JSON is invalid', async () => {
      mockFileSystem.readFile.mockResolvedValue('{ invalid json }');

      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('bad-json')).rejects.toThrow('Invalid JSON');
    });

    it('should throw error when content fails schema validation', async () => {
      const invalidContent = {
        id: 'test',
        // missing required fields
      };

      mockFileSystem.readFile.mockResolvedValue(JSON.stringify(invalidContent));

      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('invalid-schema')).rejects.toThrow(
        'Content validation failed'
      );
    });

    it('should use custom content directory', async () => {
      const mockContent = {
        id: 'custom',
        title: 'Custom',
        description: 'Custom content',
        type: 'lesson',
        content: { sections: [] },
      };

      mockFileSystem.readFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/custom/path', mockFileSystem);
      await loader.loadContent('custom');

      expect(mockFileSystem.readFile).toHaveBeenCalledWith('/custom/path/custom.json', 'utf-8');
    });
  });

  describe('loadAllContent', () => {
    it('should load multiple content files', async () => {
      const content1 = {
        id: 'content-1',
        title: 'Content 1',
        description: 'First',
        type: 'lesson',
        content: { sections: [] },
      };

      const content2 = {
        id: 'content-2',
        title: 'Content 2',
        description: 'Second',
        type: 'lesson',
        content: { sections: [] },
      };

      mockFileSystem.readFile
        .mockResolvedValueOnce(JSON.stringify(content1))
        .mockResolvedValueOnce(JSON.stringify(content2));

      const loader = new ContentLoader('/content', mockFileSystem);
      const results = await loader.loadAllContent(['content-1', 'content-2']);

      expect(results).toHaveLength(2);
      expect(results[0]).toEqual(content1);
      expect(results[1]).toEqual(content2);
    });

    it('should skip invalid content and continue loading', async () => {
      const validContent = {
        id: 'valid',
        title: 'Valid',
        description: 'Valid content',
        type: 'lesson',
        content: { sections: [] },
      };

      mockFileSystem.readFile
        .mockRejectedValueOnce(new Error('File not found'))
        .mockResolvedValueOnce(JSON.stringify(validContent));

      const loader = new ContentLoader('/content', mockFileSystem);
      const results = await loader.loadAllContent(['invalid', 'valid']);

      expect(results).toHaveLength(1);
      expect(results[0]).toEqual(validContent);
    });
  });

  describe('getContentMetadata', () => {
    it('should return only metadata fields', async () => {
      const mockContent = {
        id: 'test',
        title: 'Test Content',
        description: 'A test',
        type: 'lesson',
        content: {
          sections: [
            {
              type: 'text',
              content: 'Large content that should not be in metadata',
            },
          ],
        },
      };

      mockFileSystem.readFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);
      const metadata = await loader.getContentMetadata('test');

      expect(metadata).toEqual({
        id: 'test',
        title: 'Test Content',
        description: 'A test',
        type: 'lesson',
      });
      expect(metadata).not.toHaveProperty('content');
    });
  });

  describe('Caching', () => {
    it('should cache loaded content', async () => {
      const mockContent = {
        id: 'cached',
        title: 'Cached Content',
        description: 'Should be cached',
        type: 'lesson',
        content: { sections: [] },
      };

      mockFileSystem.readFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);

      // First load
      await loader.loadContent('cached');
      // Second load (should use cache)
      await loader.loadContent('cached');

      // File should only be read once
      expect(mockFileSystem.readFile).toHaveBeenCalledTimes(1);
    });

    it('should clear cache when requested', async () => {
      const mockContent = {
        id: 'test',
        title: 'Test',
        description: 'Test',
        type: 'lesson',
        content: { sections: [] },
      };

      mockFileSystem.readFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);

      await loader.loadContent('test');
      loader.clearCache();
      await loader.loadContent('test');

      // File should be read twice (once before clear, once after)
      expect(mockFileSystem.readFile).toHaveBeenCalledTimes(2);
    });
  });
});
