import { describe, it, expect, beforeEach, vi } from 'vitest';
import { z } from 'zod';
import { ContentLoader, ContentSchema, type FileSystem } from './content-loader';

describe('ContentLoader', () => {
  let mockFileSystem: FileSystem;
  let mockReadFile: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockReadFile = vi.fn();
    mockFileSystem = {
      readFile: mockReadFile as FileSystem['readFile'],
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

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);
      const result = await loader.loadContent('test-content');

      expect(result).toEqual(mockContent);
      expect(mockReadFile).toHaveBeenCalledWith('/content/test-content.json', 'utf-8');
    });

    it('should throw error when file not found', async () => {
      mockReadFile.mockRejectedValue(
        Object.assign(new Error('File not found'), { code: 'ENOENT' })
      );

      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('nonexistent')).rejects.toThrow(
        'Content not found: nonexistent'
      );
    });

    it('should throw error when JSON is invalid', async () => {
      mockReadFile.mockResolvedValue('{ invalid json }');

      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('bad-json')).rejects.toThrow('Invalid JSON');
    });

    it('should throw error when content fails schema validation', async () => {
      const invalidContent = {
        id: 'test',
        // missing required fields
      };

      mockReadFile.mockResolvedValue(JSON.stringify(invalidContent));

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

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/custom/path', mockFileSystem);
      await loader.loadContent('custom');

      expect(mockReadFile).toHaveBeenCalledWith('/custom/path/custom.json', 'utf-8');
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

      mockReadFile
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

      mockReadFile
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

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

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

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);

      // First load
      await loader.loadContent('cached');
      // Second load (should use cache)
      await loader.loadContent('cached');

      // File should only be read once
      expect(mockReadFile).toHaveBeenCalledTimes(1);
    });

    it('should clear cache when requested', async () => {
      const mockContent = {
        id: 'test',
        title: 'Test',
        description: 'Test',
        type: 'lesson',
        content: { sections: [] },
      };

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);

      await loader.loadContent('test');
      loader.clearCache();
      await loader.loadContent('test');

      // File should be read twice (once before clear, once after)
      expect(mockReadFile).toHaveBeenCalledTimes(2);
    });
  });

  describe('Security: Path Traversal Prevention', () => {
    it('should reject ID containing forward slash', async () => {
      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('../../etc/passwd')).rejects.toThrow(
        'Invalid content ID: ../../etc/passwd (path traversal not allowed)'
      );
      expect(mockReadFile).not.toHaveBeenCalled();
    });

    it('should reject ID containing backslash', async () => {
      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('..\\..\\windows\\system32')).rejects.toThrow(
        'Invalid content ID: ..\\..\\windows\\system32 (path traversal not allowed)'
      );
      expect(mockReadFile).not.toHaveBeenCalled();
    });

    it('should reject ID containing parent directory reference', async () => {
      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('../secret')).rejects.toThrow(
        'Invalid content ID: ../secret (path traversal not allowed)'
      );
      expect(mockReadFile).not.toHaveBeenCalled();
    });

    it('should reject ID with special characters', async () => {
      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('test<script>')).rejects.toThrow(
        'Invalid content ID: test<script> (only alphanumeric, underscore, and hyphen allowed)'
      );
      expect(mockReadFile).not.toHaveBeenCalled();
    });

    it('should reject ID with null byte', async () => {
      const loader = new ContentLoader('/content', mockFileSystem);

      await expect(loader.loadContent('test\0file')).rejects.toThrow(
        'Invalid content ID: test\0file (only alphanumeric, underscore, and hyphen allowed)'
      );
      expect(mockReadFile).not.toHaveBeenCalled();
    });

    it('should allow valid alphanumeric IDs with hyphens and underscores', async () => {
      const mockContent = {
        id: 'valid-id_123',
        title: 'Valid',
        description: 'Valid content',
        type: 'lesson',
        content: { sections: [] },
      };

      mockReadFile.mockResolvedValue(JSON.stringify(mockContent));

      const loader = new ContentLoader('/content', mockFileSystem);
      const result = await loader.loadContent('valid-id_123');

      expect(result).toEqual(mockContent);
      expect(mockReadFile).toHaveBeenCalledWith('/content/valid-id_123.json', 'utf-8');
    });
  });
});
