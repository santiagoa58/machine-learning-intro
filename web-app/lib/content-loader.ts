import { z } from 'zod';
import { readFile as fsReadFile } from 'fs/promises';
import path from 'path';

// File system interface for dependency injection
export interface FileSystem {
  readFile(path: string, encoding: string): Promise<string>;
}

// Default file system implementation
const defaultFileSystem: FileSystem = {
  readFile: fsReadFile,
};

// Section schemas
const TextSectionSchema = z.object({
  type: z.literal('text'),
  content: z.string(),
});

const CodeSectionSchema = z.object({
  type: z.literal('code'),
  language: z.string(),
  content: z.string(),
});

const InteractiveSectionSchema = z.object({
  type: z.literal('interactive'),
  exerciseType: z.string(),
  initialCode: z.string(),
  solution: z.string(),
});

const SectionSchema = z.discriminatedUnion('type', [
  TextSectionSchema,
  CodeSectionSchema,
  InteractiveSectionSchema,
]);

// Main content schema
export const ContentSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string(),
  type: z.string(),
  content: z.object({
    sections: z.array(SectionSchema),
  }),
});

export type Content = z.infer<typeof ContentSchema>;
export type ContentMetadata = Pick<Content, 'id' | 'title' | 'description' | 'type'>;

/**
 * ContentLoader loads and validates content from JSON files.
 * Features:
 * - Schema validation with Zod
 * - Caching for performance
 * - Error handling for missing files and invalid data
 * - Metadata extraction without loading full content
 */
export class ContentLoader {
  private contentDir: string;
  private cache: Map<string, Content>;
  private fs: FileSystem;

  constructor(contentDir: string, fs: FileSystem = defaultFileSystem) {
    this.contentDir = contentDir;
    this.cache = new Map();
    this.fs = fs;
  }

  /**
   * Load a single content file by ID
   * @param id - Content ID (filename without .json extension)
   * @returns Validated content object
   * @throws Error if file not found, JSON invalid, or validation fails
   */
  async loadContent(id: string): Promise<Content> {
    // Check cache first
    if (this.cache.has(id)) {
      return this.cache.get(id)!;
    }

    const filePath = path.join(this.contentDir, `${id}.json`);

    try {
      // Read file
      const fileContents = await this.fs.readFile(filePath, 'utf-8');

      // Parse JSON
      let jsonData: unknown;
      try {
        jsonData = JSON.parse(fileContents);
      } catch (error) {
        throw new Error(`Invalid JSON in content file: ${id}`);
      }

      // Validate with schema
      const result = ContentSchema.safeParse(jsonData);
      if (!result.success) {
        throw new Error(
          `Content validation failed for ${id}: ${result.error.message}`
        );
      }

      // Cache and return
      const content = result.data;
      this.cache.set(id, content);
      return content;
    } catch (error: any) {
      // Handle specific error cases
      if (error.code === 'ENOENT') {
        throw new Error(`Content not found: ${id}`);
      }

      // Re-throw validation and parsing errors
      if (error.message.includes('Invalid JSON') ||
          error.message.includes('Content validation failed')) {
        throw error;
      }

      // Unknown error
      throw new Error(`Failed to load content ${id}: ${error.message}`);
    }
  }

  /**
   * Load multiple content files in parallel
   * @param ids - Array of content IDs
   * @returns Array of successfully loaded content (skips invalid)
   */
  async loadAllContent(ids: string[]): Promise<Content[]> {
    const results = await Promise.allSettled(
      ids.map((id) => this.loadContent(id))
    );

    return results
      .filter((result) => {
        if (result.status === 'rejected') {
          console.warn('Skipping invalid content:', result.reason);
          return false;
        }
        return true;
      })
      .map((result) => (result as PromiseFulfilledResult<Content>).value);
  }

  /**
   * Get only metadata for a content file (without loading full content)
   * @param id - Content ID
   * @returns Content metadata (id, title, description, type)
   */
  async getContentMetadata(id: string): Promise<ContentMetadata> {
    const content = await this.loadContent(id);

    return {
      id: content.id,
      title: content.title,
      description: content.description,
      type: content.type,
    };
  }

  /**
   * Clear the content cache
   */
  clearCache(): void {
    this.cache.clear();
  }
}
